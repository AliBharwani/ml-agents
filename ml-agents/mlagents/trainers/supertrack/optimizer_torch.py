from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast
from mlagents.st_buffer import CharTypePrefix, CharTypeSuffix, PDTargetPrefix, PDTargetSuffix, STBuffer

from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr
import pdb
from mlagents.torch_utils import torch, nn, default_device
from mlagents.trainers.supertrack import world_model
from mlagents.trainers.torch_entities.encoders import Normalizer
import numpy as np
import pytorch3d.transforms as pyt
from mlagents.trainers.supertrack.supertrack_utils import  NUM_BONES, NUM_T_BONES, POLICY_INPUT_LEN, POLICY_OUTPUT_LEN, STSingleBufferKey, SupertrackUtils, nsys_profiler
from mlagents.trainers.supertrack.world_model import WorldModelNetwork
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.layers import Initialization, LinearEncoder
from mlagents.trainers.torch_entities.networks import Actor
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents_envs.logging_util import get_logger

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
)
from mlagents.trainers.torch_entities.agent_action import AgentAction

@attr.s(auto_attribs=True)
class SuperTrackSettings(OffPolicyHyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    num_epoch: int = 3
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    loss_weights_init_steps : int = 100
    min_loss_weight: float = 0.5
    max_loss_weight: float = 10.0
    gradient_clipping : float = -1
    policy_includes_global_data : bool = False

def hn(x):
    if isinstance(x, list):
        return any([y.isnan().any() for y in x])
    return torch.isnan(x).any()
import torch
import torch.nn as nn

class DynamicLoss(nn.Module):
    def __init__(self, min_loss_weight, max_loss_weight, num_iterations=100, alpha=0.01):
        super().__init__()
        # Initialize weights with some non-zero value to prevent division by zero
        self.wpos_loss = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.wrot_loss = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.wvel_loss = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.wrvel_loss = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
        # Parameters for Exponential Moving Average
        self.alpha = alpha  # Smoothing factor, determines update rate
        self.ema_losses = nn.Parameter(torch.ones(4), requires_grad=False)  # Initial EMA values
        self.total_losses = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.iteration_count = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.num_iterations = num_iterations
        self.initialized = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.min_loss_weight = nn.Parameter(torch.tensor(min_loss_weight), requires_grad=False)
        self.max_loss_weight = nn.Parameter(torch.tensor(max_loss_weight), requires_grad=False)

    def update_loss_weights(self, pos_loss, rot_loss, vel_loss, rvel_loss):
        if self.initialized:
            return
        with torch.no_grad():
            # Convert current losses to tensor
            current_losses = torch.tensor([pos_loss, rot_loss, vel_loss, rvel_loss])
            self.total_losses.data += current_losses
            # Update EMA of losses
            if self.iteration_count.item() == 0:
                self.ema_losses.data = current_losses
            else:
                self.ema_losses.data = self.alpha * current_losses + (1 - self.alpha) * self.ema_losses
            
            # Increment the iteration count
            self.iteration_count += 1

            if self.iteration_count < self.num_iterations:
                losses_to_use = self.ema_losses
            elif self.iteration_count >= self.num_iterations:
                losses_to_use = self.total_losses
                self.initialized.data = torch.tensor(True)

            total_avg_loss = losses_to_use.mean()
            weights = torch.clamp(total_avg_loss / losses_to_use, self.min_loss_weight, self.max_loss_weight)
            self.wpos_loss.data, self.wrot_loss.data, self.wvel_loss.data, self.wrvel_loss.data = weights

    def get_reweighted_losses(self, pos_loss, rot_loss, vel_loss, rvel_loss):
        self.update_loss_weights(pos_loss, rot_loss, vel_loss, rvel_loss)
        return self.wpos_loss * pos_loss, self.wrot_loss * rot_loss, self.wvel_loss * vel_loss, self.wrvel_loss * rvel_loss

    def to_str(self):
        return f"""
                Pos: {self.wpos_loss.item()}
                Rot: {self.wrot_loss.item()}
                Vel: {self.wvel_loss.item()}
                Rvel: {self.wrvel_loss.item()}"""

class TorchSuperTrackOptimizer(TorchOptimizer):
    dtime = 1 / 60

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.trainer_settings = trainer_settings
        self.hyperparameters: SuperTrackSettings = cast(
            SuperTrackSettings, trainer_settings.hyperparameters
        )
        self.wm_lr = trainer_settings.world_model_network_settings.learning_rate
        self.policy_lr = trainer_settings.policy_network_settings.learning_rate
        self.first_wm_update = True
        self.first_policy_update = True
        self.split_actor_devices = self.trainer_settings.use_pytorch_mp and default_device() == torch.device('cuda')
        self.actor_gpu = None
        self.policy_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.policy_lr)
        self.logger = get_logger(__name__)
        self.wm_loss_weights = DynamicLoss(self.hyperparameters.min_loss_weight, self.hyperparameters.max_loss_weight, num_iterations=self.hyperparameters.loss_weights_init_steps)
        self.policy_loss_weights = DynamicLoss(self.hyperparameters.min_loss_weight, self.hyperparameters.max_loss_weight, num_iterations=self.hyperparameters.loss_weights_init_steps)
        self.policy_includes_global_data = self.hyperparameters.policy_includes_global_data
                
    def _init_world_model(self):
        """
        Initializes the world model
        """
        self._world_model = WorldModelNetwork(
            self.trainer_settings.world_model_network_settings
        )
        self._world_model.to(default_device())
        self.world_model_optimzer = torch.optim.Adam(self._world_model.parameters(), lr=self.wm_lr)
        self._world_model.train()

    def export_world_model(self, output_filepath : str):
        final_export_path = self._world_model.export(output_filepath)
        self.logger.info(f"Exported {output_filepath}")
        return final_export_path

    def set_actor_gpu_to_optimizer(self):
        policy_optimizer_state = self.policy_optimizer.state_dict()
        self.policy_optimizer = torch.optim.Adam(self.actor_gpu.parameters(), lr=self.policy_lr)
        self.policy_optimizer.load_state_dict(policy_optimizer_state)
    
    def update_world_model(self, batch, raw_window_size: int) -> Dict[str, float]:
        self._world_model.train()
        # suffixes = [CharTypeSuffix.POSITION, CharTypeSuffix.ROTATION, CharTypeSuffix.VEL, CharTypeSuffix.RVEL, CharTypeSuffix.HEIGHT, CharTypeSuffix.UP_DIR]
        suffixes = [CharTypeSuffix.POSITION, CharTypeSuffix.ROTATION, CharTypeSuffix.VEL, CharTypeSuffix.RVEL]
        positions, rotations, vels, rot_vels = [batch[(CharTypePrefix.SIM, suffix)] for suffix in suffixes]
        kin_rot_t, kin_rvel_t = batch[(PDTargetPrefix.POST, PDTargetSuffix.ROT)], batch[(PDTargetPrefix.POST, PDTargetSuffix.RVEL)]
        # remove root bone from PDTargets 
        kin_rot_t, kin_rvel_t = kin_rot_t[:, :, 1:, :], kin_rvel_t[:, :, 1:, :] 
        kin_rot_t =  pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(kin_rot_t))
        # Remove root bone data before copying over
        # predicted_pos = positions[:, :, 1:, :].detach().clone() # shape [batch_size, window_size, num_bones, 3]
        # predicted_rots = rotations[:, :, 1:, :].detach().clone()
        # predicted_vels = vels[:, :, 1:, :].detach().clone()
        # predicted_rot_vels = rot_vels[:, :, 1:, :].detach().clone()

        batch_size, window_size, num_bones, _ = positions.shape
        predicted_pos = torch.empty(batch_size, window_size, NUM_T_BONES, 3) # shape [batch_size, window_size, num_bones, 3]
        predicted_rots = torch.empty(batch_size, window_size, NUM_T_BONES, 4) 
        predicted_vels = torch.empty(batch_size, window_size, NUM_T_BONES, 3) 
        predicted_rot_vels = torch.empty(batch_size, window_size, NUM_T_BONES, 3) 

        predicted_pos[:, 0, :, :] = positions[:, 0, 1:, :] # shape [batch_size, window_size, num_bones, 3]
        predicted_rots[:, 0, :, :]  = rotations[:, 0, 1:, :] 
        predicted_vels[:, 0, :, :]  = vels[:, 0, 1:, :]
        predicted_rot_vels[:, 0, :, :]  = rot_vels[:, 0, 1:, :]

        world_model_tensors = [predicted_pos, predicted_rots, predicted_vels, predicted_rot_vels, kin_rot_t, kin_rvel_t]

        for i in range(raw_window_size):
            # Take one step through world
            next_predicted_values = SupertrackUtils.integrate_through_world_model(self._world_model, self.dtime, *[t[:, i, ...] for t in world_model_tensors], update_normalizer=i==0) # Only update normalizer if using ground truth values
            predicted_pos[:, i+1, ...], predicted_rots[:, i+1, ...], predicted_vels[:, i+1, ...], predicted_rot_vels[:, i+1, ...] = next_predicted_values


        # We slice using [:, 1:, 1:, :] because we want to compute losses over the entire batch, skip the first window step (since that was not predicted by
        # the world model), and skip the root bone 
        raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l = self.char_state_loss(positions[:, 1:, 1:, :],
                                                        predicted_pos[:, 1:, :, :],
                                                        rotations[:, 1:, 1:, :],
                                                        predicted_rots[:, 1:, :, :], 
                                                        vels[:, 1:, 1:, :], 
                                                        predicted_vels[:, 1:, :, :], 
                                                        rot_vels[:, 1:, 1:, :],
                                                        predicted_rot_vels[:, 1:, :, :])
        
        pos_loss, rot_loss, vel_loss, rvel_loss = self.wm_loss_weights.get_reweighted_losses(raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l)
        loss = pos_loss + rot_loss + vel_loss + rvel_loss
        update_stats = {'World Model/wpos_loss': pos_loss.item(),
                         'World Model/wrot_loss': rot_loss.item(),
                         'World Model/wvel_loss': vel_loss.item(),
                         'World Model/wrvel_loss': rvel_loss.item(),
                         'World Model/total loss': loss.item(),
                         'World Model/learning_rate': self.wm_lr}

        self.world_model_optimzer.zero_grad(set_to_none=True)
        loss.backward()
        if self.hyperparameters.gradient_clipping > 0: 
            torch.nn.utils.clip_grad_norm_(self._world_model.parameters(), self.hyperparameters.gradient_clipping)
        self.world_model_optimzer.step()
        self.first_wm_update = False
        return update_stats
    
    def char_state_loss(self, pos1, pos2, rot1, rot2, vel1, vel2, rvel1, rvel2):
        # Input shapes: [batch_size, window_size, NUM_T_BONES, 3 or 4]

        def l1_norm(a, b, c = None):
            diff = c if c is not None else a - b # shape: [batch_size, window_size, num_bones, 3]
            return diff.abs().sum(dim=(1,2,3)).mean()
            # return torch.mean(torch.sum(diff, dim=(1,2,3)))
        
        raw_pos_l = l1_norm(pos1, pos2) #torch.mean(torch.sum(torch.abs(pos1-pos2), dim =(1,2,3)))
        raw_vel_l = l1_norm(vel1, vel2) 
        raw_rvel_l = l1_norm(rvel1, rvel2)
        if rot1.shape[-1] != 4: # rots are in quat form
            raise Exception(f"Rots in unexpected shape: {rot1.shape}")

        # From Stack Overflow:
        # If you want to find a quaternion diff such that diff * q1 == q2, then you need to use the multiplicative inverse:
        # diff * q1 = q2  --->  diff = q2 * inverse(q1)
        # https://stackoverflow.com/questions/21513637/dot-product-of-two-quaternion-rotations
        quat_diffs = pyt.quaternion_multiply(SupertrackUtils.normalize_quat(rot2) , pyt.quaternion_invert(SupertrackUtils.normalize_quat(rot1) ))
        # vec_part = quat_diffs[..., 1:] # The magnitude of the vec part of a quaternion equals sin(angle/2) where angle is the angle of the quat
        norms = torch.norm(quat_diffs[..., 1:], p=2, dim=-1, keepdim=True) # The magnitude of the vec part of a quaternion equals sin(angle/2) where angle is the angle of the quat
        # scalar_part = quat_diffs[...,:1] # The real/scalar part of a quaternion equals cos(angle/2) where angle is the angle of the quat
        # We're basically breaking the quaternion down into the sin and cos values of its angle, and 
        # atan2() is a function that, given a cos and sin value for an angle, returns the angle between it and the unit vector (1, 0)
        halfangles = torch.atan2(norms, quat_diffs[..., :1])
        quat_logs = halfangles * (quat_diffs[..., 1:] / norms)
        raw_rot_l = l1_norm(None, None, c=quat_logs) #quat_logs.abs().sum(dim=(1,2,3)).mean()

        # batch_size, window_size, num_bones, num_entries = rot1.shape
        # quat_diffs = pyt.quaternion_multiply(rot2, pyt.quaternion_invert(rot1))
        # quat_logs = pyt.so3_log_map(pyt.quaternion_to_matrix(quat_diffs).reshape(-1, 3, 3)).reshape(batch_size, window_size, num_bones, 3)
        # raw_rot_l = l1_norm(None, None, c=quat_logs.abs())

        # d = torch.abs(torch.sum(SupertrackUtils.normalize_quat(rot1) * SupertrackUtils.normalize_quat(rot2), dim=-1))
        # d = torch.clamp(d, min=-1.0, max=1.0)
        # theta = 2 * torch.acos(d)
        # raw_rot_l = theta.sum(dim=(1,2)).mean()

        return raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l

    def update_policy(self, batch: STBuffer, batch_size: int, raw_window_size: int, nsys_profiler_running: bool = False) -> Dict[str, float]:
        cur_actor = self.policy.actor
        if self.split_actor_devices:
            cur_actor = self.actor_gpu
        cur_actor.train()
        window_size = raw_window_size + 1
        suffixes = [CharTypeSuffix.POSITION, CharTypeSuffix.ROTATION, CharTypeSuffix.VEL, CharTypeSuffix.RVEL]
        # suffixes = [CharTypeSuffix.POSITION, CharTypeSuffix.ROTATION, CharTypeSuffix.VEL, CharTypeSuffix.RVEL, CharTypeSuffix.HEIGHT, CharTypeSuffix.UP_DIR]
        s_pos, s_rots, s_vels, s_rvels = [batch[(CharTypePrefix.SIM, suffix)][:, :, 1:, :]  for suffix in suffixes]
        ground_truth_sim_data = [s_pos, s_rots, s_vels, s_rvels] # shape [batch_size, window_size, num_t_bones, 3]

        k_pos, k_rots, k_vels, k_rvels = [batch[(CharTypePrefix.KIN, suffix)][:, :, 1:, :] for suffix in suffixes]
        pre_target_rots, pre_target_vels =  batch[(PDTargetPrefix.PRE, PDTargetSuffix.ROT)],  batch[(PDTargetPrefix.PRE, PDTargetSuffix.RVEL)] 
        # Remove root bone from PDTargets 
        pre_target_rots, pre_target_vels = pre_target_rots[:, :, 1:, :], pre_target_vels[:, :, 1:, :]

        predicted_spos, predicted_srots, predicted_svels, predicted_srvels = [s.detach().clone() for s in ground_truth_sim_data]
        predicted_global_sim_state =  [predicted_spos, predicted_srots, predicted_svels, predicted_srvels]

        predicted_local_spos, predicted_local_srots, predicted_local_svels, predicted_local_srvels = [torch.empty(batch_size, raw_window_size, NUM_T_BONES, s.shape[-1]) for s in ground_truth_sim_data]

        local_kin_with_quat = SupertrackUtils.local(k_pos, k_rots, k_vels, k_rvels, include_quat_rots=True)
        local_kin = local_kin_with_quat[:-1]

        def get_tensor_at_window_step_i(t, i):
            return t[:, i, ...]
        
        all_means = torch.empty(batch_size, raw_window_size, POLICY_OUTPUT_LEN)

        sim_state_window_step_i =  [get_tensor_at_window_step_i(t, 0) for t in predicted_global_sim_state]
        local_sim_window_step_i = SupertrackUtils.local(*sim_state_window_step_i, include_quat_rots=False)

        for window_step_i in range(raw_window_size):
            # Predict PD offsets
            local_kin_at_kin_idx = [get_tensor_at_window_step_i(k, window_step_i) for k in local_kin]
            input = [*local_kin_at_kin_idx , *local_sim_window_step_i]
            global_drift = None
            if self.policy_includes_global_data:
                global_drift = torch.empty(batch_size, 3)
                sim_hip_world_pos = sim_state_window_step_i[0][:, 0, :]
                kin_hip_world_pos = k_pos[:, window_step_i, 0, :]
                global_drift = kin_hip_world_pos - sim_hip_world_pos
            action, determinstic_action = cur_actor.get_action_during_training(input, global_drift=global_drift)
            all_means[:, window_step_i, :] = determinstic_action
            output = action.reshape(batch_size, NUM_T_BONES, 3)
            output =  pyt.axis_angle_to_quaternion(output)
            # Compute PD targets
            cur_kin_targets = pyt.quaternion_multiply(output, pre_target_rots[:, window_step_i, ...])
            # Pass through world model
            next_sim_state = SupertrackUtils.integrate_through_world_model(self._world_model, self.dtime, *sim_state_window_step_i,
                                                                pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(cur_kin_targets)),
                                                                pre_target_vels[:, window_step_i, ...],
                                                                local_tensors = local_sim_window_step_i,
                                                                update_normalizer=False)
            sim_state_window_step_i =  next_sim_state # Set for next iteration of loop

            if self.policy_includes_global_data:
                predicted_spos[:, window_step_i+1, ...], predicted_srots[:, window_step_i+1, ...], predicted_svels[:, window_step_i+1, ...], predicted_srvels[:, window_step_i+1, ...] = next_sim_state
                local_sim_window_step_i = SupertrackUtils.local(*sim_state_window_step_i, include_quat_rots=False)
            else:
                local_sim_window_step_i_w_quats = SupertrackUtils.local(*sim_state_window_step_i, include_quat_rots=True)
                local_sim_window_step_i = local_sim_window_step_i_w_quats[:-1]
                predicted_local_spos[:, window_step_i, ...]  = local_sim_window_step_i[0].reshape(batch_size, NUM_T_BONES, -1)
                predicted_local_srots[:, window_step_i, ...] =  local_sim_window_step_i_w_quats[-1].reshape(batch_size, NUM_T_BONES, -1)
                predicted_local_svels[:, window_step_i, ...] = local_sim_window_step_i[2].reshape(batch_size, NUM_T_BONES, -1)
                predicted_local_srvels[:, window_step_i, ...] = local_sim_window_step_i[3].reshape(batch_size, NUM_T_BONES, -1)

        # Compute losses:
        # "The difference between this prediction [of simulated state] and the target
        # kinematic states K is then computed in the local space, and the
        # losses used to update the weights of the policy"

        # We don't want to use the last window step bc of the way we read info in from unity - a sim state
        # for time t is actually paired with kin state at t + 1, because for a given frame we update kin state,
        # take in kin state and sim state into policy, apply offsets, then step physics sim
        # So for window step 0, we want to compare how loss/dist between sim state for window step 1 and kin state for window step 0 
        reshape_kin_data = lambda x : x.reshape(batch_size, window_size, NUM_T_BONES, -1)[:, :-1, ...] 
        reshape_sim_data = lambda x : x

        if self.policy_includes_global_data:
            reshape_sim_data = lambda x : x[:, 1:, ...] 
            kin_tensors_for_loss = k_pos, k_rots, k_vels, k_rvels
            sim_tensors_for_loss = predicted_spos, predicted_srots, predicted_svels, predicted_srvels
        else:
            # We don't need to filter out the root bone because SuperTrackUtils.local already does that
            kin_tensors_for_loss = local_kin[0], local_kin_with_quat[-1], local_kin[2], local_kin[3]
            sim_tensors_for_loss = predicted_local_spos, predicted_local_srots, predicted_local_svels, predicted_local_srvels

        loss_kpos, loss_krots, loss_kvels, loss_krvels = [reshape_kin_data(t) for t in kin_tensors_for_loss]
        loss_spos, loss_srots, loss_svels, loss_srvels = [reshape_sim_data(t) for t in sim_tensors_for_loss]
        
        raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l = self.char_state_loss(loss_kpos,
                                                                        loss_spos,
                                                                        loss_krots,
                                                                        loss_srots,
                                                                        loss_kvels,
                                                                        loss_svels,
                                                                        loss_krvels,
                                                                        loss_srvels)
        pos_loss, rot_loss, vel_loss, rvel_loss = self.policy_loss_weights.get_reweighted_losses(raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l)
        # Compute regularization losses
        # Take the norm of the last dimensions, sum across windows, and take mean over batch 
        lreg = torch.norm(all_means, p=2 ,dim=-1).sum(dim=-1).mean()
        lsreg = torch.norm(all_means, p=1 ,dim=-1).sum(dim=-1).mean()
        # Weigh regularization losses to contribute 1/100th of the other losses
        lreg /= 10
        lsreg /= 10
        loss = pos_loss + rot_loss + vel_loss + rvel_loss + lreg + lsreg

        update_stats = {"Policy/Loss": loss.item(),
                        "Policy/pos_loss": pos_loss.item(),
                        "Policy/rot_loss": rot_loss.item(),
                        "Policy/vel_loss": vel_loss.item(),
                        "Policy/rvel_loss": rvel_loss.item(),
                        "Policy/reg_loss": lreg.item(),
                        "Policy/sreg_loss": lsreg.item(),
                        "Policy/learning_rate": self.policy_lr}
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.hyperparameters.gradient_clipping > 0: 
            torch.nn.utils.clip_grad_norm_(cur_actor.parameters(), self.hyperparameters.gradient_clipping)
        self.policy_optimizer.step()
        self.first_policy_update = False
        return update_stats
    
    def get_modules(self):
        modules = {
            "Optimizer:WorldModel": self._world_model,
            "Optimizer:world_model_optimzer": self.world_model_optimzer,
            "Optimizer:policy_optimizer": self.policy_optimizer,
            "Optimizer:world_model_loss_weights": self.wm_loss_weights,
            "Optimizer:policy_loss_weights": self.policy_loss_weights,
         }
        return modules
    

# POLICY_NORMALIZATION_SIZE = what do we want to normalize? local_pos, local_vels for sim and kin
# = (NUM_T_BONES * (3+3))*2 ; for NUM_T_BONES = 16 => (16*6)*2 = 96 * 2 = 192
POLICY_NORMALIZATION_SIZE = 192
class PolicyNetworkBody(nn.Module):
    def __init__(
            self,
            network_settings: NetworkSettings,
            policy_includes_global_data: bool = False,
    ):
        super().__init__()
        self.network_settings = network_settings
        self.input_size = self.network_settings.input_size
        if (self.input_size == -1):
            raise Exception("SuperTrack Policy Network created without input_size designated in yaml file")
        
        _layers = []
        # Used to normalize inputs
        if self.network_settings.normalize:
            self.normalizer = Normalizer(POLICY_NORMALIZATION_SIZE)

        input_size = self.network_settings.input_size
        input_size += 3 if policy_includes_global_data else 0

        _layers += [LinearEncoder(
            input_size,
            self.network_settings.num_layers,
            self.network_settings.hidden_units,
            Initialization.KaimingHeNormal,
            1,
            network_settings.activation_function)]
        self.layers = nn.Sequential(*_layers)

    @property
    def memory_size(self) -> int:
        return 0

    def forward(self, k_local_pos : torch.Tensor, # [batch_size, NUM_T_BONES * 3]
            k_local_rots_6d: torch.Tensor,    # [batch_size, NUM_T_BONES * 6] 
            k_local_vels: torch.Tensor,       # [batch_size, NUM_T_BONES * 3]
            k_local_rot_vels: torch.Tensor,   # [batch_size, NUM_T_BONES * 3]
            k_local_up_dir: torch.Tensor,     # [batch_size, 3]
            s_local_pos : torch.Tensor,       # [batch_size, NUM_T_BONES * 3]
            s_local_rots_6d: torch.Tensor,    # [batch_size, NUM_T_BONES * 6] 
            s_local_vels: torch.Tensor,       # [batch_size, NUM_T_BONES * 3]
            s_local_rot_vels: torch.Tensor,   # [batch_size, NUM_T_BONES * 3]
            s_local_up_dir: torch.Tensor,     # [batch_size, 3]
            update_normalizer: bool = False,
            global_drift: torch.Tensor = None, # [batch_size, 3]
    ) -> torch.Tensor:
        normalizable_inputs = torch.cat((k_local_pos, k_local_vels, s_local_pos, s_local_vels), dim=-1)
        if self.network_settings.normalize:
            if update_normalizer:
                self.normalizer.update(normalizable_inputs)
            normalizable_inputs = self.normalizer(normalizable_inputs)
        input_tensors = [normalizable_inputs,
            k_local_rots_6d,
            k_local_rot_vels, 
            k_local_up_dir,   
            s_local_rots_6d,
            s_local_rot_vels, 
            s_local_up_dir]
        if global_drift is not None:
            input_tensors += [global_drift]
        inputs = torch.cat(input_tensors, dim=-1)
        return self.layers(inputs)

class SuperTrackPolicyNetwork(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 3
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
        device: str = None,
        clip_action: bool = True,
        policy_includes_global_data: bool = False,
    ):
        super().__init__()
        self.network_body = PolicyNetworkBody(network_settings, policy_includes_global_data=policy_includes_global_data)
        self.action_spec = action_spec
        self.continuous_act_size_vector = torch.nn.Parameter(
            torch.Tensor([network_settings.output_size]), requires_grad=False
        )
        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )
        self.memory_size_vector = torch.nn.Parameter(
            torch.Tensor([0]), requires_grad=False
        )
        self.encoding_size = network_settings.hidden_units
        # Could convert action_spec to class instead of tuple, but having the dependency that Unity action size == Python action size
        # is not a huge constraint
        # action_spec.continuous_size = network_settings.output_size
        self.output_scale = network_settings.output_scale
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
            init_near_zero=network_settings.init_near_zero,
            noise_scale=.3,
            clip_action=clip_action,
            output_scale=self.output_scale,
        )
        self.policy_includes_global_data = policy_includes_global_data
        # self.highest_kin_vel = torch.tensor([0, 0, 0], dtype=torch.float32, device= torch.device("cpu"))
        # self.highest_sim_vel = torch.tensor([0, 0, 0], dtype=torch.float32, device= torch.device("cpu"))

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size
    

    def forward(
        self,
        inputs: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.

        At this moment, torch.onnx.export() doesn't accept None as tensor to be exported,
        so the size of return tuple varies with action spec.
        """

        # For SuperTrack, we have a special encoding. We never call forward directly, only get_action_and_stats 
        # for actions during gym step generation and get_action_during_training for the action during train time
        # So we just assume this forward is only being called to export the policy and it's easier to modify this
        # then setup a new model serializer

        dummy_obs = inputs

        SIZE_POS = NUM_T_BONES * 3
        SIZE_ROTS_6D = NUM_T_BONES * 6
        SIZE_VELS = NUM_T_BONES * 3
        SIZE_ROT_VELS = NUM_T_BONES * 3
        SIZE_UP_DIR = 3
        idx = 0
        k_local_pos = dummy_obs[:, idx:idx + SIZE_POS]
        idx += SIZE_POS
        k_local_rots_6d = dummy_obs[:, idx:idx + SIZE_ROTS_6D]
        idx += SIZE_ROTS_6D
        k_local_vels = dummy_obs[:, idx:idx + SIZE_VELS]
        idx += SIZE_VELS
        k_local_rot_vels = dummy_obs[:, idx:idx + SIZE_ROT_VELS]
        idx += SIZE_ROT_VELS
        k_local_up_dir = dummy_obs[:, idx:idx + SIZE_UP_DIR]
        idx += SIZE_UP_DIR
        s_local_pos = dummy_obs[:, idx:idx + SIZE_POS]
        idx += SIZE_POS
        s_local_rots_6d = dummy_obs[:, idx:idx + SIZE_ROTS_6D]
        idx += SIZE_ROTS_6D
        s_local_vels = dummy_obs[:, idx:idx + SIZE_VELS]
        idx += SIZE_VELS
        s_local_rot_vels = dummy_obs[:, idx:idx + SIZE_ROT_VELS]
        idx += SIZE_ROT_VELS
        s_local_up_dir = dummy_obs[:, idx:idx + SIZE_UP_DIR]
        idx += SIZE_UP_DIR
        global_drift = None
        if self.policy_includes_global_data:
            global_drift = dummy_obs[:, idx: idx + SIZE_UP_DIR] # it's also 3
            idx += SIZE_UP_DIR

        encoding = self.network_body(
            k_local_pos,
            k_local_rots_6d,
            k_local_vels,
            k_local_rot_vels,
            k_local_up_dir,
            s_local_pos,
            s_local_rots_6d,
            s_local_vels,
            s_local_rot_vels,
            s_local_up_dir,
            global_drift=global_drift,
        )

        (
            cont_action_out,
            _disc_action_out,
            _action_out_deprecated,
            deterministic_cont_action_out,
            _deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, masks)
        export_out = [ 
            self.version_number,
            self.memory_size_vector,
            cont_action_out,
            self.continuous_act_size_vector,
            deterministic_cont_action_out,
        ]
        return tuple(export_out)

    @timed
    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """
        if (len(inputs) != 1):
            raise Exception(f"SuperTrack policy network body initialized with multiple observations: {len(inputs)} ")

        # should be shape [num_obs_types (1), num_agents, POLICY_INPUT_LEN or NUM_OBS]
        policy_input = inputs[0]
        supertrack_data = SupertrackUtils.parse_supertrack_data_field(policy_input)
        policy_input, global_drift = SupertrackUtils.process_raw_observations_to_policy_input(supertrack_data, self.policy_includes_global_data)

        encoding = self.network_body(*policy_input, global_drift=global_drift)

        action, log_probs, entropies, _means = self.action_model(encoding, None, include_log_probs_entropies=True) 
        run_out = {}
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.
        run_out["env_action"] = action.to_action_tuple(clip=self.action_model.clip_action,  output_scale=self.output_scale)
        # For some reason, sending CPU tensors causes the training to hang
        # This does not occur with CUDA tensors of numpy ndarrays
        # if supertrack_data is not None:
        #     run_out["supertrack_data"] = supertrack_data
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return action, run_out, None
    
    def get_action_during_training(
        self,
        inputs: List[torch.Tensor],
        global_drift: torch.Tensor = None,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param inputs: inputs of size [batch_size, POLICY_INPUT_LEN]
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """
        encoding = self.network_body(*inputs, update_normalizer=True, global_drift=global_drift)
        (
            cont_action_out,
            _disc_action_out,
            _action_out_deprecated,
            deterministic_cont_action_out,
            _deterministic_disc_action_out,
        ) = self.action_model.get_action_out(encoding, None)
        return cont_action_out, deterministic_cont_action_out
    

    def update_normalization(self, buffer) -> None:
        pass # Not needed because we use call our own updates through a kwarg during training

