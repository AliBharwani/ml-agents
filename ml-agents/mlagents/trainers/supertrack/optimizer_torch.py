import json
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast
from mlagents.st_buffer import CharTypePrefix, CharTypeSuffix, PDTargetPrefix, PDTargetSuffix, STBuffer

from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr
import pdb
from mlagents.torch_utils import torch, nn, default_device
from mlagents.trainers.supertrack import world_model
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
    offset_scale: float = 120.0


def hn(x):
    if isinstance(x, list):
        return any([y.isnan().any() for y in x])
    return torch.isnan(x).any()

class DynamicLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.wpos_loss = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.wrot_loss = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.wvel_loss = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.wrvel_loss = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.initialized = nn.Parameter(torch.tensor(False), requires_grad=False)

    def to_str(self):
        return f"""
                Pos: {self.wpos_loss.data}
                Rot: {self.wrot_loss.data}
                Vel: {self.wvel_loss.data}
                Rvel: {self.wrvel_loss.data}"""

    def get_reweighted_losses(self, pos_loss, rot_loss, vel_loss, rvel_loss):
        if not self.initialized.item():
            total_loss = pos_loss + rot_loss + vel_loss + rvel_loss
            losses = [pos_loss, rot_loss, vel_loss, rvel_loss]
            for i, loss in enumerate(losses):
                if loss.item() == 0:
                    losses[i] = torch.tensor(1.0)  # Avoid division by zero
            avg_loss = total_loss / 4
            self.wpos_loss.data = avg_loss / losses[0]
            self.wrot_loss.data = avg_loss / losses[1]
            self.wvel_loss.data = avg_loss / losses[2]
            self.wrvel_loss.data = avg_loss / losses[3]
            self.initialized.data = torch.tensor(True)
        return self.wpos_loss * pos_loss , self.wrot_loss * rot_loss , self.wvel_loss * vel_loss , self.wrvel_loss * rvel_loss

class TorchSuperTrackOptimizer(TorchOptimizer):
    dtime = 1 / 60

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.trainer_settings = trainer_settings
        self.hyperparameters: SuperTrackSettings = cast(
            SuperTrackSettings, trainer_settings.hyperparameters
        )
        self.offset_scale = self.hyperparameters.offset_scale
        self.wm_lr = trainer_settings.world_model_network_settings.learning_rate
        self.policy_lr = trainer_settings.policy_network_settings.learning_rate
        self.first_wm_update = True
        self.first_policy_update = True
        self.split_actor_devices = self.trainer_settings.use_pytorch_mp and default_device() == torch.device('cuda')
        self.actor_gpu = None
        self.policy_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.policy_lr)
        self.logger = get_logger(__name__)
        self.wm_loss_weights = DynamicLoss()
        self.policy_loss_weights = DynamicLoss()
                
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
        if self.first_wm_update:
            self.logger.debug(f"WORLD MODEL DEVICE:  {next(self._world_model.parameters()).device}")
            cur_actor = self.policy.actor
            if self.split_actor_devices:
                cur_actor = self.actor_gpu
            self.logger.debug(f"POLICY DEVICE: {next(cur_actor.parameters()).device}")
        
        self._world_model.train()
        suffixes = [CharTypeSuffix.POSITION, CharTypeSuffix.ROTATION, CharTypeSuffix.VEL, CharTypeSuffix.RVEL, CharTypeSuffix.HEIGHT, CharTypeSuffix.UP_DIR]
        positions, rotations, vels, rot_vels, heights, up_dir = [batch[(CharTypePrefix.SIM, suffix)] for suffix in suffixes]
        kin_rot_t, kin_rvel_t = batch[(PDTargetPrefix.POST, PDTargetSuffix.ROT)], batch[(PDTargetPrefix.POST, PDTargetSuffix.RVEL)]
        # remove root bone from PDTargets 
        kin_rot_t, kin_rvel_t = kin_rot_t[:, :, 1:, :], kin_rvel_t[:, :, 1:, :] 
        kin_rot_t =  pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(kin_rot_t))

        predicted_pos = positions.detach().clone() # shape [batch_size, window_size, num_bones, 3]
        predicted_rots = rotations.detach().clone()
        predicted_vels = vels.detach().clone()
        predicted_rot_vels = rot_vels.detach().clone()
        # world_model_tensors = [predicted_pos, predicted_rots, predicted_vels, predicted_rot_vels, heights, up_dir, kin_rot_t, kin_rvel_t]
        world_model_tensors = [predicted_pos, predicted_rots, predicted_vels, predicted_rot_vels, kin_rot_t, kin_rvel_t]

        for i in range(raw_window_size):
            # Take one step through world
            next_predicted_values = SupertrackUtils.integrate_through_world_model(self._world_model, self.dtime, *[t[:, i, ...] for t in world_model_tensors])
            # This overwrites the next window step with the root data of the current pos / rot, since we are updating everything 
            predicted_pos[:, i+1, ...], predicted_rots[:, i+1, ...], predicted_vels[:, i+1, ...], predicted_rot_vels[:, i+1, ...] = next_predicted_values
            # Copy over root pos and root rot, because world model does not update them
            predicted_pos[:, i+1, 0, :] = positions[:, i+1, 0, :]
            predicted_rots[:, i+1, 0, :] = rotations[:, i+1, 0, :]

        # We slice using [:, 1:, 1:, :] because we want to compute losses over the entire batch, skip the first window step (since that was not predicted by
        # the world model), and skip the root bone 
        raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l = self.char_state_loss(positions[:, 1:, 1:, :],
                                                        predicted_pos[:, 1:, 1:, :],
                                                        rotations[:, 1:, 1:, :],
                                                        predicted_rots[:, 1:, 1:, :], 
                                                        vels[:, 1:, 1:, :], 
                                                        predicted_vels[:, 1:, 1:, :], 
                                                        rot_vels[:, 1:, 1:, :],
                                                        predicted_rot_vels[:, 1:, 1:, :])
        
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
        self.world_model_optimzer.step()
        self.first_wm_update = False
        return update_stats
    
    def char_state_loss(self, pos1, pos2, rot1, rot2, vel1, vel2, rvel1, rvel2):
        # Input shapes: [batch_size, window_size, NUM_T_BONES, 3 or 4]

        def l1_norm(a, b, c = None):
            diff = torch.abs(c if c is not None else a - b) # shape: [batch_size, window_size, num_bones, 3]
            return torch.mean(torch.sum(diff, dim=(1,2,3)))
        
        raw_pos_l = l1_norm(pos1, pos2) #torch.mean(torch.sum(torch.abs(pos1-pos2), dim =(1,2,3)))
        raw_vel_l = l1_norm(vel1, vel2) 
        raw_rvel_l = l1_norm(rvel1, rvel2)
        if rot1.shape[-1] != 4: # rots are in quat form
            raise Exception(f"Rots in unexpected shape: {rot1.shape}")

        # From Stack Overflow:
        # If you want to find a quaternion diff such that diff * q1 == q2, then you need to use the multiplicative inverse:
        # diff * q1 = q2  --->  diff = q2 * inverse(q1)
        # https://stackoverflow.com/questions/21513637/dot-product-of-two-quaternion-rotations
        quat_diffs = pyt.quaternion_multiply(rot2, pyt.quaternion_invert(rot1))
        vec_part = quat_diffs[..., 1:] # The magnitude of the vec part of a quaternion equals sin(angle/2) where angle is the angle of the quat
        scalar_part = quat_diffs[..., 0:1] # The real/scalar part of a quaternion equals cos(angle/2) where angle is the angle of the quat
        # We're basically breaking the quaternion down into the sin and cos values of its angle, and 
        # atan2() is a function that, given a cos and sin value for an angle, returns the angle between it and the unit vector (1, 0)
        angles = 2 * torch.atan2(vec_part.norm(p=2, dim=-1), scalar_part.squeeze(-1))
        raw_rot_l = angles.abs().sum(dim=(1,2)).mean()

        return raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l

    def update_policy(self, batch: STBuffer, batch_size: int, raw_window_size: int, nsys_profiler_running: bool = False) -> Dict[str, float]:
        cur_actor = self.policy.actor
        if self.split_actor_devices:
            cur_actor = self.actor_gpu
        cur_actor.train()
        window_size = raw_window_size + 1

        suffixes = [CharTypeSuffix.POSITION, CharTypeSuffix.ROTATION, CharTypeSuffix.VEL, CharTypeSuffix.RVEL, CharTypeSuffix.HEIGHT, CharTypeSuffix.UP_DIR]
        s_pos, s_rots, s_vels, s_rvels, s_h, s_up = [batch[(CharTypePrefix.SIM, suffix)]  for suffix in suffixes]
        ground_truth_sim_data = [s_pos, s_rots, s_vels, s_rvels] # shape [batch_size, window_size, num_bones, 3]

        k_pos, k_rots, k_vels, k_rvels, k_h, k_up = [batch[(CharTypePrefix.KIN, suffix)] for suffix in suffixes]
        pre_target_rots, pre_target_vels =  batch[(PDTargetPrefix.PRE, PDTargetSuffix.ROT)],  batch[(PDTargetPrefix.PRE, PDTargetSuffix.RVEL)] 
        # Remove root bone from PDTargets 
        pre_target_rots, pre_target_vels = pre_target_rots[:, :, 1:, :], pre_target_vels[:, :, 1:, :]

        predicted_spos, predicted_srots, predicted_svels, predicted_srvels = [s.detach().clone() for s in ground_truth_sim_data]
        sim_state =  [predicted_spos, predicted_srots, predicted_svels, predicted_srvels]

        predicted_local_spos, predicted_local_srots, predicted_local_svels, predicted_local_srvels = [torch.empty(batch_size, raw_window_size, NUM_T_BONES, s.shape[-1]) for s in ground_truth_sim_data]
        # local_srots = torch.empty(batch_size, raw_window_size, NUM_T_BONES, 6) # We will put two-axis rotations into this tensor
        predicted_sim_state = [predicted_local_spos, predicted_local_srots, predicted_local_svels, predicted_local_srvels]

        local_kin_with_quat = SupertrackUtils.local(k_pos, k_rots, k_vels, k_rvels, include_quat_rots=True)
        local_kin = local_kin_with_quat[:-1]

        def get_tensor_at_window_step_i(t, i):
            return t[:, i, ...]
        
        all_means = torch.empty(batch_size, raw_window_size, POLICY_OUTPUT_LEN)

        sim_state_window_step_i =  [get_tensor_at_window_step_i(t, 0) for t in sim_state]
        local_sim_window_step_i_w_quats = SupertrackUtils.local(*sim_state_window_step_i, include_quat_rots=True)
        local_sim_window_step_i = local_sim_window_step_i_w_quats[:-1]

        for window_step_i in range(raw_window_size):
            # Predict PD offsets
            # local_kin_at_window_step_i_plus_1 = [get_tensor_at_window_step_i(k, i + 1) for k in local_kin]
            # input = torch.cat((*local_kin_at_window_step_i_plus_1, *local_sim_window_step_i), dim=-1)
            local_kin_at_window_step_i = [get_tensor_at_window_step_i(k, window_step_i) for k in local_kin]
            input = torch.cat((*local_kin_at_window_step_i, *local_sim_window_step_i), dim=-1)

            action, runout, _ = cur_actor.get_action_and_stats([input], inputs_already_formatted=True, return_means=True)
            all_means[:, window_step_i, :] = runout['means']
            output = action.continuous_tensor.reshape(batch_size, NUM_T_BONES, 3)
            output =  pyt.axis_angle_to_quaternion(output * self.offset_scale)
            # Compute PD targets
            # cur_kin_targets = pyt.quaternion_multiply(output, pre_target_rots[:, i + 1, ...])
            cur_kin_targets = pyt.quaternion_multiply(output, pre_target_rots[:, window_step_i , ...])
            # Pass through world model
            next_sim_state = SupertrackUtils.integrate_through_world_model(self._world_model, self.dtime, *sim_state_window_step_i,
                                                                pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(cur_kin_targets)),
                                                                # pre_target_vels[:, i + 1, ...],
                                                                pre_target_vels[:, window_step_i, ...],
                                                                local_tensors = local_sim_window_step_i)
            predicted_spos[:, window_step_i+1, ...], predicted_srots[:, window_step_i+1, ...], predicted_svels[:, window_step_i+1, ...], predicted_srvels[:, window_step_i+1, ...] = next_sim_state
            # Copy over root pos and root rot, because world model does not update them
            predicted_spos[:, window_step_i+1, 0, :] = s_pos[:, window_step_i+1, 0, :]
            predicted_srots[:, window_step_i+1, 0, :] = s_rots[:, window_step_i+1, 0, :]

            sim_state_window_step_i =  [get_tensor_at_window_step_i(t, window_step_i + 1) for t in sim_state]
            # local_sim_window_step_i = SupertrackUtils.local(*sim_state_window_step_i)
            local_sim_window_step_i_w_quats = SupertrackUtils.local(*sim_state_window_step_i, include_quat_rots=True)
            local_sim_window_step_i = local_sim_window_step_i_w_quats[:-1]
            for idx_into_list in range(4):
                predicted_local_sim_tensor_to_update = predicted_sim_state[idx_into_list]
                if predicted_local_sim_tensor_to_update.shape[-1] == 4: # Handle rotations separately
                    tensor_to_copy = local_sim_window_step_i_w_quats[-1] # rot
                else:
                    tensor_to_copy = local_sim_window_step_i[idx_into_list] # pos, vel, rvel 
                predicted_local_sim_tensor_to_update[:, window_step_i, ...] = tensor_to_copy.reshape(batch_size, NUM_T_BONES, -1)
                
            # We've converted this steps output into local space, copy that over for loss computation
            # for local_sim_for_loss, local_sim_calculated in zip(local_sim, local_sim_window_step_i):
            #     local_sim_for_loss[:, i, ...] = local_sim_calculated.reshape(batch_size, NUM_T_BONES, -1)

        # Compute losses:
        # "The difference between this prediction [of simulated state] and the target
        # kinematic states K is then computed in the local space, and the
        # losses used to update the weights of the policy"

        # We don't want to use the first window step because those were ground truth values (for local_kin data)
        # We don't need to filter out the root bone because SuperTrackUtils.local already does that
        # local_kpos, local_krots, local_kvels, local_krvels = [k.reshape(batch_size, window_size, NUM_T_BONES, -1)[:, 1:, ...] for k in local_kin]
        reshape_local_kin_data = lambda x : x.reshape(batch_size, window_size, NUM_T_BONES, -1)[:, 1:, ...] 
        local_kpos = reshape_local_kin_data(local_kin[0])
        local_krots = reshape_local_kin_data(local_kin_with_quat[-1])
        local_kvels = reshape_local_kin_data(local_kin[2])
        local_krvels = reshape_local_kin_data(local_kin[3])
        
        raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l = self.char_state_loss(local_kpos,
                                                                        predicted_local_spos,
                                                                        local_krots,
                                                                        predicted_local_srots,
                                                                        local_kvels, 
                                                                        predicted_local_svels,
                                                                        local_krvels,
                                                                        predicted_local_srvels)
        pos_loss, rot_loss, vel_loss, rvel_loss = self.policy_loss_weights.get_reweighted_losses(raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l)
        # Compute regularization losses
        # Take the norm of the last dimensions, sum across windows, and take mean over batch 
        lreg = torch.norm(all_means, p=2 ,dim=-1).sum(dim=-1).mean()
        lsreg = torch.norm(all_means, p=1 ,dim=-1).sum(dim=-1).mean()
        # Weigh regularization losses to contribute 1/100th of the other losses
        lreg /= 100
        lsreg /= 100
        loss = pos_loss + rot_loss + vel_loss + rvel_loss + lreg + lsreg

        update_stats = {"Policy/Loss": loss.item(),
                        "Policy/pos_loss": pos_loss.item(),
                        "Policy/rot_loss": rot_loss.item(),
                        "Policy/vel_loss": vel_loss.item(),
                        "Policy/ang_loss": rvel_loss.item(),
                        "Policy/reg_loss": lreg.item(),
                        "Policy/sreg_loss": lsreg.item(),
                        "Policy/learning_rate": self.policy_lr}
        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
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
    

class PolicyNetworkBody(nn.Module):
    def __init__(
            self,
            network_settings: NetworkSettings,
    ):
        super().__init__()
        self.network_settings = network_settings
        self.input_size = self.network_settings.input_size
        if (self.input_size == -1):
            raise Exception("SuperTrack Policy Network created without input_size designated in yaml file")
        
        _layers = []
        # Used to normalize inputs
        # if self.network_settings.normalize:
        #     _layers += [nn.LayerNorm(self.input_size)]

        _layers += [LinearEncoder(
            self.network_settings.input_size,
            self.network_settings.num_layers,
            self.network_settings.hidden_units,
            Initialization.KaimingHeNormal,
            1,
            network_settings.activation_function)]
        self.layers = nn.Sequential(*_layers)

    @property
    def memory_size(self) -> int:
        return 0

    def forward(self, inputs: torch.Tensor):
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
    ):
        super().__init__()
        # self.network_body = NetworkBody(observation_specs, network_settings)
        self.network_body = PolicyNetworkBody(network_settings)
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
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
            init_near_zero=network_settings.init_near_zero,
            noise_scale=.1,
            clip_action=clip_action,
        )

    @property
    def memory_size(self) -> int:
        return self.network_body.memory_size
    

    def forward(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        """
        Note: This forward() method is required for exporting to ONNX. Don't modify the inputs and outputs.

        At this moment, torch.onnx.export() doesn't accept None as tensor to be exported,
        so the size of return tuple varies with action spec.
        """
        encoding = self.network_body(inputs[0])

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
        sequence_length: int = 1,
        deterministic=False,
        inputs_already_formatted=False,
        return_means=False,
    ) -> Tuple[AgentAction, Dict[str, Any], torch.Tensor]:
        """
        Returns sampled actions.
        If memory is enabled, return the memories as well.
        :param inputs: A List of inputs as tensors.
        :param masks: If using discrete actions, a Tensor of action masks.
        :param memories: If using memory, a Tensor of initial memories.
        :param sequence_length: If using memory, the sequence length.
        :param deterministic: Whether to use deterministic actions.
        :param inputs_already_formatted: Whether the inputs are already formatted.
        :param return_means: Whether to return the means of the action distribution.
        :return: A Tuple of AgentAction, ActionLogProbs, entropies, and memories.
            Memories will be None if not using memory.
        """
        if (len(inputs) != 1):
            raise Exception(f"SuperTrack policy network body initialized with multiple observations: {len(inputs)} ")

        supertrack_data = None
        # should be shape [num_obs_types (1), num_agents, POLICY_INPUT_LEN or NUM_OBS]
        policy_input = inputs[0]
        if not inputs_already_formatted:
            supertrack_data = SupertrackUtils.parse_supertrack_data_field_batched(policy_input)
            policy_input = SupertrackUtils.process_raw_observations_to_policy_input(supertrack_data)
        # if policy_input.shape[-1] != POLICY_INPUT_LEN:
            # raise Exception(f"SuperTrack policy network body forward called with policy input of length {policy_input.shape[-1]}, expected {POLICY_INPUT_LEN}")
        encoding = self.network_body(policy_input)
        action, log_probs, entropies, means = self.action_model(encoding, None) 
        run_out = {}
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        # For some reason, sending CPU tensors causes the training to hang
        # This does not occur with CUDA tensors of numpy ndarrays
        # if supertrack_data is not None:
        #     run_out["supertrack_data"] = supertrack_data
        if return_means:
            run_out["means"] = means
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies

        return action, run_out, None
    

    def update_normalization(self, buffer) -> None:
        pass # Not needed because we use layernorm

