
import math
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, cast


from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr
import pdb
from mlagents.torch_utils import torch, nn, default_device
import pytorch3d.transforms as pyt
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.supertrack.supertrack_utils import  NUM_T_BONES, POLICY_INPUT_LEN, SupertrackUtils
from mlagents.trainers.supertrack.world_model import WorldModelNetwork
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.layers import LinearEncoder
from mlagents.trainers.torch_entities.networks import Actor
from mlagents_envs.base_env import ActionSpec, ObservationSpec

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
    
    
class TorchTestOptimizer(TorchOptimizer):
    dtime = 1 / 60

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.tls = threading.local()
        self.trainer_settings = trainer_settings
        self._world_model = WorldModelNetwork(
            trainer_settings.world_model_network_settings
        )
        self.init_thread_name = threading.current_thread().name
        print(f"TorchTestOptimizer is on thread: {threading.current_thread().name}")
        self._world_model.to(default_device())
        # policy.actor.to(default_device())
        self.hyperparameters: SuperTrackSettings = cast(
            SuperTrackSettings, trainer_settings.hyperparameters
        )
        self.offset_scale = self.hyperparameters.offset_scale
        self.wm_lr = trainer_settings.world_model_network_settings.learning_rate
        self.policy_lr = trainer_settings.policy_network_settings.learning_rate
        self.world_model_optimzer = torch.optim.Adam(self._world_model.parameters(), lr=self.wm_lr)
        self.policy_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=self.policy_lr)
        self._world_model.train()
        self.policy.actor.train()
        self.first_update = True


    def check_wm_layernorm(self, print_on_true : str = None):
        try: 
            for layer in self._world_model.layers:
                if isinstance(layer, nn.LayerNorm):
                    if torch.allclose(layer.weight, torch.zeros_like(layer.weight)):
                        print(f"Layer norm weight is 0! at: {print_on_true}")
                        print(f"World model layer norm data ptr: {self._world_model.layers[0].weight.data_ptr()}")
                        # pdb.set_trace()
        except Exception as e:
            pass
            # print(f"Exception in check_wm_layernorm at {print_on_true}: {e}") 

    @timed
    def update_world_model(self, batch: AgentBuffer, batch_size: int, raw_window_size: int) -> Dict[str, float]:
        if self.first_update:
            print(f"Updating on thread: {threading.current_thread().name}")
            self.check_wm_layernorm("First update call")
        return {}

        if self.first_update:
            print(f"Updating on thread: {threading.current_thread().name}")
            self._world_model = WorldModelNetwork(
                self.trainer_settings.world_model_network_settings
            )
            self._world_model.to(default_device())
            self.world_model_optimzer = torch.optim.Adam(self._world_model.parameters(), lr=self.wm_lr)
            self._world_model.train()
            self.check_wm_layernorm("Initializing in update_world_model")
            print(f"World model layer norm data ptr: {self._world_model.layers[0].weight.data_ptr()}")
            self.first_update = False
        self.check_wm_layernorm(f"At start of update_world_model")
        window_size = raw_window_size + 1
        if (batch.num_experiences // window_size != batch_size):
                raise Exception(f"Unexpected update size - expected len of batch to be {window_size} * {batch_size}, received {batch.num_experiences}")

        # sim_char_tensors = [data.as_tensors() for data in batch[BufferKey.SUPERTRACK_DATA].sim_char_state]
        st_data = [batch[BufferKey.SUPERTRACK_DATA][i] for i in range(batch.num_experiences)]
        sim_char_tensors = [st_datum.sim_char_state.as_tensors for st_datum in st_data]
        positions, rotations, vels, rot_vels, heights, up_dir = self._convert_to_usable_tensors(sim_char_tensors, batch_size, window_size)
        
        kin_targets = [st_datum.post_targets.as_tensors for st_datum in st_data]
        kin_rot_t, kin_rvel_t = self._convert_pdtargets_to_usable_tensors(kin_targets, batch_size, window_size, True)
        kin_rot_t =  pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(kin_rot_t))

        cur_pos = positions[:, 0, ...].clone().detach()
        cur_rots = rotations[:, 0, ...].clone().detach()
        cur_vels = vels[:, 0, ...].clone().detach()
        cur_rot_vels = rot_vels[:, 0, ...].clone().detach()

        for tensor, name in [(cur_pos, 'cur_pos'), (cur_rots, 'cur_rots'), (cur_vels, 'cur_vels'), (cur_rot_vels, 'cur_rot_vels')]:
            if torch.isnan(tensor).any():
                raise Exception(f"Nan in {name} at start of update_world_model")

        loss = 0
        wpos_loss = wvel_loss = wang_loss = wrot_loss = 0
        for i in range(raw_window_size):
            self.check_wm_layernorm(f"{i}")
            cur_heights = heights[:, i, ...]
            cur_up_dir = up_dir[:, i, ...]
            # Since the world model does not predict the root position, we have to copy it from the data and not use it in the loss function
            cur_pos[:, 0, :] = positions[:, i, 0, :].clone().detach()
            cur_rots[:, 0, :] = rotations[:, i , 0, :].clone().detach()
            cur_vels[:, 0, :] = vels[:, i, 0, :].clone().detach()
            cur_rot_vels[:, 0, :] = rot_vels[:, i, 0, :].clone().detach()
            # Take one step through world
            cur_pos, cur_rots, cur_vels, cur_rot_vels = self._integrate_through_world_model(cur_pos, cur_rots, cur_vels, cur_rot_vels, cur_heights, cur_up_dir, kin_rot_t[:, i, ...], kin_rvel_t[:, i, ...])
            # Update loss
            step_loss, wp, wv, wrvel, wr, raw_losses = self.char_state_loss(cur_pos[:, 1:, :],
                                                        positions[:, i+1, 1:, :],
                                                        cur_rots[:, 1:, :],
                                                        rotations[:, i+1, 1:, :],
                                                        cur_vels[:, 1:, :], 
                                                        vels[:, i+1, 1:, :], 
                                                        cur_rot_vels[:, 1:, :], 
                                                        rot_vels[:, i+1, 1:, :])
            loss += step_loss
            wpos_loss += raw_losses[0]
            wvel_loss += raw_losses[1]
            wang_loss += raw_losses[2]
            wrot_loss += raw_losses[3]
        update_stats = {'World Model/wpos_loss': wpos_loss.item(),
                         'World Model/wvel_loss': wvel_loss.item(),
                         'World Model/wrvel_loss': wang_loss.item(),
                         'World Model/wrot_loss': wrot_loss.item(),
                         'World Model/total loss': loss.item(),
                         'World Model/learning_rate': self.wm_lr}

        self.world_model_optimzer.zero_grad()
        loss.backward()
        self.world_model_optimzer.step()
        return update_stats

    @timed
    def char_state_loss(self, pos1, pos2, rot1, rot2, vel1, vel2, rvel1, rvel2):
        nans = [torch.isnan(t).any() for t in [pos1, pos2, rot1, rot2, vel1, vel2, rvel1, rvel2]]
        if any(nans):
            print("Nan in char_state_loss!")
            print(nans)
        # We want every loss to give roughly equal contribution
        # to do this, we make sure that, eg, w_pos_loss * pos_loss = total_loss / 4
        # w_pos_loss = total_loss/(pos_loss * 4)
        # dim=(1,2) because we want to sum over the 3 dimensions of each bone, and then sum over the bones
        raw_pos_l = torch.mean(torch.sum(torch.abs(pos1-pos2), dim =(1,2)))
        raw_vel_l = torch.mean(torch.sum(torch.abs(vel1-vel2), dim =(1,2)))
        raw_rvel_l = torch.mean(torch.sum(torch.abs(rvel1-rvel2), dim =(1,2)))
        # quat_diffs = SupertrackUtils.normalize_quat(pyt.quaternion_multiply(rot1, pyt.quaternion_invert(rot2)))
        # mat_diffs = pyt.quaternion_to_matrix(quat_diffs).reshape(-1, 3, 3)
        # raw_rot_l = torch.mean(torch.sum(torch.abs(pyt.so3_rotation_angle(mat_diffs))))
        quat_diffs = SupertrackUtils.normalize_quat(pyt.quaternion_multiply(rot1, pyt.quaternion_invert(rot2)))
        # batch_size, num_bones, _ = quat_diffs.shape
        # quat_logs = pyt.so3_log_map(pyt.quaternion_to_matrix(quat_diffs).reshape(-1, 3, 3)).reshape(batch_size, num_bones, 3)
        # raw_rot_l = torch.mean(torch.sum(torch.abs(quat_logs), dim=(1,2)))
        raw_rot_l = torch.mean(torch.sum(torch.abs(quat_diffs), dim=(1,2)))
        # print(raw_rot_l)
        # Make sure they all contribute equally to the total loss
        # total_loss = raw_pos_l + raw_vel_l + raw_rvel_l + raw_rot_l 
        # losses = [raw_pos_l, raw_vel_l, raw_rvel_l, raw_rot_l]
        # Avoid divide by 0
        # weights = [total_loss / max(1, (4 * l)) for l in losses]
        # wp, wv, wrvel, wr = weights
        wp = wv = wrvel = wr = 1
        loss = wp*raw_pos_l + wv*raw_vel_l + wrvel*raw_rvel_l + wr*raw_rot_l
        return loss, wp, wv, wrvel, wr, (raw_pos_l, raw_vel_l, raw_rvel_l, raw_rot_l)


    def _integrate_through_world_model(self,
                                    pos: torch.Tensor, # shape [batch_size, num_bones, 3]
                                    rots: torch.Tensor, # shape [batch_size, num_bones, 4]
                                    vels: torch.Tensor,  # shape [batch_size, num_bones, 3]
                                    rvels: torch.Tensor, # shape [batch_size, num_bones, 3]
                                    heights: torch.Tensor, # shape [batch_size, num_bones]
                                    up_dir: torch.Tensor,  # shape [batch_size, 3]
                                    kin_rot_t: torch.Tensor, # shape [batch_size, num_t_bones, 6] num_t_bones = 16 
                                    kin_rvel_t: torch.Tensor, # shape [batch_size, num_t_bones, 3]
                                    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Integrate a character state through the world model
        Params should be in world space, and will be returned in world space.
        :param exclude_root: Whether to exclude the root bone from the output, useful for training the policy since we 
        don't want to compute loss with root bone
        """
        batch_size = pos.shape[0]
        input = torch.cat((*SupertrackUtils.local(pos, rots, vels, rvels, heights, up_dir),
                            kin_rot_t.reshape(batch_size, -1),
                            kin_rvel_t.reshape(batch_size, -1)), dim = -1)
        output = self._world_model(input)
        if torch.isnan(output).any():
            print("World model output has nan!")
            print(f"Input has nan: {torch.isnan(input).any()}")
            if torch.isnan(input).any():
                input_parts = [*SupertrackUtils.local(pos, rots, vels, rvels, heights, up_dir), kin_rot_t.reshape(batch_size, -1), kin_rvel_t.reshape(batch_size, -1)]
                [print(torch.isnan(t).any()) for t in input_parts]
            pdb.set_trace()
        local_accel, local_rot_accel = SupertrackUtils.split_world_model_output(output)
        # Convert to world space
        root_rot = rots[:, 0:1, :]
        accel = pyt.quaternion_apply(root_rot, local_accel) 
        rot_accel = pyt.quaternion_apply(root_rot, local_rot_accel)

        padding_for_root_bone = torch.zeros((batch_size, 1, 3))
        accel = torch.cat((padding_for_root_bone, accel), dim=1)
        rot_accel = torch.cat((padding_for_root_bone, rot_accel), dim=1)
        # Integrate using Semi-Implicit Euler
        # We use semi-implicit so the model can influence position and velocity losses for the first timestep
        # Also that's what the paper does
        vels = vels + accel*self.dtime
        rvels = rvels + rot_accel*self.dtime

        pos = pos + vels*self.dtime
        rots_before = rots.clone().detach()
        rots = pyt.quaternion_multiply(pyt.axis_angle_to_quaternion(rvels*self.dtime) , rots)
        avg_quat_norm = torch.mean(torch.norm(rots, p=2, dim=-1))
        # print(f"Average norm for quaternions: {avg_quat_norm}")
        # check if quat norm is 0
        if math.isclose(avg_quat_norm, 0):
            print("Quat norm is 0!")
            pdb.set_trace()
        # rots = SupertrackUtils.normalize_quat(rots)
        nans = [torch.isnan(t).any() for t in [pos, rots, vels, rvels]]
        if any(nans):
            print("Nan in world model integration!")
            print(nans)
        
        return pos, rots, vels, rvels
        
    @timed
    def update_policy(self, batch: AgentBuffer, batch_size: int, raw_window_size: int) -> Dict[str, float]: 
        window_size = raw_window_size + 1
        if (batch.num_experiences // window_size != batch_size):
                raise Exception(f"Unexpected update size - expected len of batch to be {window_size} * {batch_size}, received {batch.num_experiences}")

        # sim_char_tensors = [data.as_tensors() for data in batch[BufferKey.SUPERTRACK_DATA].sim_char_state]
        st_data = [batch[BufferKey.SUPERTRACK_DATA][i] for i in range(batch.num_experiences)]
        sim_char_tensors = [st_datum.sim_char_state.as_tensors for st_datum in st_data]
        kin_char_tensors = [st_datum.kin_char_state.as_tensors for st_datum in st_data]

        s_pos, s_rots, s_vels, s_rvels, s_h, s_up = self._convert_to_usable_tensors(sim_char_tensors, batch_size, window_size)
        k_pos, k_rots, k_vels, k_rvels, k_h, k_up = self._convert_to_usable_tensors(kin_char_tensors, batch_size, window_size)

        cur_spos = s_pos[:, 0, ...].clone().detach()
        cur_srots = s_rots[:, 0, ...].clone().detach()
        cur_svels = s_vels[:, 0, ...].clone().detach()
        cur_srvels = s_rvels[:, 0, ...].clone().detach()

        kin_pre_targets = [st_datum.pre_targets.as_tensors for st_datum in st_data]
        # It's okay to use pre target vels because we're not predicting velocity targets right now
        pre_target_rots, pre_target_vels =  self._convert_pdtargets_to_usable_tensors(kin_pre_targets, batch_size, window_size, True)
        hn = lambda x : torch.isnan(x).any()
        loss = lpos = lvel = lrot = lang = lreg = lsreg = 0
        self.policy.actor.train()
        self._world_model.eval()
        for param in self._world_model.parameters():
            param.requires_grad = False
        for i in range(raw_window_size):
            local_kin = SupertrackUtils.local(k_pos[:, i, ...], k_rots[:, i, ...], k_vels[:, i, ...], k_rvels[:, i, ...], k_h[:, i, ...], k_up[:, i, ...])
            # Since the world model does not predict the root position, we have to copy it from the data and not use it in the loss function
            cur_spos[:, 0, :] = s_pos[:, i, 0, :]
            cur_srots[:, 0, :] = s_rots[:, i , 0, :]
            local_sim = SupertrackUtils.local(cur_spos, cur_srots, cur_svels, cur_srvels, s_h[:, i, ...], s_up[:, i, ...])
            # Predict PD offsets
            input = torch.cat((*local_kin, *local_sim), dim=-1)
            if hn(input):
                print(f"Input has nan! i: {i}")
                [print(hn(t)) for t in [*local_kin, *local_sim]]
                pdb.set_trace()
            action, runout, _ = self.policy.actor.get_action_and_stats([input], inputs_already_formatted=True)
            means = runout['means']
            output = action.continuous_tensor.reshape(batch_size, NUM_T_BONES, 3)
            # output = SupertrackUtils.convert_actions_to_quat(output, self.offset_scale)
            output =  pyt.axis_angle_to_quaternion(output * self.offset_scale)
            # Compute PD targets
            cur_kin_targets = pyt.quaternion_multiply(pre_target_rots[:, i, ...], output)
            if hn(cur_kin_targets):
                print(f"cur_kin_targets has nan! i: {i}")
                if hn(action.continuous_tensor.reshape(batch_size, NUM_T_BONES, 3)):
                    print("action has nan before being converted to quat")
                if hn(output):
                    print("Policy ouptput has nan after being converted to quat")
                pdb.set_trace()
            # Pass through world model
            cur_spos, cur_srots, cur_svels, cur_srvels = self._integrate_through_world_model(
                                                                cur_spos,
                                                                cur_srots,
                                                                cur_svels,
                                                                cur_srvels,
                                                                s_h[:, i, ...],
                                                                s_up[:, i, ...],
                                                                pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(cur_kin_targets)),
                                                                pre_target_vels[:, i, ...])
            # Compute losses
            """
            The difference between this prediction [of simulated state] and the target
            kinematic states K is then computed in the local space, and the
            losses used to update the weights of the policy
            """
            step_loss, wp, wv, wrvel, wr, raw_losses = self.char_state_loss(cur_spos[:, 1:, :], 
                                    k_pos[:, i+1, 1:, :],
                                    cur_srots[:, 1:, :],
                                    k_rots[:, i+1, 1:, :],
                                    cur_svels[:, 1:, :], 
                                    k_vels[:, i+1, 1:, :], 
                                    cur_srvels[:, 1:, :], 
                                    k_rvels[:, i+1, 1:, :])
            loss += step_loss
            lpos += raw_losses[0]
            lvel += raw_losses[1]
            lang += raw_losses[2]
            lrot += raw_losses[3]
            # Compute regularization losses
            step_lreg = torch.norm(means, p=2 ,dim=-1).mean()
            step_lsreg = torch.norm(means, p=1 ,dim=-1).mean()
            # Weigh regularization losses to contribute 1/100th of the other losses
            step_lreg /= 100
            step_lsreg /= 100
            lreg += step_lreg
            lsreg += step_lsreg
            
        update_stats = {"Policy/loss": loss.item(),
                        "Policy/pos_loss": lpos.item(),
                        "Policy/vel_loss": lvel.item(),
                        "Policy/rot_loss": lrot.item(),
                        "Policy/ang_loss": lang.item(),
                        "Policy/reg_loss": lreg.item(),
                        "Policy/sreg_loss": lsreg.item(),
                        "Policy/learning_rate": self.policy_lr}
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        self._world_model.train()
        for param in self._world_model.parameters():
            param.requires_grad = True
        return update_stats


    
    def _convert_to_usable_tensors(self, char_states_as_tensors: List[Tuple[torch.Tensor]],  batch_size: int, window_size: int):
        # Unholy python wizardry. We take the char_state_tensors in the form: [(pos1, rots1, vels1, ...) , (pos2, rots2, vels2, ...), ...]
        # we unpack it with * to make it as if we were passing the each tuple as a separate argument to zip: zip((pos1, rots1, vels1, ...), (pos2, rots2, vels2, ...), ...)
        # zip then TRANSPOSES our tuples to go along the axis of the first element of each tuple: (pos1, pos2, ...), (rots1, rots2, ...), (vels1, vels2, ...), ...)
        # We then STACK every tuple so : (pos1: [17, 3], pos2: [17, 3], ... pos_n), => tensor of shape [n, 17, 3]
        zipped_tensors = [torch.stack(data_tuple, dim=0) for data_tuple in zip(*char_states_as_tensors)]
        # We then reshape to [batch_size, window_size, num_bones, 3]
        # We pick this order because of the return order from CharState.as_tensors()
        pos, rots, vels, rvels, heights, up_dir = [tensor.reshape(batch_size, window_size, *list(tensor.shape)[1:]) for tensor in zipped_tensors]
        return pos, rots, vels, rvels, heights, up_dir
    
    def _convert_pdtargets_to_usable_tensors(self, pd_targets_as_tensors: List[Tuple[torch.Tensor]],  batch_size: int, window_size: int, normalize_quat: bool = True):
        zipped_targets = [torch.stack(data_tuple, dim=0) for data_tuple in zip(*pd_targets_as_tensors)]
        # We then reshape to [batch_size, window_size, num_t_bones, 3], getting to num_t_bones by removing the root bone
        rot_t, rvel_t = [tensor.reshape(batch_size, window_size, *list(tensor.shape)[1:])[:, :, 1:, :] for tensor in zipped_targets]
        if normalize_quat:
            return SupertrackUtils.normalize_quat(rot_t), rvel_t
        else:
            return rot_t, rvel_t

    def get_modules(self):
        modules = {
            # "Optimizer:WorldModel": self._world_model,
            # "Optimizer:world_model_optimzer": self.world_model_optimzer,
            # "Optimizer:policy_optimizer": self.policy_optimizer,
         }
        return modules
    
