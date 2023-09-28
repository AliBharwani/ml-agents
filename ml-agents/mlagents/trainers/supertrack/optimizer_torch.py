
from typing import Any, Dict, List, Optional, Tuple, Union, cast


from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr
import pdb
from mlagents.torch_utils import torch, nn, default_device
import pytorch3d.transforms as pyt
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.supertrack.supertrack_utils import  POLICY_INPUT_LEN, SupertrackUtils
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
    
    
class TorchSuperTrackOptimizer(TorchOptimizer):
    dtime = 1 / 60

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self._world_model = WorldModelNetwork(
            trainer_settings.world_model_network_settings
        )
        self._world_model.to(default_device())
        policy.actor.to(default_device())
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

    @timed
    def update_world_model(self, batch: AgentBuffer, batch_size: int, raw_window_size: int) -> Dict[str, float]:

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

        cur_pos = positions[:, 0, ...].clone()
        cur_rots = rotations[:, 0, ...].clone()
        cur_vels = vels[:, 0, ...].clone()
        cur_rot_vels = rot_vels[:, 0, ...].clone()

        for tensor, name in [(cur_pos, 'cur_pos'), (cur_rots, 'cur_rots'), (cur_vels, 'cur_vels'), (cur_rot_vels, 'cur_rot_vels')]:
            if torch.isnan(tensor).any():
                raise Exception(f"Nan in {name} at start of update_world_model")

        loss = 0
        wpos_loss = wvel_loss = wang_loss = wrot_loss = 0
        for i in range(raw_window_size):
            cur_heights = heights[:, i, ...]
            cur_up_dir = up_dir[:, i, ...]
            # Since the world model does not predict the root position, we have to copy it from the data and not use it in the loss function
            cur_pos[:, 0, :] = positions[:, i, 0, :]
            cur_rots[:, 0, :] = rotations[:, i , 0, :]
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
        # We want every loss to give roughly equal contribution
        # to do this, we make sure that, eg, w_pos_loss * pos_loss = total_loss / 4
        # w_pos_loss = total_loss/(pos_loss * 4)
        raw_pos_l = torch.mean(torch.sum(torch.abs(pos1-pos2), dim = -1))
        raw_vel_l = torch.mean(torch.sum(torch.abs(vel1-vel2), dim = -1))
        raw_rvel_l = torch.mean(torch.sum(torch.abs(rvel1-rvel2), dim = -1))
        quat_diffs = SupertrackUtils.normalize_quat(pyt.quaternion_multiply(rot1, pyt.quaternion_invert(rot2)))
        mat_diffs = pyt.quaternion_to_matrix(quat_diffs).reshape(-1, 3, 3)
        raw_rot_l = torch.mean(torch.sum(torch.abs(pyt.so3_rotation_angle(mat_diffs))))
        # Make sure they all contribute equally to the total loss
        total_loss = raw_pos_l + raw_vel_l + raw_rvel_l + raw_rot_l 
        losses = [raw_pos_l, raw_vel_l, raw_rvel_l, raw_rot_l]
        weights = [total_loss / (4 * l) for l in losses]
        wp, wv, wrvel, wr = weights
        loss = wp*raw_pos_l + wv*raw_vel_l + wrvel*raw_rvel_l + wr*raw_rot_l
        return loss, wp, wv, wrvel, wr, (raw_pos_l, raw_vel_l, raw_rvel_l, raw_rot_l)


    def _integrate_through_world_model(self,
                                    pos: torch.Tensor, # shape [batch_size, num_bones, 3]
                                    rots: torch.Tensor, # shape [batch_size, num_bones, 4]
                                    vels: torch.Tensor,  # shape [batch_size, num_bones, 3]
                                    rvels: torch.Tensor, # shape [batch_size, num_bones, 3]
                                    heights: torch.Tensor, # shape [batch_size, num_bones]
                                    up_dir: torch.Tensor,  # shape [batch_size, 3]
                                    kin_rot_t: torch.Tensor, # shape [batch_size, num_t_bones, 6] num_t_bones = 17 
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
        local_accel, local_rot_accel = SupertrackUtils.split_world_model_output(output)
        # Convert to world space
        root_rot = rots[:, 0:1, :]
        accel = pyt.quaternion_apply(root_rot, local_accel)
        rot_accel_q = pyt.quaternion_multiply(root_rot, pyt.axis_angle_to_quaternion(local_rot_accel))
        rot_accel = pyt.quaternion_to_axis_angle(SupertrackUtils.normalize_quat(rot_accel_q))
        padding_for_root_bone = torch.zeros((batch_size, 1, 3))
        accel = torch.cat((padding_for_root_bone, accel), dim=1)
        rot_accel = torch.cat((padding_for_root_bone, rot_accel), dim=1)
        # Integrate
        pos = pos + self.dtime*vels
        rots = pyt.quaternion_multiply(pyt.axis_angle_to_quaternion(rvels*self.dtime) , rots)
        vels = vels + self.dtime*accel
        rvels =  pyt.matrix_to_axis_angle(pyt.axis_angle_to_matrix(rvels) @ pyt.axis_angle_to_matrix(self.dtime*rot_accel))
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

        cur_spos = s_pos[:, 0, ...].clone()
        cur_srots = s_rots[:, 0, ...].clone()
        cur_svels = s_vels[:, 0, ...].clone()
        cur_srvels = s_rvels[:, 0, ...].clone()

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
                print(f"Input nan: {hn(input)}")
                pdb.set_trace()
            action, runout, _ = self.policy.actor.get_action_and_stats([input], inputs_already_formatted=True)
            means = runout['means']
            output = action.continuous_tensor.reshape(batch_size, -1, 3)
            output = SupertrackUtils.convert_actions_to_quat(output, self.offset_scale)
            # Compute PD targets
            cur_kin_targets = pyt.quaternion_multiply(pre_target_rots[:, i, ...], output)
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
            "Optimizer:WorldModel": self._world_model,
            "Optimizer:world_model_optimzer": self.world_model_optimzer,
            "Optimizer:policy_optimizer": self.policy_optimizer,
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
        if self.network_settings.normalize:
            _layers += [nn.LayerNorm(self.input_size)]

        _layers += [LinearEncoder(
            self.network_settings.input_size,
            self.network_settings.num_layers,
            self.network_settings.hidden_units)]
        self.layers = nn.Sequential(*_layers)

    @property
    def memory_size(self) -> int:
        return 0

    def forward(self, inputs: torch.Tensor):
        return self.layers(inputs)

class SuperTrackPolicyNetwork(nn.Module, Actor):
    MODEL_EXPORT_VERSION = 1
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
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
        )

    def update_normalization(self, buffer: AgentBuffer) -> None:
        pass # Not needed because we use batchnorm

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

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        deterministic=False,
        inputs_already_formatted=False,
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
        # pdb.set_trace()
        supertrack_data = None
        if inputs_already_formatted:
            policy_input = inputs[0]
        else:
            supertrack_data = SupertrackUtils.parse_supertrack_data_field(inputs[0])
            policy_input = SupertrackUtils.process_raw_observations_to_policy_input(supertrack_data)
        if policy_input.shape[-1] != POLICY_INPUT_LEN:
            raise Exception(f"SuperTrack policy network body forward called with policy input of length {policy_input.shape[-1]}, expected {POLICY_INPUT_LEN}")
        encoding = self.network_body(policy_input)
        action, log_probs, entropies, means = self.action_model(encoding, None) 
        hn = lambda x: torch.isnan(x).any()
        if hn(policy_input) or hn(encoding) or hn(action.continuous_tensor):
            print(f"Policy_input nan: {hn(policy_input)}, encoding nan: {hn(encoding)}, action nan: {hn(action.continuous_tensor)}")
            pdb.set_trace()
        run_out = {}
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["supertrack_data"] = supertrack_data
        run_out["means"] = means
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies

        return action, run_out, None