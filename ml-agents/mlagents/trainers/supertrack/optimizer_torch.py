from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr

from mlagents.torch_utils import torch, nn, default_device
import pytorch3d.transforms as pyt
from mlagents.trainers.buffer import AgentBuffer, AgentBufferField, BufferKey, RewardSignalUtil
from mlagents.trainers.supertrack.supertrack_utils import NUM_BONES, CharState, SuperTrackDataField, SupertrackUtils
from mlagents.trainers.supertrack.world_model import WorldModelNetwork
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.layers import LinearEncoder
from mlagents.trainers.torch_entities.networks import Critic
from mlagents.trainers.torch_entities.networks import Actor
from mlagents_envs.base_env import ActionSpec, ObservationSpec

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.networks import ValueNetwork
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil

@attr.s(auto_attribs=True)
class SuperTrackSettings(OffPolicyHyperparamSettings):
    batch_size: int = 128
    buffer_size: int = 50000
    buffer_init_steps: int = 0
    tau: float = 0.005
    num_epoch: int = 3
    steps_per_update: float = 1
    save_replay_buffer: bool = False
    init_entcoef: float = 1.0
    
class TorchSuperTrackOptimizer(TorchOptimizer):
    dtime = 1 / 60

    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self._world_model = WorldModelNetwork(
            trainer_settings.world_model_network_settings
        )
        self._world_model.to(default_device())
        policy.actor.to(default_device())
        hyperparameters: SuperTrackSettings = cast(
            SuperTrackSettings, trainer_settings.hyperparameters
        )
        self.world_model_optimzer = torch.optim.Adam(self._world_model.parameters(), lr=hyperparameters.learning_rate)

    @timed
    def local(self, 
            cur_pos, # shape [batch_size, num_bones, 3]
            cur_rots, # shape [batch_size, num_bones, 4]
            cur_vels,   # shape [batch_size, num_bones, 3]
            cur_rot_vels, # shape [batch_size, num_bones, 3]
            cur_heights, # shape [batch_size, num_bones, 1]
            cur_up_dir): # shape [batch_size, num_bones]
        
        B = cur_pos.shape[0] # batch_size
        root_pos = cur_pos[:, 0:1 , :] # shape [batch_size, 1, 3]
        inv_root_rots = pyt.quaternion_invert(cur_pos[:, 0:1, :]) # shape [batch_size, 1, 4]
        local_pos = pyt.quaternion_apply(inv_root_rots, cur_pos[:, 1:, :] - root_pos) # shape [batch_size, num_bones, 3]
        local_rots = pyt.quaternion_multiply(inv_root_rots, cur_rots[:, 1:, :]) # shape [batch_size, num_bones, 4]
        two_axis_rots = pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(local_rots)) # shape [batch_size, num_bones, 6]
        local_vels = pyt.quaternion_apply(inv_root_rots, cur_vels[:, 1:, :]) # shape [batch_size, num_bones, 3]
        local_rot_vels = pyt.quaternion_apply(inv_root_rots, cur_rot_vels[:, 1:, :]) # shape [batch_size, num_bones, 3]

        return local_pos.view(B, -1),  two_axis_rots.view(B, -1), local_vels.view(B, -1), local_rot_vels.view(B, -1), cur_heights[:, 1:, :].view(B, -1), cur_up_dir[:, 1:, :]


    @timed
    def update_world_model(self, batch: AgentBuffer, batch_size: int, raw_window_size: int) -> Dict[str, float]:

        window_size = raw_window_size + 1
        if (batch.num_experiences // window_size != batch_size):
                raise Exception(f"Unexpected update size - expected len of batch to be {window_size} * {batch_size}, received {batch.num_experiences}")

        # sim_char_tensors = [data.to_tensors() for data in batch[BufferKey.SUPERTRACK_DATA].sim_char_state]
        st_data = [batch[BufferKey.SUPERTRACK_DATA][i] for i in range(batch.num_experiences)]
        sim_char_tensors = [st_datum.sim_char_state.to_tensors() for st_datum in st_data]

        # Unholy python wizardry. We take the st_tensors in the form: [(pos1, rots1, vels1, ...) , (pos2, rots2, vels2, ...), ...]
        # we unpack it with * to make it as if we were passing the each tuple as a separate argument to zip: zip((pos1, rots1, vels1, ...), (pos2, rots2, vels2, ...), ...)
        # zip then TRANSPOSES our tuples to go along the axis of the first element of each tuple: (pos1, pos2, ...), (rots1, rots2, ...), (vels1, vels2, ...), ...)
        zipped_sim_tensors = [torch.stack(data_tuple, dim=0) for data_tuple in zip(*sim_char_tensors)]
        # print([list(tensor.shape) for tensor in zipped_tensors])
        # reshape to [batch_size, window_size, ...]
        # print([tensor.shape[1:] for tensor in zipped_tensors])
        # print([list(tensor.shape)[1:] for tensor in zipped_tensors])

        positions, rotations, vels, rot_vels, heights, up_dir = [tensor.view(batch_size, window_size, *list(tensor.shape)[1:]) for tensor in zipped_sim_tensors]
        # positions, rotations, vels, rot_vels, heights, up_dir = list(zip(*sim_char_tensors))
        # positions = positions.view(batch_size, window_size, NUM_BONES, 3)
        # rotations = rotations.view(batch_size, window_size, NUM_BONES, 4)
        # vels = vels.view(batch_size, window_size, NUM_BONES, 3)
        # rot_vels = rot_vels.view(batch_size, window_size, NUM_BONES, 3)
        # heights = heights.view(batch_size, window_size, NUM_BONES, 1)
        # up_dir = up_dir.view(batch_size, window_size, 3)
        
        kin_target_tensors = [st_datum.post_targets.to_tensors() for st_datum in st_data]
        kin_rot_t, kin_rvel_t = [torch.stack(data_tuple, dim=0) for data_tuple in zip(*kin_target_tensors)]
        kin_rot_t =  pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(kin_rot_t)).view(batch_size, window_size, NUM_BONES, 6)
        kin_rvel_t = kin_rvel_t.view(batch_size, window_size, NUM_BONES, 3)

        cur_pos = positions[:, 0, ...].clone()
        cur_rots = rotations[:, 0, ...].clone()
        cur_vels = vels[:, 0, ...].clone()
        cur_rot_vels = rot_vels[:, 0, ...].clone()

        loss = 0
        wpos_loss = wvel_loss = wang_loss = wrot_loss = 0


        for i in range(window_size):
            cur_heights = heights[:, i, ...]
            cur_up_dir = up_dir[:, i, ...]
            input = torch.cat((*self.local(cur_pos, cur_rots, cur_vels, cur_rot_vels, cur_heights, cur_up_dir),
                            kin_rot_t[:, i, ...].view(batch_size, -1),
                            kin_rvel_t[:, i, ...].view(batch_size, -1)), 
                            dim=-1)
            output = self._world_model(input)
            local_accel, local_rot_accel = SupertrackUtils.split_world_model_output(output)
            # Convert to world space
            root_rot = cur_rots[:, 0:1, :]
            accel = pyt.quaternion_apply(root_rot, local_accel)
            rot_accel = pyt.quaternion_apply(root_rot, local_rot_accel)
            # Integrate
            cur_pos = cur_pos + self.dtime*cur_vels
            cur_rots = pyt.matrix_to_quaternion(pyt.axis_angle_to_matrix(cur_rot_vels*self.dtime) @ pyt.quaternion_to_matrix(cur_rots))
            cur_vels = cur_vels + self.dtime*accel
            cur_rot_vels =  pyt.matrix_to_axis_angle(pyt.axis_angle_to_matrix(cur_rot_vels) @ pyt.axis_angle_to_matrix(self.dtime*rot_accel))
            # Update loss
            loss, wp, wv, wa, wr = self.world_model_loss(cur_pos, positions[:, i, ...], cur_rots, rotations[:, i, ...], cur_vels, vels[:, i, ...], cur_rot_vels, rot_vels[:, i, ...])
            loss += loss
            wpos_loss += wp
            wvel_loss += wv
            wang_loss += wa
            wrot_loss += wr
        update_stats = {'wpos_loss': wpos_loss, 'wvel_loss': wvel_loss, 'wang_loss': wang_loss, 'wrot_loss': wrot_loss}
        # We want every loss to give roughly equal contribution
        # to do this, we make sure that, eg, w_pos_loss * pos_loss = total_loss / 4
        # w_pos_loss = total_loss/(pos_loss * 4)
        # pos_loss = 
        # ...
        self.world_model_optimzer.zero_grad()
        loss.backward()
        self.world_model_optimzer.step()
        return update_stats

    @timed
    def world_model_loss(self, pos1, pos2, rot1, rot2, vel1, vel2, rvel1, rvel2):
        wpos = wvel = wrot = wang = 0.1
        wp = wpos*torch.mean(torch.sum(torch.abs(pos1-pos2), dim = -1))
        wv += wvel*torch.mean(torch.sum(torch.abs(vel1-vel2), dim = -1))
        wa += wang*torch.mean(torch.sum(torch.abs(rvel1-rvel2), dim = -1))
        wr += wrot*torch.mean(torch.sum(
                            pyt.so3_rotation_angle(
                                pyt.quaternion_to_matrix(
                                    pyt.quaternion_multiply(rot1, pyt.quaternion_invert(rot2))))))
        loss = wp + wv + wa + wr
        return loss, wp, wv, wa, wr

        
    @timed
    def update_policy(self, batch: AgentBuffer, window_size: int) -> Dict[str, float]: 
        pass
    
    def get_modules(self):
        modules = {
            "Optimizer:WorldModel": self._world_model,
         }
        return modules
    


class PolicyNetworkBody(nn.Module):
    def __init__(
            self,
            network_settings: NetworkSettings,
    ):
        super().__init__()
        self.network_settings = network_settings
        self.normalize = network_settings.normalize
        self.h_size = network_settings.hidden_units
        self.input_size = self.network_settings.input_size
        if (self.input_size == -1):
            raise Exception("SuperTrack Policy Network created without input_size designated in yaml file")
        
        # Used to normalize inputs
        self._obs_encoder : nn.Module = VectorInput(self.input_size, self.normalize)
        self._body_encoder = LinearEncoder(
            self.network_settings.input_size,
            self.network_settings.num_layers,
            self.h_size)

    @property
    def memory_size(self) -> int:
        return 0
        
    def update_normalization(self, buffer: AgentBuffer) -> None:
        # self._obs_encoder.update_normalization(buffer.input)
        pass

    def forward(self, inputs: torch.Tensor):
        # if len(inputs) != 1:
        #     raise Exception(f"SuperTrack policy network body initialized with multiple observations: {len(inputs)} ")
        encoded_self = self._obs_encoder(inputs)
        encoding = self._body_encoder(encoded_self)
        return encoding

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
        self.network_body.update_normalization(buffer)


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

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        encoding = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )

        log_probs, entropies = self.action_model.evaluate(encoding, masks, actions)
        run_out = {}
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        return run_out

    def get_action_and_stats(
        self,
        inputs: List[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
        deterministic=False,
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
        policy_input = SupertrackUtils.process_raw_observations_to_policy_input(inputs[0])
        encoding = self.network_body(policy_input)
        action, log_probs, entropies = self.action_model(encoding, None) 
        run_out = {}
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies

        return action, run_out, None