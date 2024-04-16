import json
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, cast
from mlagents.st_buffer import CharTypePrefix, CharTypeSuffix, PDTargetPrefix, PDTargetSuffix, STBuffer

from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr
import pdb
from mlagents.torch_utils import torch, nn, default_device
from mlagents.trainers.supertrack.optimizer_torch import SuperTrackPolicyNetwork
import numpy as np
import pytorch3d.transforms as pyt
from mlagents.trainers.supertrack.supertrack_utils import  NUM_BONES, NUM_T_BONES, POLICY_INPUT_LEN, POLICY_OUTPUT_LEN, TOTAL_OBS_LEN, STSingleBufferKey, SupertrackUtils, nsys_profiler
from mlagents.trainers.supertrack.world_model import WorldModelNetwork
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.layers import Initialization, LinearEncoder
from mlagents.trainers.torch_entities.networks import Actor
from mlagents_envs.base_env import ActionSpec, ActionTuple, ObservationSpec
from mlagents_envs.logging_util import get_logger

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
)
from mlagents.trainers.torch_entities.agent_action import AgentAction


class STVisualizationActor(SuperTrackPolicyNetwork):


    nframes = 8
    dtime = 1 / 60

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
        _action_spec = ActionSpec(continuous_size=48, discrete_branches=())
        super().__init__(observation_specs, network_settings, _action_spec, conditional_sigma=conditional_sigma, tanh_squash=tanh_squash,
                         device=device,clip_action=clip_action)

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
        # should be shape [num_obs_types (1), num_agents, POLICY_INPUT_LEN]
        policy_input = inputs[0][:, :TOTAL_OBS_LEN] # This 
        world_model_data = inputs[0][0, TOTAL_OBS_LEN:]
        # pdb.set_trace()
        supertrack_data = SupertrackUtils.parse_supertrack_data_field(policy_input)
        policy_input = SupertrackUtils.process_raw_observations_to_policy_input(supertrack_data)
        encoding = self.network_body(policy_input)
        action, log_probs, entropies, means = self.action_model(encoding, None) 
        run_out = {}
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.

        num_agents = action.continuous_tensor.shape[0]
        if num_agents != 1:
            raise Exception("More than one agent not compatible with visualizer")
        # pdb.set_trace()
        parsed_wm_data = self.parse_world_model_data(world_model_data)
        predicted_bone_pos, predicted_bone_rots = self.get_predictions_from_wm(supertrack_data, parsed_wm_data, action.continuous_tensor)
        # flatten
        final_flat_tensor = torch.cat((predicted_bone_pos, predicted_bone_rots), dim=2).reshape(-1)
        final_tensor = torch.cat((action.continuous_tensor, torch.unsqueeze(final_flat_tensor,0)), dim=-1)
        run_out["env_action"] = action.to_action_tuple(
            clip=self.action_model.clip_action
        )
        run_out["env_action"] = ActionTuple(continuous=final_tensor.numpy())
        # For some reason, sending CPU tensors causes the training to hang
        # This does not occur with CUDA tensors of numpy ndarrays
        # if supertrack_data is not None:
        #     run_out["supertrack_data"] = supertrack_data
        if return_means:
            run_out["means"] = means
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies

        return action, run_out, None
    

    def parse_world_model_data(self, world_model_data): # shape [952]
        nframes = self.nframes
        # B = 1
        # root_poses = torch.empty(nframes, 3)
        # root_rots = torch.empty(nframes, 4)
        k_pos = torch.empty(nframes, NUM_BONES, 3)
        k_rot = torch.empty(nframes, NUM_BONES, 4)
        k_vel = torch.empty(nframes, NUM_BONES, 3)
        k_rvel = torch.empty(nframes, NUM_BONES, 3)
        pd_rots = torch.empty(nframes, NUM_T_BONES, 4)
        pd_rvels = torch.empty(nframes, NUM_T_BONES, 3)
        
        cur_pd_rots = torch.empty(NUM_T_BONES, 4)
        cur_pd_rvels = torch.empty(NUM_T_BONES, 3)
        idx = 0
        for i in range(NUM_T_BONES):
            cur_pd_rots[i, :] = world_model_data[idx:idx+4]
            idx += 4
            cur_pd_rvels[i, :] = world_model_data[idx:idx+3]
            idx += 3
        
        for frame in range(nframes):
            k_pos[frame, 0, :] = world_model_data[idx:idx+3]
            idx += 3
            k_rot[frame, 0, :] = world_model_data[idx:idx+4]
            idx += 4
            for bone in range(NUM_T_BONES):
                k_pos[frame, bone + 1, :] = world_model_data[idx:idx+3]
                idx += 3
                k_rot[frame, bone + 1, :] = world_model_data[idx:idx+4]
                idx += 4
                k_vel[frame, bone + 1, :] = world_model_data[idx:idx+4]
                idx += 4
                k_rvel[frame, bone + 1, :] = world_model_data[idx:idx+4]
                idx += 4
                pd_rots[frame, bone + 1, :] = world_model_data[idx:idx+4]
                idx += 4
                pd_rvels[frame, bone + 1, :] = world_model_data[idx:idx+3]
                idx += 3
        return [cur_pd_rots, cur_pd_rvels, k_pos, k_rot, k_vel, k_rvel, pd_rots, pd_rvels]
        # Add batch dim
        # return [torch.unsqueeze(t) for t in [root_poses, root_rots, pd_rots, pd_rvels]]
    
    def get_predictions_from_wm(self, st_data, world_model_data, policy_action):
        nframes = self.nframes
        predicted_bone_poses = torch.empty(nframes, NUM_BONES, 3) 
        predicted_bone_rots = torch.empty(nframes, NUM_BONES, 4) 
        B = 1

        cur_pd_rots, cur_pd_rvels, k_pos, k_rot, k_vel, k_rvel, pd_rots, pd_rvels = world_model_data

        # Gives us a list of tensors of shape [(pos, rots, etc) of len batch_size ]
        sim_inputs = [st_datum.sim_char_state.values() for st_datum in st_data]
        # Convert them to [batch_size, num_bones, 3] for pos, [batch_size, num_bones, 4] for rots, etc
        sim_state = [torch.stack(t) for t in zip(*sim_inputs)]
        cur_kin_targets = SupertrackUtils.apply_policy_action_to_pd_targets(cur_pd_rots, policy_action)
        
        for i in range(nframes):
            # Predict next state with world model
            # pdb.set_trace()
            next_sim_state = SupertrackUtils.integrate_through_world_model(self.world_model, self.dtime, *sim_state,
                                                                pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(cur_kin_targets)),
                                                                cur_pd_rvels)
            next_spos, next_srots, next_svels, next_srvels = next_sim_state
            # Copy over ground truth spos / srots since world model does not predict world location
            next_spos[:, 0] = k_pos[i, 0]
            next_srots[:, 0] = k_rot[i, 0]

            predicted_bone_poses[i, ...] = next_spos.detach().clone()
            predicted_bone_rots[i, ...] = next_srots.detach().clone()
            sim_state = next_spos, next_srots, next_svels, next_srvels

            # Apply offsets from policy
            pre_offset_pd_target_rots = pd_rots[i]
            kin_state = k_pos[i], k_rot[i], k_vel[i], k_rvel[i]
            offset = self.get_policy_action(sim_state, kin_state)

            cur_kin_targets = SupertrackUtils.apply_policy_action_to_pd_targets(pre_offset_pd_target_rots, offset)
            cur_pd_rvels = pd_rvels[i]
        return predicted_bone_poses, predicted_bone_rots
    
    def get_policy_action(self, sim_state, kin_state):
        local_sim = SupertrackUtils.local(*sim_state) 
        local_kin = SupertrackUtils.local(*kin_state)
        # Not sure if I need to add batch dim before claling local...
        return torch.cat((*local_kin, *local_sim), dim=-1)