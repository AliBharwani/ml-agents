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

        supertrack_data = SupertrackUtils.parse_supertrack_data_field_batched(policy_input)
        policy_input = SupertrackUtils.process_raw_observations_to_policy_input(supertrack_data)
        encoding = self.network_body(policy_input)
        action, log_probs, entropies, means = self.action_model(encoding, None) 
        run_out = {}
        # This is the clipped action which is not saved to the buffer
        # but is exclusively sent to the environment.
        NUM_EXTRA_PADDING = 952 
        num_agents = action.continuous_tensor.shape[0]
        final_tensor = torch.cat((action.continuous_tensor, torch.zeros(num_agents, NUM_EXTRA_PADDING)), dim=-1)
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

        # APPEND TO ACTION VECTOR


        return action, run_out, None
    

    def parse_world_model_data(world_model_data): # shape [896]
        nframes = 8
        B = 1
        root_poses = torch.empty(nframes, 3)
        root_rots = torch.empty(nframes, 4)
        pd_rots = torch.empty(nframes, NUM_T_BONES, 4)
        pd_rvels = torch.empty(nframes, NUM_T_BONES, 3)
        idx = 0 
        for i in range(nframes):
            root_poses[i, :] = world_model_data[idx:idx+3]
            idx += 3
            root_rots[i, :] = world_model_data[idx:idx+4]
            idx += 4
            for j in range(NUM_T_BONES):
                pd_rots[i, j, :] = world_model_data[idx:idx+4]
                idx += 4
                pd_rvels[i, j, :] = world_model_data[idx:idx+3]
                idx += 3
        # Add batch dim
        return [torch.unsqueeze(t) for t in [root_poses, root_rots, pd_rots, pd_rvels]]
            

