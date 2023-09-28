from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.layers import Initialization, LinearEncoder, linear_layer
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



class WorldModelNetwork(nn.Module):

    def __init__(
            self,
            network_settings: NetworkSettings,
    ):
        super().__init__()
        self.network_settings = network_settings
        h_size = network_settings.hidden_units
        input_size = network_settings.input_size
        if (input_size == -1):
            raise Exception("SuperTrack World Model created without input_size designated in yaml file")

        _layers = []

        # Normalize inputs if required
        if network_settings.normalize:
            _layers += [nn.LayerNorm(input_size)]
        
        _layers += [LinearEncoder(
            input_size,
            self.network_settings.num_layers - 1,
            h_size,
            Initialization.KaimingHeNormal,
            1,
            network_settings.activation_function)]

        _layers += [linear_layer(h_size, self.network_settings.output_size)]
        self.layers = nn.Sequential(*_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)