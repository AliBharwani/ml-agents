from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.encoders import VectorInput
from mlagents.trainers.torch_entities.layers import Initialization, LinearEncoder
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
        self.normalize = network_settings.normalize
        self.h_size = network_settings.hidden_units
        self.input_size = network_settings.input_size
        if (self.input_size == -1):
            raise Exception("SuperTrack World Model created without input_size designated in yaml file")
        # MY TODO: Replace with 1DBatchNorm layer from pytorch
        self._obs_encoder : nn.Module = VectorInput(self.input_size, self.normalize)
        self._body_encoder = LinearEncoder(
            self.network_settings.input_size,
            self.network_settings.num_layers,
            self.h_size,
            Initialization.KaimingHeNormal,
            1,
            network_settings.activation_function)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_self = self._obs_encoder(inputs)
        encoding = self._body_encoder(encoded_self)
        return encoding