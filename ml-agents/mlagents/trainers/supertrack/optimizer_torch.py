from typing import Any, Dict, List, Optional, Tuple, Union, cast

from mlagents.trainers.settings import NetworkSettings, OffPolicyHyperparamSettings
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.torch_entities.action_model import ActionModel
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
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        return
    
    def get_modules(self):
        modules = { }
        return modules
    
    @property
    def critic(self):
        raise Exception("Super Track Optimizer critic property called - should never happen!")
    

class PolicyNetworkBody(nn.Module):
     def __init__(
               self,
               network_settings: NetworkSettings,
     ):
          nn.Module.__init__(self)
          

class SuperTrackPolicyNetwork(nn.Module, Actor):
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__()
        self.network_body = PolicyNetworkBody
        self.action_spec = action_spec
        self.encoding_size = network_settings.hidden_units
        action_spec.continuous_size = network_settings.output_size
        self.action_model = ActionModel(
            self.encoding_size,
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash,
            deterministic=network_settings.deterministic,
        )

    
    def update_normalization(self, buffer: AgentBuffer) -> None:
        self.network_body.update_normalization(buffer)

    def forward(
        self,
        inputs: List[torch.Tensor],
    ) -> Tuple[Union[int, torch.Tensor], ...]:
        pass

    def get_stats(
    self,
    inputs: List[torch.Tensor],
    actions: AgentAction,
    masks: Optional[torch.Tensor] = None,
    memories: Optional[torch.Tensor] = None,
    sequence_length: int = 1,
    ) -> Dict[str, Any]:
        encoding, actor_mem_outs = self.network_body(
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
        _masks: Optional[torch.Tensor] = None,
        _memories: Optional[torch.Tensor] = None,
        _sequence_length: int = 1,
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
        action_out = self.network_body(inputs)
        run_out = {}
        return action_out, run_out, torch.Tensor([])
