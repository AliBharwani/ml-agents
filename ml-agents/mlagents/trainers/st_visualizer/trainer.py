from collections import defaultdict
import copy
from datetime import datetime
import itertools
from typing import Dict, Tuple, cast
import os
from mlagents.st_buffer import CharTypePrefix, CharTypeSuffix, PDTargetPrefix, PDTargetSuffix, STBuffer

from mlagents.trainers.st_visualizer.optimizer_torch import STVisualizationActor
from mlagents.trainers.stats import StatsPropertyType
from mlagents.trainers.supertrack.supertrack_utils import STSingleBufferKey, SupertrackUtils, nsys_profiler
from mlagents.trainers.trainer.trainer import Trainer
from mlagents.trainers.trajectory import Trajectory

from mlagents_envs.base_env import BehaviorSpec

from mlagents.trainers.policy.torch_policy import TorchPolicy

from mlagents.trainers.model_saver.torch_model_saver import TorchModelSaver
from mlagents.trainers.policy.checkpoint_manager import ModelCheckpoint

from mlagents_envs.logging_util import get_logger
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod
from mlagents.trainers.policy import Policy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trainer.rl_trainer import ProfilerState, RLTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TorchSettings, TrainerSettings
from mlagents.trainers.supertrack.optimizer_torch import SuperTrackPolicyNetwork, TorchSuperTrackOptimizer, SuperTrackSettings
from mlagents.torch_utils import torch, default_device

logger = get_logger(__name__)

TRAINER_NAME = "st_visualizer"

class STVisualizerTrainer(Trainer):
    """
    This is an implementation of the SuperTrack world-based RL algorithm
    """
    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
        stats_reporter_override = None,
        run_log_path = "",
    ):
        """
        Responsible for collecting experiences and training SuperTrack model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            trainer_settings,
            training,
            load,
            artifact_path,
            reward_buff_cap,
            run_log_path=run_log_path,
        )

        self.seed = seed
        self.policy: TorchPolicy = None  # type: ignore
        self.optimizer: TorchSuperTrackOptimizer = None  # type: ignore
        # self.hyperparameters: SuperTrackSettings = cast(
        #     SuperTrackSettings, trainer_settings.hyperparameters
        # )
        self.model_saver = TorchModelSaver(  # type: ignore
            self.trainer_settings, self.artifact_path, self.load
        )
        self._step = 0
        # self.update_steps = 0


    # We need to delay initialization because the trainer is constructed in the main process but trains
    # in a sub process, and if it is using CUDA then CUDA tensors should be initialized in the subprocess as well
    def _initialize(self) -> None:
        self.optimizer._init_world_model()
        
        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()
        self._step = self.policy.get_current_step()
        self.update_steps = self.policy.get_current_training_iteration()


    def _checkpoint(self) -> ModelCheckpoint:
        """
        Writes a checkpoint model to memory
        Overrides the default to save the replay buffer.
        """
        output_filepath =  os.path.join(self.model_saver.model_path, f"WorldModel-{self.update_steps}")
        final_export_path = self.optimizer.export_world_model(output_filepath)
        ckpt = super()._checkpoint(addtl_paths=[final_export_path])
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()
        return ckpt

    def save_model(self) -> None:
        """
        Saves the final training model to memory
        Overrides the default to save the replay buffer.
        """
        pass

    def _is_ready_update(self) -> bool:
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not _update_policy() can be run
        """
        return False

    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy
    ) -> None:
        """
        Adds policy to trainer.
        """
        if self.policy:
            logger.warning(
                "Your environment contains multiple teams, but {} doesn't support adversarial games. Enable self-play to \
                    train adversarial games.".format(
                    self.__class__.__name__
                )
            )
        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy
        self.optimizer = self.create_optimizer()
        self._initialize()
        # Give policy actor a copy of the world model on CPU
        cpu_world_model = copy.deepcopy(self.optimizer._world_model).to('cpu')
        # policy.actor.world_model = cpu_world_model
        policy.actor.world_model = copy.deepcopy(self.optimizer._world_model)
        # Needed to resume loads properly
        self._step = policy.get_current_step()
        

    def _update_policy(self, max_update_iterations : int = 512) -> bool:
        """
        Uses update_buffer to update the policy. We sample the update_buffer and update
        until the steps_per_update ratio is met.
        """
        return False


    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the replay buffer.
        """
        pass

    def create_optimizer(self) -> TorchOptimizer:
        return TorchSuperTrackOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore


    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec, torch_settings: TorchSettings
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and Supertrack hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        actor_cls = STVisualizationActor
        actor_kwargs = {"conditional_sigma": False, "tanh_squash": False, "clip_action": self.trainer_settings.clip_action, 
                           "policy_includes_global_data": self.trainer_settings.hyperparameters.policy_includes_global_data,
                           "st_debug": torch_settings.st_debug}

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.policy_network_settings,
            actor_cls,
            actor_kwargs,
            split_on_cpugpu=self.multiprocess,
        )

        return policy

    def get_policy(self, name_behavior_id: str) -> Policy:
        """
        Gets policy from trainer associated with name_behavior_id
        :param name_behavior_id: full identifier of policy
        """
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

    def end_episode(self) -> None:
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        pass

    def advance(self, profiling_enabled = False) -> None:
        pass