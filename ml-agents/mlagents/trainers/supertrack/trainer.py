# ## ML-Agent Learning (SAC)
# Contains an implementation of SAC as described in https://arxiv.org/abs/1801.01290
# and implemented in https://github.com/hill-a/stable-baselines

from collections import defaultdict
import copy
import threading
from typing import Dict, cast
import os


from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.supertrack.supertrack_utils import SupertrackUtils
from mlagents.trainers.trajectory import Trajectory
from mlagents_envs import logging_util

from mlagents_envs.base_env import BehaviorSpec

from mlagents.trainers.policy.torch_policy import TorchPolicy

import numpy as np
from mlagents.trainers.policy.checkpoint_manager import ModelCheckpoint

from mlagents_envs.logging_util import get_logger
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod
from mlagents_envs.timers import hierarchical_timer, timed
from mlagents.trainers.policy import Policy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TorchSettings, TrainerSettings
from mlagents.trainers.supertrack.optimizer_torch import SuperTrackPolicyNetwork, TorchSuperTrackOptimizer, SuperTrackSettings
from torch.profiler import profile, record_function, ProfilerActivity
from mlagents.torch_utils import torch, default_device

logger = get_logger(__name__)

BUFFER_TRUNCATE_PERCENT = 0.8

TRAINER_NAME = "supertrack"

class SuperTrackTrainer(RLTrainer):
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
        StatsReporterOverride = None,
        run_log_path = "",
    ):
        """
        Responsible for collecting experiences and training SAC model.
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
            StatsReporterOverride=StatsReporterOverride,
            run_log_path=run_log_path,
        )

        self.seed = seed
        self.policy: TorchPolicy = None  # type: ignore
        self.optimizer: TorchSuperTrackOptimizer = None  # type: ignore
        self.hyperparameters: SuperTrackSettings = cast(
            SuperTrackSettings, trainer_settings.hyperparameters
        )
        self._step = 0

         # Don't divide by zero
        self.update_steps = 1
        self.steps_per_update = self.hyperparameters.steps_per_update
        self.checkpoint_replay_buffer = self.hyperparameters.save_replay_buffer
        self.wm_window = self.trainer_settings.world_model_network_settings.training_window
        self.policy_window = self.trainer_settings.policy_network_settings.training_window
        self.effective_wm_window = self.wm_window + 1 # we include an extra piece of dating during training to simplify code
        self.effective_policy_window = self.policy_window + 1 
        self.batch_size = self.hyperparameters.batch_size
        self.wm_batch_size = self.trainer_settings.world_model_network_settings.batch_size
        self.policy_batch_size = self.trainer_settings.policy_network_settings.batch_size
        self.first_update = True

    @timed
    def _initialize(self):
        self.optimizer._init_world_model()
        
        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()
        self._step = self.policy.get_current_step()
        # MY TODO: MAKE SURE SUPER TRACK WORKS W CPU TRAINING
        if self.multiprocess:
            logger.info("intializing GPU instance of actor")
            actor_gpu = copy.deepcopy(self.policy.actor)
            actor_gpu.to("cuda")
            # actor_gpu.to("cpu")
            actor_gpu.train()
            self.optimizer.actor_gpu = actor_gpu
            self.optimizer.set_actor_gpu_to_optimizer()
        

### FROM OFFPOLICYTRAINER LEVEL

    def _checkpoint(self) -> ModelCheckpoint:
        """
        Writes a checkpoint model to memory
        Overrides the default to save the replay buffer.
        """
        ckpt = super()._checkpoint()
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()
        return ckpt

    def save_model(self) -> None:
        """
        Saves the final training model to memory
        Overrides the default to save the replay buffer.
        """
        super().save_model()
        logger.info("Finished calling super().save_model()")
        # if self.checkpoint_replay_buffer:
        #     self.save_replay_buffer()
    
    # COMMENTED OUT WHILE USING SUPERTRACK_BUFFER 

    # def save_replay_buffer(self) -> None:
    #     """
    #     Save the training buffer's update buffer to a pickle file.
    #     """
    #     filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
    #     logger.info(f"Saving Experience Replay Buffer to {filename}...")
    #     with open(filename, "wb") as file_object:
    #         self.update_buffer.save_to_file(file_object)
    #         logger.info(
    #             f"Saved Experience Replay Buffer ({os.path.getsize(filename)} bytes)."
    #         )

    # def maybe_load_replay_buffer(self):
    #     # Load the replay buffer if load
    #     if self.load and self.checkpoint_replay_buffer:
    #         logger.info("Trying to load")
    #         try:
    #             self.load_replay_buffer()
    #         except (AttributeError, FileNotFoundError):
    #             logger.warning(
    #                 "Replay buffer was unable to load, starting from scratch."
    #             )
    #         logger.debug(
    #             "Loaded update buffer with {} sequences".format(
    #                 self.update_buffer.num_experiences
    #             )
    #         )

    # def load_replay_buffer(self) -> None:
    #     """
    #     Loads the last saved replay buffer from a file.
    #     """
    #     filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
    #     logger.info(f"Loading Experience Replay Buffer from {filename}...")
    #     with open(filename, "rb+") as file_object:
    #         self.update_buffer.load_from_file(file_object)
    #     logger.debug(
    #         "Experience replay buffer has {} experiences.".format(
    #             self.update_buffer.num_experiences
    #         )
    #     )
    
    def _has_enough_data_to_train(self) -> bool:
        """
        Returns whether or not there is enough data in the buffer to train
        :return: A boolean corresponding to whether or not there is enough data to train
        """
        max_data_required = max(self.effective_wm_window * self.wm_batch_size, self.effective_policy_window * self.policy_batch_size)
        return (
            self.update_buffer.num_experiences - max(self.effective_wm_window, self.effective_policy_window)
              >= max_data_required
        )

    def _is_ready_update(self) -> bool:
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to whether or not _update_policy() can be run
        """
        return (
            self._has_enough_data_to_train()
            and self._step >= self.hyperparameters.buffer_init_steps
        )

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

        # Needed to resume loads properly
        self._step = policy.get_current_step()
        # Assume steps were updated at the correct ratio before
        self.update_steps = int(max(1, self._step / self.steps_per_update))


    def start_profiler(self):
        if not self.first_update or not self.torch_settings.profile:
            return
        # def trace_handler(prof: profile):
        #     prof.export_chrome_trace("./results/ignore/st_trace.pt.trace.json")
        #     torch.profiler.tensorboard_trace_handler('./results/ignore/log/supertracktrace')(prof)
        #     # torch.profiler.tensorboard_trace_handler("E:/Unity Projects/SuperTrack-Unity/results/ignore/st_data")
        #     print(f"Update step: {self.update_steps}:\n", prof.key_averages().table(row_limit=-1))
        self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                       schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1), 
                       profile_memory=True, 
                       on_trace_ready= torch.profiler.tensorboard_trace_handler('./results/ignore/trace'), 
                       with_stack=True,
                       record_shapes=True)
    
    def step_profiler(self):
        if not self.first_update or not self.torch_settings.profile:
            return
        try:
            self.prof.step()
        except Exception as e:
            print(e)
            print("Profiler failed to step")

    def stop_profiler(self):
        if not self.first_update or not self.torch_settings.profile:
            return
        try:
            self.prof.stop()
        except Exception as e:
            print(e)
            print("Profiler failed to stop")

    @timed
    def _update_policy(self) -> bool:
        """
        Uses update_buffer to update the policy. We sample the update_buffer and update
        until the steps_per_update ratio is met.
        """
        has_updated = False
        batch_update_stats: Dict[str, list] = defaultdict(list)
        self.start_profiler()
        while (
            self._step - self.hyperparameters.buffer_init_steps
        ) / self.update_steps > self.steps_per_update:
            buffer = self.update_buffer
            if self._has_enough_data_to_train():
                world_model_minibatch = buffer.supertrack_sample_mini_batch(self.wm_batch_size,self.wm_window)
                policy_minibatch = buffer.supertrack_sample_mini_batch(self.policy_batch_size, self.policy_window)

                update_stats = self.optimizer.update_world_model(world_model_minibatch, self.wm_batch_size, self.wm_window)
                update_stats.update(self.optimizer.update_policy(policy_minibatch, self.policy_batch_size, self.policy_window))
                for stat_name, value in update_stats.items():
                    batch_update_stats[stat_name].append(value)

                for stat, stat_list in batch_update_stats.items():
                    self._stats_reporter.add_stat(stat, np.mean(stat_list))
                self.update_steps += 1
                has_updated = True
            else:
                raise Exception(f"Update policy called with insufficient data in buffer. Buffer has {self.update_buffer.num_experiences} experiences, but needs {max(self.effective_wm_window * self.wm_batch_size, self.effective_policy_window * self.policy_batch_size)} to update")
            self.step_profiler()
        self.stop_profiler()
        if has_updated:
            self._stats_reporter.set_stat("Num Training Updates", self.update_steps)
            self.first_update = False
        # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
        # a large buffer at each update.
        with hierarchical_timer("update_buffer.truncate"):
            if self.update_buffer.num_experiences > self.hyperparameters.buffer_size * 1.5:
                self.update_buffer.truncate_on_traj_end(
                    int(self.hyperparameters.buffer_size * BUFFER_TRUNCATE_PERCENT)
                )
        return has_updated

### FROM SAC TRAINER LEVEL
    @timed
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the replay buffer.
        """
        super()._process_trajectory(trajectory)
        agent_buffer_trajectory = trajectory.to_supertrack_agentbuffer()
        if agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA][0] is None:
            # self.stats_reporter.add_stat(f"Supertrack Data in {default_device()}", len(agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA]), StatsAggregationMethod.SUM)
            SupertrackUtils.add_supertrack_data_field_OLD(agent_buffer_trajectory, device=default_device())
        else:
            # Bring CPU tensors to GPU 
            for st_datum in agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA]:
                # print("Moving supertrack data on trajectory to GPU")
                st_datum.to(default_device())
        self._append_to_update_buffer(agent_buffer_trajectory)
        # supertrack_trajectory = trajectory.to_supertrack_trajectory()
        # self._append_to_supertrack_buffer(supertrack_trajectory)

    def create_optimizer(self) -> TorchOptimizer:
        return TorchSuperTrackOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore


    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and SAC hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        actor_cls = SuperTrackPolicyNetwork
        actor_kwargs = {"conditional_sigma": False, "tanh_squash": True}

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.policy_network_settings,
            actor_cls,
            actor_kwargs,
            split_on_cpugpu=self.multiprocess,
        )
        # self.maybe_load_replay_buffer()
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
