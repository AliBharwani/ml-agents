# ## ML-Agent Learning (SAC)
# Contains an implementation of SAC as described in https://arxiv.org/abs/1801.01290
# and implemented in https://github.com/hill-a/stable-baselines

from bdb import effective
from collections import defaultdict
import copy
from datetime import datetime
import threading
import time
from typing import Dict, Tuple, cast
import os

from sympy import E

import torch.multiprocessing as mp

from mlagents import simple_queue_with_size, torch_utils
from mlagents.trainers.agent_processor import AgentManagerQueue

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.supertrack import mp_queue
from mlagents.trainers.supertrack.supertrack_utils import SupertrackUtils
from mlagents.trainers.torch_entities.utils import ModelUtils
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
from mlagents.trainers.trainer.rl_trainer import ProfilerState, RLTrainer
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

    # @timed
    def _initialize(self, torch_settings: TorchSettings) -> None:
        self.optimizer._init_world_model()
        
        self.model_saver.register(self.policy)
        self.model_saver.register(self.optimizer)
        self.model_saver.initialize_or_load()
        self._step = self.policy.get_current_step()
        # MY TODO: MAKE SURE SUPER TRACK WORKS W CPU TRAINING
        if not default_device().type == "cuda":
            print(f"WARNING: SUPERTRACK IS NOT TRAINING ON GPU! DEVICE: {default_device()}")
        if self.multiprocess and default_device().type == "cuda":
            logger.info("intializing GPU instance of actor")
            actor_gpu = copy.deepcopy(self.policy.actor)
            actor_gpu.to("cuda")
            # actor_gpu.to("cpu")
            actor_gpu.train()
            self.optimizer.actor_gpu = actor_gpu
            self.optimizer.set_actor_gpu_to_optimizer()
        if self.trainer_settings.multiprocess_trainer:
            # This trainer process will just be a consumer of gpu trajectories
            # CREATE PRODUCER PROCESS
            self.gpu_batch_queue = simple_queue_with_size.SimpleQueueWithSize()
            self.dummy_q = torch.multiprocessing.SimpleQueue()
            # self.num_batches_in_q = mp.Value("i", 0)
            self.num_steps_processed = mp.Value("i", 0)
            batch_size_data = (self.wm_window, self.wm_batch_size, self.policy_window, self.policy_batch_size)
            gpu_batch_producer = mp.Process(
                target=SuperTrackTrainer.gpu_batch_producer_process,
                  name="gpu_batch_producer", 
                  args=(torch_settings, 
                        self.trajectory_queues[0], 
                        self.gpu_batch_queue,
                        # self.num_batches_in_q,
                        self._stats_reporter, 
                        self.num_steps_processed, 
                        self.hyperparameters.buffer_size,
                        batch_size_data, self.dummy_q), 
                  daemon=True)
            gpu_batch_producer.start()


    @staticmethod
    def gpu_batch_producer_process(torch_settings, traj_queue, gpu_batch_queue, stats_reporter, num_steps_processed, buffer_size, batch_size_data, dummy_q):
        logging_util.set_log_level(logging_util.INFO)
        torch_utils.set_torch_config(torch_settings)
        logger.info(f"gpu_batch_producer_process started on pid {os.getpid()} parent pid {os.getppid()}")
        gpu_buffer: AgentBuffer = AgentBuffer()
        wm_window, wm_batch_size, policy_window, policy_batch_size = batch_size_data   
        effective_wm_window = wm_window + 1
        effective_policy_window = policy_window + 1
        max_data_required = max(effective_wm_window * wm_batch_size, effective_policy_window * policy_batch_size)
        MAX_NUM_BATCHES_IN_Q = 4
        dummy_buffer = []
        try:
            while True:
                # read from traj_queue
                _queried = False
                num_read = 0
                num_steps_processed_this_iteration = 0
                processed_large_number_of_trajectories = traj_queue.qsize() > 150
                if (processed_large_number_of_trajectories):
                    print(f"{datetime.now().strftime('%I:%M:%S ')} Large number of trajectories in queue: {traj_queue.qsize()}")
                for _ in range(traj_queue.qsize()):
                    _queried = True
                    try:
                        traj = traj_queue.get_nowait()
                        # self._process_trajectory(t)
                        agent_buffer_trajectory = traj.to_supertrack_agentbuffer()
                        for st_datum in agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA]:
                            # print("Moving supertrack data on trajectory to GPU")
                            st_datum.to(default_device())
                        # Allocate a CUDA tensor to see if it fucks up memory
                        dummy_tensor = torch.ones( 30,  device=default_device())
                        dummy_buffer.append(dummy_tensor)
                        num_steps_processed_this_iteration += len(traj.steps)
                        agent_buffer_trajectory.resequence_and_append(
                           gpu_buffer, training_length=1
                        )
                        num_read += 1
                    except AgentManagerQueue.Empty:
                        break
                if not _queried:
                    # Yield thread to avoid busy-waiting
                    time.sleep(0.001)
                if (processed_large_number_of_trajectories):
                    print(f"{datetime.now().strftime('%I:%M:%S ')} Finished processing trajectories in queue, num_read: {num_read}")
                if num_read > 0:
                    stats_reporter.add_stat('Avg # Traj Read', num_read, StatsAggregationMethod.AVERAGE)

                # create batch and add to gpu_batch_queue
                if (gpu_buffer.num_experiences - max(effective_wm_window, effective_policy_window) >= max_data_required) and gpu_batch_queue.qsize() < MAX_NUM_BATCHES_IN_Q:
                    wm_minibatch = gpu_buffer.supertrack_sample_mini_batch(wm_batch_size, wm_window)
                    policy_minibatch = gpu_buffer.supertrack_sample_mini_batch(policy_batch_size, policy_window)
                    num_issue = 0
                    num_zero = 0
                    num_nan = 0
                    st_data = [wm_minibatch[BufferKey.SUPERTRACK_DATA][i] for i in range(wm_minibatch.num_experiences)]
                    for st_datum in st_data:
                        tensor = st_datum.sim_char_state.positions
                        num_zero += 1 if torch.count_nonzero(tensor).item() == 0 else 0
                        num_nan += 1 if torch.isnan(tensor).any() else 0
                        num_issue += 1 if ModelUtils.check_values_near_zero_or_nan(st_datum.sim_char_state.positions) else 0
                    # print(f"=========== PRODUCER THREAD: World model batch has {num_issue} issues out of {len(st_data)} possible") 
                    # print(f"=========== PRODUCER THREAD: World model batch has {num_zero} zero and {num_nan} nan out of {len(st_data)} possible") 
                    gpu_batch_queue.put((wm_minibatch, policy_minibatch))
                    dummy_batch = torch.stack(dummy_buffer[:1024]) # shape: [1024, 1, 30]
                    print(f"Producer dummy batch [:10] : {dummy_batch[:10]}")
                    # print(f"Dummy batch dtype: {dummy_batch.dtype}, device: {dummy_batch.device}")
                    idxes_of_zeroes = torch.argwhere(torch.where(dummy_batch.flatten(0, -1) == 0, 1, 0.)) 
                    # print(f"idxes_of_zeroes dtype: {idxes_of_zeroes.dtype}, device: {idxes_of_zeroes.device}")
                    print(f"=========== PRODUCER Idxes of zeroes [0]: {idxes_of_zeroes[0]} at [1]: {idxes_of_zeroes[-1]} median: {torch.median(idxes_of_zeroes)} mean: {torch.mean(idxes_of_zeroes.to(torch.float32))}")
                    print(f"=========== PRODUCER THREAD: World model batch has { torch.sum(dummy_batch == 0).item()} zero out of {dummy_batch.numel()} possible")
                    dummy_q.put(dummy_batch)
                    
                # Do this after setting up batch queue 
                num_steps_processed.value += num_steps_processed_this_iteration

                 # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
                # a large buffer at each update.
                if gpu_buffer.num_experiences > buffer_size:
                    with hierarchical_timer("update_buffer.truncate"):
                        gpu_buffer.truncate_on_traj_end(
                            int(buffer_size * BUFFER_TRUNCATE_PERCENT)
                        )
                        logger.info(f"Truncated update buffer to {gpu_buffer.num_experiences} experiences")

        except KeyboardInterrupt:
            logger.info("gpu_batch_producer_process received KeyboardInterrupt")
        # finally:
            # prof.stop()
            # write_timing_tree(trainer.run_log_path)

    def advance_consumer(self):
        """
        Advances the consumer/training part of this trainer
        return: whether or not the trainer read a batch 
        """
        # this would normally happen in super._process_trajectory() but we do it manually here: 
        num_new_steps_processed = self.num_steps_processed.value
        if num_new_steps_processed > 0:
            self._maybe_write_summary(self.get_step + num_new_steps_processed)
            self._maybe_save_model(self.get_step + num_new_steps_processed)
            self._increment_step(num_new_steps_processed, self.policy_queues[0].behavior_id)
            self.num_steps_processed.value = 0
        _update_occured = False
        if self.should_still_train:
            if self._is_ready_update():
                with hierarchical_timer("_update_policy"):
                    # print(f"{datetime.now().strftime('%I:%M:%S ')} Entering trainer update policy")
                    batches = None
                    try:
                        batches = self.gpu_batch_queue.get()
                        dummy_tensor = self.dummy_q.get()
                        # print(f"======= TRAINER THREAD GOT DUMMY BATCH WITH NUM ZERO: {torch.sum(dummy_tensor == 0).item()} out of {dummy_tensor.numel()} possible")
                        idxes_of_zeroes = torch.argwhere(torch.where(dummy_tensor.flatten(0, -1) == 0, 1, 0.))
                        # print(f"======= TRAINER THREAD Idxes of zeroes [0]: {idxes_of_zeroes[0]} at [1]: {idxes_of_zeroes[-1]} median: {torch.median(idxes_of_zeroes)} mean: {torch.mean(idxes_of_zeroes.to(torch.float32))}")
                        # print(f"======= TRAINER THREAD GOT DUMMY BATCH WITH NUM ZERO: {torch.sum(dummy_tensor == 0).item()} out of {dummy_tensor.numel()} possible")
                        print(f"TRAINER dummy batch [:10] : {dummy_tensor[:10]}")

                        # print(dummy_tensor)
                        # Check if dummy_tensor is all ones
                        # if not torch.all(torch.eq(dummy_tensor, torch.ones_like(dummy_tensor))):
                            # print("ERROR: GPU batch producer process did not allocate a new tensor")
                            # print(dummy_tensor)
                            # raise Exception("GPU batch producer process did not allocate a new tensor")
                        # self.num_batches_in_q.value -= 1
                    except mp_queue.Empty:
                        pass
                    if batches is not None and self._update_policy(batches=batches, max_update_iterations=1): # only update once since provided one batch
                        del batches
                        _update_occured = True
                        # if self.profiler_state == ProfilerState.RUNNING: torch.cuda.nvtx.range_push("put in policy queue")
                        for q in self.policy_queues:
                            # Get policies that correspond to the policy queue in question
                            q.put(self.get_policy(q.behavior_id))
                        # if self.profiler_state == ProfilerState.RUNNING: torch.cuda.nvtx.range_pop()
                    # print(f"{datetime.now().strftime('%I:%M:%S ')} Exiting trainer update policy")
        return _update_occured

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
        if self.checkpoint_replay_buffer:
            self.save_replay_buffer()
    
    # COMMENTED OUT WHILE USING SUPERTRACK_BUFFER 
    def save_replay_buffer(self) -> None:
        """
        Save the training buffer's update buffer to a pickle file.
        """
        filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
        logger.info(f"Saving Experience Replay Buffer to {filename}...")
        with open(filename, "wb") as file_object:
            self.update_buffer.save_to_file(file_object)
            logger.info(
                f"Saved Experience Replay Buffer ({os.path.getsize(filename)} bytes)."
            )

    def maybe_load_replay_buffer(self):
        # Load the replay buffer if load
        if self.load and self.checkpoint_replay_buffer:
            logger.info("Trying to load")
            try:
                self.load_replay_buffer()
            except (AttributeError, FileNotFoundError):
                logger.warning(
                    "Replay buffer was unable to load, starting from scratch."
                )
            logger.debug(
                "Loaded update buffer with {} sequences".format(
                    self.update_buffer.num_experiences
                )
            )

    def load_replay_buffer(self) -> None:
        """
        Loads the last saved replay buffer from a file.
        """
        filename = os.path.join(self.artifact_path, "last_replay_buffer.hdf5")
        logger.info(f"Loading Experience Replay Buffer from {filename}...")
        with open(filename, "rb+") as file_object:
            self.update_buffer.load_from_file(file_object)
        logger.debug(
            "Experience replay buffer has {} experiences.".format(
                self.update_buffer.num_experiences
            )
        )
    
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
        if self.trainer_settings.multiprocess_trainer:
            return self._step >= self.hyperparameters.buffer_init_steps and not self.gpu_batch_queue.empty()
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

    @timed
    def _update_policy(self, batches: Tuple[torch.Tensor, torch.Tensor] = None, max_update_iterations : int = 100) -> bool:
        """
        Uses update_buffer to update the policy. We sample the update_buffer and update
        until the steps_per_update ratio is met.
        """
        has_updated = False
        nsys_profiler_running = self.profiler_state == ProfilerState.RUNNING
        batch_update_stats: Dict[str, list] = defaultdict(list)
        update_steps_before = self.update_steps
        num_steps_to_update = min(max_update_iterations, int(((self._step - self.hyperparameters.buffer_init_steps) / self.steps_per_update)) - self.update_steps + 1)
        print(f"{datetime.now().strftime('%I:%M:%S ')} Will update {num_steps_to_update} times - self._step: {self._step}")
        if batches:
            print(f"Batches num_experiences: {batches[0].num_experiences}, {batches[1].num_experiences}")
        while  (self.update_steps - update_steps_before) < max_update_iterations and (
            self._step - self.hyperparameters.buffer_init_steps
        ) / self.update_steps > self.steps_per_update:
            buffer = self.update_buffer
            if nsys_profiler_running: torch.cuda.nvtx.range_push(f"iteration {self.update_steps - update_steps_before}")

            # if self._has_enough_data_to_train():
            # print(f"{datetime.now().strftime('%I:%M:%S ')} Entering supertrack_sample_mini_batch")
            if nsys_profiler_running: torch.cuda.nvtx.range_push("supertrack_sample_mini_batch")
            if not batches:
                world_model_minibatch = buffer.supertrack_sample_mini_batch(self.wm_batch_size, self.wm_window)
                policy_minibatch = buffer.supertrack_sample_mini_batch(self.policy_batch_size, self.policy_window)
            else:
                world_model_minibatch, policy_minibatch = batches
            if nsys_profiler_running: torch.cuda.nvtx.range_pop()

            # print(f"{datetime.now().strftime('%I:%M:%S ')} Entering update_world_model")
            if nsys_profiler_running: torch.cuda.nvtx.range_push("update_world_model")
            update_stats = self.optimizer.update_world_model(world_model_minibatch, self.wm_batch_size, self.wm_window)
            if nsys_profiler_running: torch.cuda.nvtx.range_pop()

            # print(f"{datetime.now().strftime('%I:%M:%S ')} Entering optimizer update_policy")
            if nsys_profiler_running: torch.cuda.nvtx.range_push("update_policy")
            update_stats.update(self.optimizer.update_policy(policy_minibatch, self.policy_batch_size, self.policy_window, nsys_profiler_running=True))
            if nsys_profiler_running: torch.cuda.nvtx.range_pop()

            for stat_name, value in update_stats.items():
                batch_update_stats[stat_name].append(value)

            for stat, stat_list in batch_update_stats.items():
                self._stats_reporter.add_stat(stat, np.mean(stat_list))
            self.update_steps += 1
            has_updated = True
            # else:
            #     raise Exception(f"Update policy called with insufficient data in buffer. Buffer has {self.update_buffer.num_experiences} experiences, but needs {max(self.effective_wm_window * self.wm_batch_size, self.effective_policy_window * self.policy_batch_size)} to update")
            if nsys_profiler_running: torch.cuda.nvtx.range_pop()

        if has_updated:
            print("Finished with updates")
            num_updates = self.update_steps - update_steps_before
            self._stats_reporter.add_stat("Avg # Updates", num_updates, StatsAggregationMethod.AVERAGE)
            self._stats_reporter.set_stat("Num Training Updates", self.update_steps)
            self.first_update = False
        # Truncate update buffer if neccessary. Truncate more than we need to to avoid truncating
        # a large buffer at each update.
        if self.update_buffer.num_experiences > self.hyperparameters.buffer_size:
            with hierarchical_timer("update_buffer.truncate"):
                self.update_buffer.truncate_on_traj_end(
                    int(self.hyperparameters.buffer_size * BUFFER_TRUNCATE_PERCENT)
                )
                logger.info(f"Truncated update buffer to {self.update_buffer.num_experiences} experiences")
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
        # else:
            # Bring CPU tensors to GPU 
        for st_datum in agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA]:
            # print("Moving supertrack data on trajectory to GPU")
            st_datum.to(default_device())
        self._append_to_update_buffer(agent_buffer_trajectory)

    def create_optimizer(self) -> TorchOptimizer:
        return TorchSuperTrackOptimizer(  # type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore


    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Creates a policy with a PyTorch backend and Supertrack hyperparameters
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
