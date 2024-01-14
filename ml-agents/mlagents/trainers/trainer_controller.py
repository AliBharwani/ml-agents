# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import threading
import time
import trace
import traceback
from typing import Dict, Set, List
from collections import defaultdict
from mlagents.trainers.trainer.rl_trainer import ProfilerState
from mlagents_envs.base_env import BehaviorSpec
from sympy import Q
import torch
import torch.multiprocessing as mp

import numpy as np
from mlagents.torch_utils.torch import default_device
from mlagents.trainers import stats
from mlagents.trainers.settings import TorchSettings
from mlagents.trainers.stats import StatsReporter, StatsReporterCommand, StatsReporterMP, StatsSummary, StatsWriter
from mlagents_envs import logging_util

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.env_manager import EnvManager, EnvironmentStep
from mlagents_envs.exception import (
    UnityEnvironmentException,
    UnityCommunicationException,
    UnityCommunicatorStoppedException,
)
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod
from mlagents_envs.timers import (
    hierarchical_timer,
    timed,
    get_timer_stack_for_thread,
    merge_gauges,
    write_timing_tree,
)
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.agent_processor import AgentManager, AgentManagerQueue
from mlagents import torch_utils
from mlagents.torch_utils.globals import get_rank
from torch.profiler import profile, record_function, ProfilerActivity



class TrainerController:
    def __init__(
        self,
        trainer_factory: TrainerFactory,
        output_path: str,
        run_id: str,
        param_manager: EnvironmentParameterManager,
        train: bool,
        training_seed: int,
        torch_settings: TorchSettings,
    ):
        """
        :param output_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param param_manager: EnvironmentParameterManager object which stores information about all
        environment parameters.
        :param train: Whether to train model, or only run inference.
        :param training_seed: Seed to use for Numpy and Torch random number generation.
        :param threaded: Whether or not to run trainers in a separate thread. Disable for testing/debugging.
        """
        self.trainers: Dict[str, Trainer] = {}
        self.brain_name_to_identifier: Dict[str, Set] = defaultdict(set)
        self.trainer_factory = trainer_factory
        self.output_path = output_path
        self.logger = get_logger(__name__)
        self.run_id = run_id
        self.train_model = train
        self.param_manager = param_manager
        self.ghost_controller = self.trainer_factory.ghost_controller
        self.registered_behavior_ids: Set[str] = set()

        self.trainer_threads: List[threading.Thread] = []
        self.trainer_processes : List[mp.Process] = []
        self.kill_trainers = False
        np.random.seed(training_seed)
        torch_utils.torch.manual_seed(training_seed)
        self.rank = get_rank()
        self.first_update = True
        self.multiprocess = False
        self.stats_queue = None
        self.torch_settings = torch_settings

    @timed
    def _save_models(self):
        """
        Saves current model to checkpoint folder.
        """
        if self.rank is not None and self.rank != 0:
            return

        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.debug("Saved Model")

    @staticmethod
    def _create_output_path(output_path):
        try:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except Exception:
            raise UnityEnvironmentException(
                f"The folder {output_path} containing the "
                "generated model could not be "
                "accessed. Please make sure the "
                "permissions are set correctly."
            )

    @timed
    def _reset_env(self, env_manager: EnvManager) -> None:
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        new_config = self.param_manager.get_current_samplers()
        env_manager.reset(config=new_config)
        # Register any new behavior ids that were generated on the reset.
        self._register_new_behaviors(env_manager, env_manager.first_step_infos)

    def _not_done_training(self) -> bool:
        return (
            any(t.should_still_train for t in self.trainers.values())
            or not self.train_model
        ) or len(self.trainers) == 0

    def _create_trainer_and_manager(
        self, env_manager: EnvManager, name_behavior_id: str
    ) -> None:

        parsed_behavior_id = BehaviorIdentifiers.from_name_behavior_id(name_behavior_id)
        behavior_spec = env_manager.training_behaviors[name_behavior_id]
        brain_name = parsed_behavior_id.brain_name
        trainerthread = None
        trainer_process = None
        if brain_name in self.trainers:
            trainer = self.trainers[brain_name]
        else:
            trainer_config = self.trainer_factory.trainer_config[brain_name]
            if trainer_config.threaded:
                trainer = self.trainer_factory.generate(brain_name, self.torch_settings)
                # Only create trainer thread for new trainers
                trainerthread = threading.Thread(
                    target=self.trainer_update_func, args=(trainer,), daemon=True
                )
                self.trainer_threads.append(trainerthread)
            elif trainer_config.use_pytorch_mp:
                self.multiprocess = True
                stats_queue = mp.Queue(maxsize=0)
                self.stats_queue = stats_queue
                trainer = self.trainer_factory.generate(brain_name, StatsReporterMP(brain_name, stats_queue))
                # trainer_process = mp.Process(target=TrainerController.dummy_func, 
                #                             daemon=True, 
                #                             name=f"trainer_process")
                trainer_process = mp.Process(target=TrainerController.trainer_process_update_func,
                                            args=(trainer, self.torch_settings, behavior_spec, self.logger.getEffectiveLevel()), 
                                            daemon=True, 
                                            name=f"trainer_process")
                stats_reporter_process = mp.Process(target=stats.stats_processor, args=(brain_name, stats_queue, StatsReporter.writers,), daemon=True, name=f"stats_reporter_process")
                self.trainer_processes += [trainer_process, stats_reporter_process]
            else:
                print(f"Running trainer on same thread & process as env manager")
                trainer = self.trainer_factory.generate(brain_name)
                
            env_manager.on_training_started(
                brain_name, self.trainer_factory.trainer_config[brain_name]
            )
            self.trainers[brain_name] = trainer

        # if trainer.trainer_settings.use_pytorch_mp:
            # Make sure policy actor is not initialized using CUDA
            # torch_utils.set_torch_config(TorchSettings(device="cpu"))
        policy = trainer.create_policy(
            parsed_behavior_id,
            behavior_spec,
        )
        # if trainer.trainer_settings.use_pytorch_mp:
            # Reset
            # torch_utils.set_torch_config(self.torch_settings)
        trainer.torch_settings = self.torch_settings
        # if not trainer.multiprocess:
        trainer.add_policy(parsed_behavior_id, policy)

        agent_manager = AgentManager(
            policy,
            name_behavior_id,
            trainer.stats_reporter,
            self.torch_settings,
            trainer.parameters.time_horizon,
            threaded=trainer.threaded,
            process_trajectory_on_termination=trainer.parameters.process_trajectory_on_termination,
            use_pytorch_mp=trainer.parameters.use_pytorch_mp,
        )
        env_manager.set_agent_manager(name_behavior_id, agent_manager)
        env_manager.set_policy(name_behavior_id, policy)
        self.brain_name_to_identifier[brain_name].add(name_behavior_id)

        trainer.publish_policy_queue(agent_manager.policy_queue)
        trainer.subscribe_trajectory_queue(agent_manager.trajectory_queue)
        self.DEBUG_ONLY_trajectory_queue = agent_manager.trajectory_queue
        # Only start new trainers
        if trainerthread is not None or trainer.multiprocess:
            if trainerthread is not None:
                trainerthread.start()
            if trainer.multiprocess: 
                trainer_process.start()
                stats_reporter_process.start()
        elif trainer.get_trainer_name() == "supertrack": 
            try:
                trainer._initialize(self.torch_settings)
            except Exception as e:
                print(f"Failed to initialize trainer", e.with_traceback(e.__traceback__))
            

    def _create_trainers_and_managers(
        self, env_manager: EnvManager, behavior_ids: Set[str]
    ) -> None:
        for behavior_id in behavior_ids:
            self._create_trainer_and_manager(env_manager, behavior_id)

    @timed
    def start_learning(self, env_manager: EnvManager) -> None:
        self._create_output_path(self.output_path)
        try:
            # Initial reset
            self._reset_env(env_manager)
            self.param_manager.log_current_lesson()
            while self._not_done_training():
                n_steps = self.advance(env_manager)
                for _ in range(n_steps):
                    self.reset_env_if_ready(env_manager)
            # Stop advancing trainers
            self.join_threads()
        except (
            KeyboardInterrupt,
            UnityCommunicationException,
            UnityEnvironmentException,
            UnityCommunicatorStoppedException,
            Exception,
        ) as ex:
            self.join_threads()
            self.logger.info(
                "Learning was interrupted. Please wait while the graph is generated."
            )
            if isinstance(ex, KeyboardInterrupt) or isinstance(
                ex, UnityCommunicatorStoppedException
            ):
                pass
            else:
                # If the environment failed, we want to make sure to raise
                # the exception so we exit the process with an return code of 1.
                raise ex
        finally:
            if self.train_model and not self.multiprocess:
                self._save_models()
            self.logger.info("Learning was stopped. Main process exiting.")


    def end_trainer_episodes(self) -> None:
        # Reward buffers reset takes place only for curriculum learning
        # else no reset.
        for trainer in self.trainers.values():
            trainer.end_episode()

    def reset_env_if_ready(self, env: EnvManager) -> None:
        # Get the sizes of the reward buffers.
        reward_buff = {k: list(t.reward_buffer) for (k, t) in self.trainers.items()}
        curr_step = {k: int(t.get_step) for (k, t) in self.trainers.items()}
        max_step = {k: int(t.get_max_steps) for (k, t) in self.trainers.items()}
        # Attempt to increment the lessons of the brains who
        # were ready.
        updated, param_must_reset = self.param_manager.update_lessons(
            curr_step, max_step, reward_buff
        )
        if updated:
            for trainer in self.trainers.values():
                trainer.reward_buffer.clear()
        # If ghost trainer swapped teams
        ghost_controller_reset = self.ghost_controller.should_reset()
        if param_must_reset or ghost_controller_reset:
            self._reset_env(env)  # This reset also sends the new config to env
            self.end_trainer_episodes()
        elif updated:
            env.set_env_parameters(self.param_manager.get_current_samplers())

    @timed
    def advance(self, env_manager: EnvManager) -> int:
        # Get steps
        with hierarchical_timer("env_step"):
            new_step_infos = env_manager.get_steps()
            self._register_new_behaviors(env_manager, new_step_infos)

            num_steps = env_manager.process_steps(new_step_infos)
        # Report current lesson for each environment parameter
        for (
            param_name,
            lesson_number,
        ) in self.param_manager.get_current_lesson_number().items():
            for trainer in self.trainers.values():
                trainer.stats_reporter.set_stat(
                    f"Environment/Lesson Number/{param_name}", lesson_number
                )

        for trainer in self.trainers.values():
            if not (trainer.threaded or trainer.multiprocess):
                # For nsys profiling only, we want to profile how it performs reading a large number of trajectories
                # So advance if we're not in profile mode 
                # If we are in profile mode, only advance if we less than the number of warmup steps (5) or we have enough trajectories to profile
                if not self.torch_settings.profile or (trainer.update_steps < 5 or self.DEBUG_ONLY_trajectory_queue.qsize() > 100):
                    if self.torch_settings.profile and trainer.update_steps >= 5:
                        env_manager.close() # we don't care to profile the env manager
                    with hierarchical_timer("trainer_advance"):
                        trainer.advance(profiling_enabled = self.torch_settings.profile)
                if self.torch_settings.profile and trainer.profiler_state == ProfilerState.RUNNING:
                    print(f"Stopping cudart on update_step: {trainer.update_steps}")
                    trainer.profiler_state = ProfilerState.STOPPED
                    torch.cuda.cudart().cudaProfilerStop()

        self.first_update = False
        return num_steps

    def _register_new_behaviors(
        self, env_manager: EnvManager, step_infos: List[EnvironmentStep]
    ) -> None:
        """
        Handle registration (adding trainers and managers) of new behaviors ids.
        :param env_manager:
        :param step_infos:
        :return:
        """
        step_behavior_ids: Set[str] = set()
        for s in step_infos:
            step_behavior_ids |= set(s.name_behavior_ids)
        new_behavior_ids = step_behavior_ids - self.registered_behavior_ids
        self._create_trainers_and_managers(env_manager, new_behavior_ids)
        self.registered_behavior_ids |= step_behavior_ids

    def join_threads(self, timeout_seconds: float = 1.0) -> None:
        """
        Wait for threads to finish, and merge their timer information into the main thread.
        :param timeout_seconds:
        :return:
        """
        self.kill_trainers = True
        # for trainer in self.trainers.values():
        #     for traj_q in trainer.trajectory_queues:
        #         traj_q.close()
        for t in [*self.trainer_threads, *self.trainer_processes]:
            try:
                t.join(timeout_seconds)
                self.logger.info(f"Trainer thread/process {t} joined")
                t.close()
            except Exception as e:
                self.logger.error(f"Trainer thread {t} threw an exception after {timeout_seconds} seconds")
                self.logger.exception(e)
            # finally:
                # try:
                # t.terminate()
                # except ValueError as e:
                #     self.logger.debug(f"Trainer thread {t} could not be terminated, likely already closed: {e}")
                # except Exception as e:
                #     self.logger.exception(f"Trainer thread {t} could not be terminated: {e}")
        # self.logger.debug("Closing trainer processes")
        # for t in self.trainer_processes:
        #     t.close()
        # self.logger.debug("Closing of trainer processes complete.")
        if self.stats_queue is not None:
            self.logger.debug("Closing stats queue")
            self.stats_queue.close()
        with hierarchical_timer("trainer_threads") as main_timer_node:
            for trainer_thread in self.trainer_threads:
                thread_timer_stack = get_timer_stack_for_thread(trainer_thread)
                if thread_timer_stack:
                    main_timer_node.merge(
                        thread_timer_stack.root,
                        root_name="thread_root",
                        is_parallel=True,
                    )
                    merge_gauges(thread_timer_stack.gauges)
        

    def trainer_update_func(self, trainer: Trainer) -> None:
        while not self.kill_trainers:
            with hierarchical_timer("trainer_advance"):
                trainer.advance()

    @staticmethod
    def trainer_process_update_func(trainer: Trainer, torch_settings: TorchSettings,  behavior_spec : BehaviorSpec, log_level: int = logging_util.INFO) -> None:
        # Set log level. On some platforms, the logger isn't common with the
        # main process, so we need to set it again.
        logging_util.set_log_level(log_level)
        logger = get_logger(__name__)
        torch_utils.set_torch_config(torch_settings)
        logger.info(f"Trainer process started on pid {os.getpid()} parent pid {os.getppid()}")
        try:
            trainer._initialize(torch_settings)
        except Exception as e:
            print(f"Failed to initialize trainer", e.with_traceback(e.__traceback__))
        try:
            while True:
                    with hierarchical_timer("trainer_advance"):
                        # if trainer.trainer_settings.multiprocess_trainer:
                        #     _queried = trainer.advance_consumer()
                        #     if not _queried:
                        #         # Yield thread to avoid busy-waiting
                        #         time.sleep(.0001)
                        # else:
                        trainer.advance()
        except(KeyboardInterrupt) as ex:
            logger.debug("Trainer process shutting down.")
        except Exception as ex:
            logger.exception(f"An unexpected error occurred in the trainer process.: {ex}")
        finally:
            # prof.stop()
            write_timing_tree(trainer.run_log_path)
            logger.debug("Saving model")
            trainer.save_model()
            logger.info("Trainer process closing.")

    @staticmethod
    def dummy_func() -> None:
        print(f"dummy_func process started on pid {os.getpid()} parent pid {os.getppid()}")
        while True:
            time.sleep(1)
