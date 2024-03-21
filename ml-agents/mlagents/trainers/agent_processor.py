import atexit
import os
# import multiprocessing
import pdb
import sys
import time
import traceback
from mlagents.trainers.settings import TorchSettings
import numpy as np
from typing import List, Dict, TypeVar, Generic, Tuple, Any, Union
from collections import defaultdict, Counter
import queue

from sympy import use
from mlagents import simple_queue_with_size
from mlagents.torch_utils import torch
from mlagents.trainers.supertrack import mp_queue
from mlagents_envs import logging_util

from mlagents_envs.base_env import (
    ActionTuple,
    DecisionSteps,
    DecisionStep,
    TerminalSteps,
    TerminalStep,
)
from mlagents_envs.side_channel.stats_side_channel import (
    StatsAggregationMethod,
    EnvironmentStats,
)
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.trajectory import AgentStatus, Trajectory, AgentExperience
from mlagents.trainers.policy import Policy
from mlagents.trainers.action_info import ActionInfo, ActionInfoOutputs
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.behavior_id_utils import (
    get_global_agent_id,
    get_global_group_id,
    GlobalAgentId,
    GlobalGroupId,
)
from mlagents.trainers.torch_entities.action_log_probs import LogProbsTuple
from mlagents.trainers.torch_entities.utils import ModelUtils

T = TypeVar("T")

logger = logging_util.get_logger(__name__)
class AgentProcessor:
    """
    AgentProcessor contains a dictionary per-agent trajectory buffers. The buffers are indexed by agent_id.
    Buffer also contains an update_buffer that corresponds to the buffer used when updating the model.
    One AgentProcessor should be created per agent group.
    """

    def __init__(
        self,
        policy: Policy,
        behavior_id: str,
        stats_reporter: StatsReporter,
        max_trajectory_length: int = sys.maxsize,
        process_trajectory_on_termination: bool = False,
    ):
        """
        Create an AgentProcessor.

        :param trainer: Trainer instance connected to this AgentProcessor. Trainer is given trajectory
        when it is finished.
        :param policy: Policy instance associated with this AgentProcessor.
        :param max_trajectory_length: Maximum length of a trajectory before it is added to the trainer.
        :param stats_category: The category under which to write the stats. Usually, this comes from the Trainer.
        """
        self._experience_buffers: Dict[
            GlobalAgentId, List[AgentExperience]
        ] = defaultdict(list)
        self._last_step_result: Dict[GlobalAgentId, Tuple[DecisionStep, int]] = {}
        # current_group_obs is used to collect the current (i.e. the most recently seen)
        # obs of all the agents in the same group, and assemble the group obs.
        # It is a dictionary of GlobalGroupId to dictionaries of GlobalAgentId to observation.
        self._current_group_obs: Dict[
            GlobalGroupId, Dict[GlobalAgentId, List[np.ndarray]]
        ] = defaultdict(lambda: defaultdict(list))
        # group_status is used to collect the current, most recently seen
        # group status of all the agents in the same group, and assemble the group's status.
        # It is a dictionary of GlobalGroupId to dictionaries of GlobalAgentId to AgentStatus.
        self._group_status: Dict[
            GlobalGroupId, Dict[GlobalAgentId, AgentStatus]
        ] = defaultdict(lambda: defaultdict(None))
        # last_take_action_outputs stores the action a_t taken before the current observation s_(t+1), while
        # grabbing previous_action from the policy grabs the action PRIOR to that, a_(t-1).
        self._last_take_action_outputs: Dict[GlobalAgentId, ActionInfoOutputs] = {}

        self._episode_steps: Counter = Counter()
        self._episode_rewards: Dict[GlobalAgentId, float] = defaultdict(float)
        self._stats_reporter = stats_reporter
        self._max_trajectory_length = max_trajectory_length
        self._process_trajectory_on_termination = process_trajectory_on_termination
        self._trajectory_queues: List[AgentManagerQueue[Trajectory]] = []
        self._behavior_id = behavior_id

        # Note: In the future this policy reference will be the policy of the env_manager and not the trainer.
        # We can in that case just grab the action from the policy rather than having it passed in.
        self.policy = policy

    def add_experiences(
        self,
        decision_steps: DecisionSteps,
        terminal_steps: TerminalSteps,
        worker_id: int,
        previous_action: ActionInfo,
    ) -> None:
        """
        Adds experiences to each agent's experience history.
        :param decision_steps: current DecisionSteps.
        :param terminal_steps: current TerminalSteps.
        :param previous_action: The outputs of the Policy's get_action method.
        """
        take_action_outputs = previous_action.outputs
        if take_action_outputs:
            try:
                for _entropy in take_action_outputs["entropy"]:
                    if isinstance(_entropy, torch.Tensor):
                        _entropy = ModelUtils.to_numpy(_entropy)
                    self._stats_reporter.add_stat("Policy/Entropy", _entropy)
            except KeyError:
                pass

        # Make unique agent_ids that are global across workers
        action_global_agent_ids = [
            get_global_agent_id(worker_id, ag_id) for ag_id in previous_action.agent_ids
        ]
        for global_id in action_global_agent_ids:
            if global_id in self._last_step_result:  # Don't store if agent just reset
                self._last_take_action_outputs[global_id] = take_action_outputs

        # Iterate over all the terminal steps, first gather all the group obs
        # and then create the AgentExperiences/Trajectories. _add_to_group_status
        # stores Group statuses in a common data structure self.group_status
        for terminal_step in terminal_steps.values():
            self._add_group_status_and_obs(terminal_step, worker_id)
        for terminal_step in terminal_steps.values():
            local_id = terminal_step.agent_id
            global_id = get_global_agent_id(worker_id, local_id)
            self._process_step(
                terminal_step, worker_id, terminal_steps.agent_id_to_index[local_id]
            )

        # Iterate over all the decision steps, first gather all the group obs
        # and then create the trajectories. _add_to_group_status
        # stores Group statuses in a common data structure self.group_status
        for ongoing_step in decision_steps.values():
            self._add_group_status_and_obs(ongoing_step, worker_id)
        for ongoing_step in decision_steps.values():
            local_id = ongoing_step.agent_id
            self._process_step(
                ongoing_step, worker_id, decision_steps.agent_id_to_index[local_id]
            )
        # Clear the last seen group obs when agents die, but only after all of the group
        # statuses were added to the trajectory.
        for terminal_step in terminal_steps.values():
            local_id = terminal_step.agent_id
            global_id = get_global_agent_id(worker_id, local_id)
            self._clear_group_status_and_obs(global_id)

        for _gid in action_global_agent_ids:
            # If the ID doesn't have a last step result, the agent just reset,
            # don't store the action.
            if _gid in self._last_step_result:
                if "action" in take_action_outputs:
                    self.policy.save_previous_action(
                        [_gid], take_action_outputs["action"]
                    )

    def _add_group_status_and_obs(
        self, step: Union[TerminalStep, DecisionStep], worker_id: int
    ) -> None:
        """
        Takes a TerminalStep or DecisionStep and adds the information in it
        to self.group_status. This information can then be retrieved
        when constructing trajectories to get the status of group mates. Also stores the current
        observation into current_group_obs, to be used to get the next group observations
        for bootstrapping.
        :param step: TerminalStep or DecisionStep
        :param worker_id: Worker ID of this particular environment. Used to generate a
            global group id.
        """
        global_agent_id = get_global_agent_id(worker_id, step.agent_id)
        stored_decision_step, idx = self._last_step_result.get(
            global_agent_id, (None, None)
        )
        stored_take_action_outputs = self._last_take_action_outputs.get(
            global_agent_id, None
        )
        if stored_decision_step is not None and stored_take_action_outputs is not None:
            # 0, the default group_id, means that the agent doesn't belong to an agent group.
            # If 0, don't add any groupmate information.
            if step.group_id > 0:
                global_group_id = get_global_group_id(worker_id, step.group_id)
                stored_actions = stored_take_action_outputs["action"]
                action_tuple = ActionTuple(
                    continuous=stored_actions.continuous[idx],
                    discrete=stored_actions.discrete[idx],
                )
                group_status = AgentStatus(
                    obs=stored_decision_step.obs,
                    reward=step.reward,
                    action=action_tuple,
                    done=isinstance(step, TerminalStep),
                )
                self._group_status[global_group_id][global_agent_id] = group_status
                self._current_group_obs[global_group_id][global_agent_id] = step.obs

    def _clear_group_status_and_obs(self, global_id: GlobalAgentId) -> None:
        """
        Clears an agent from self._group_status and self._current_group_obs.
        """
        self._delete_in_nested_dict(self._current_group_obs, global_id)
        self._delete_in_nested_dict(self._group_status, global_id)

    def _delete_in_nested_dict(self, nested_dict: Dict[str, Any], key: str) -> None:
        for _manager_id in list(nested_dict.keys()):
            _team_group = nested_dict[_manager_id]
            self._safe_delete(_team_group, key)
            if not _team_group:  # if dict is empty
                self._safe_delete(nested_dict, _manager_id)

    def _process_step(
        self, step: Union[TerminalStep, DecisionStep], worker_id: int, index: int
    ) -> None:
        terminated = isinstance(step, TerminalStep)
        global_agent_id = get_global_agent_id(worker_id, step.agent_id)
        global_group_id = get_global_group_id(worker_id, step.group_id)
        stored_decision_step, idx = self._last_step_result.get(
            global_agent_id, (None, None)
        )
        stored_take_action_outputs = self._last_take_action_outputs.get(
            global_agent_id, None
        )
        if not terminated:
            # Index is needed to grab from last_take_action_outputs
            self._last_step_result[global_agent_id] = (step, index)

        # This state is the consequence of a past action
        if stored_decision_step is not None and stored_take_action_outputs is not None:
            obs = stored_decision_step.obs
            if self.policy.use_recurrent:
                memory = self.policy.retrieve_previous_memories([global_agent_id])[0, :]
            else:
                memory = None
            done = terminated  # Since this is an ongoing step
            interrupted = step.interrupted if terminated else False
            # Add the outputs of the last eval
            stored_actions = stored_take_action_outputs["action"]
            action_tuple = ActionTuple(
                continuous=stored_actions.continuous[idx],
                discrete=stored_actions.discrete[idx],
            )
            try:
                stored_action_probs = stored_take_action_outputs["log_probs"]
                if not isinstance(stored_action_probs, LogProbsTuple):
                    stored_action_probs = stored_action_probs.to_log_probs_tuple()
                log_probs_tuple = LogProbsTuple(
                    continuous=stored_action_probs.continuous[idx],
                    discrete=stored_action_probs.discrete[idx],
                )
            except KeyError:
                log_probs_tuple = LogProbsTuple.empty_log_probs()

            supertrack_data = None
            if "supertrack_data" in stored_take_action_outputs:
                supertrack_data = stored_take_action_outputs["supertrack_data"][idx]

            action_mask = stored_decision_step.action_mask
            prev_action = self.policy.retrieve_previous_action([global_agent_id])[0, :]

            # Assemble teammate_obs. If none saved, then it will be an empty list.
            group_statuses = []
            for _id, _mate_status in self._group_status[global_group_id].items():
                if _id != global_agent_id:
                    group_statuses.append(_mate_status)

            experience = AgentExperience(
                obs=obs,
                reward=step.reward,
                done=done,
                action=action_tuple,
                action_probs=log_probs_tuple,
                action_mask=action_mask,
                prev_action=prev_action,
                interrupted=interrupted,
                memory=memory,
                group_status=group_statuses,
                group_reward=step.group_reward,
                supertrack_data=supertrack_data,
            )
            # Add the value outputs if needed
            self._experience_buffers[global_agent_id].append(experience)
            self._episode_rewards[global_agent_id] += step.reward
            if not terminated:
                self._episode_steps[global_agent_id] += 1

            # Add a trajectory segment to the buffer if terminal or the length has reached the time horizon
            has_reached_trajectory_len = (not self._process_trajectory_on_termination) and len(self._experience_buffers[global_agent_id]) >= self._max_trajectory_length
            if (has_reached_trajectory_len or terminated):
                next_obs = step.obs
                next_group_obs = []
                for _id, _obs in self._current_group_obs[global_group_id].items():
                    if _id != global_agent_id:
                        next_group_obs.append(_obs)

                trajectory = Trajectory(
                    steps=self._experience_buffers[global_agent_id],
                    agent_id=global_agent_id,
                    next_obs=next_obs,
                    next_group_obs=next_group_obs,
                    behavior_id=self._behavior_id,
                )
                
                for traj_queue in self._trajectory_queues:
                    # trajectory.steps = trajectory.steps[:3]
                    traj_queue.put(trajectory)
                self._experience_buffers[global_agent_id] = []
                # print(f"Agent {global_agent_id} terminated at: {self._episode_steps.get(global_agent_id, 0)} steps")
            if terminated:
                # Record episode length.
                self._stats_reporter.add_stat(
                    "Environment/Episode Length",
                    self._episode_steps.get(global_agent_id, 0),
                )
                episode_end_type_str = "Interrupted" if interrupted else "SelfTerminated"
                self._stats_reporter.add_stat(f"Environment/{episode_end_type_str}", 1, StatsAggregationMethod.SUM)
                self._clean_agent_data(global_agent_id)

    def _clean_agent_data(self, global_id: GlobalAgentId) -> None:
        """
        Removes the data for an Agent.
        """
        self._safe_delete(self._experience_buffers, global_id)
        self._safe_delete(self._last_take_action_outputs, global_id)
        self._safe_delete(self._last_step_result, global_id)
        self._safe_delete(self._episode_steps, global_id)
        self._safe_delete(self._episode_rewards, global_id)
        self.policy.remove_previous_action([global_id])
        self.policy.remove_memories([global_id])

    def _safe_delete(self, my_dictionary: Dict[Any, Any], key: Any) -> None:
        """
        Safe removes data from a dictionary. If not found,
        don't delete.
        """
        if key in my_dictionary:
            del my_dictionary[key]

    def publish_trajectory_queue(
        self, trajectory_queue: "AgentManagerQueue[Trajectory]"
    ) -> None:
        """
        Adds a trajectory queue to the list of queues to publish to when this AgentProcessor
        assembles a Trajectory
        :param trajectory_queue: Trajectory queue to publish to.
        """
        self._trajectory_queues.append(trajectory_queue)

    def end_episode(self) -> None:
        """
        Ends the episode, terminating the current trajectory and stopping stats collection for that
        episode. Used for forceful reset (e.g. in curriculum or generalization training.)
        """
        all_gids = list(self._experience_buffers.keys())  # Need to make copy
        for _gid in all_gids:
            self._clean_agent_data(_gid)


class AgentManagerQueue(Generic[T]):
    """
    Queue used by the AgentManager. Note that we make our own class here because in most implementations
    deque is sufficient and faster. However, if we want to switch to multiprocessing, we'll need to change
    out this implementation.
    """

    class Empty(Exception):
        """
        Exception for when the queue is empty.
        """

        pass

    def __repr__(self) -> str:
        return self.name or super().__repr__()

    def __init__(self, behavior_id: str, maxlen: int = 0, use_pytorch_mp: bool = False, name : str = None, use_simple_queue : bool = False):
        """
        Initializes an AgentManagerQueue. Note that we can give it a behavior_id so that it can be identified
        separately from an AgentManager.
        """
        self.name = name
        if self.name == "trajectory_queue":
            print("Initializing trajectory queue")
        self._maxlen: int = maxlen
        self.use_pytorch_mp = use_pytorch_mp
        self.use_simple_queue = use_simple_queue

        if use_pytorch_mp:
            if use_simple_queue:
                self._queue = simple_queue_with_size.SimpleQueueWithSize(name=name, maxsize=maxlen)
            else:
                if sys.platform == 'win32':
                    self._queue = mp_queue.TorchQueue(name=name, maxsize=maxlen)
                else:
                    self._queue = torch.multiprocessing.Queue(maxsize=maxlen)
        else:
            self._queue: queue.Queue = queue.Queue(maxsize=maxlen)
        self._behavior_id = behavior_id

    @property
    def maxlen(self):
        """
        The maximum length of the queue.
        :return: Maximum length of the queue.
        """
        return self._maxlen

    @property
    def behavior_id(self):
        """
        The Behavior ID of this queue.
        :return: Behavior ID associated with the queue.
        """
        return self._behavior_id

    def qsize(self) -> int:
        """
        Returns the approximate size of the queue. Note that values may differ
        depending on the underlying queue implementation.
        """
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def get_nowait(self) -> T:
        """
        Gets the next item from the queue, throwing an AgentManagerQueue.Empty exception
        if the queue is empty.
        """
        try:
            if self.use_simple_queue:
                if self._queue.empty():
                    return None
                return self._queue.get()
            return self._queue.get_nowait()
        except queue.Empty:
            raise self.Empty("The AgentManagerQueue is empty.")
        except Exception as e:
            print(f"failed to get item from queue: {e}")
            raise e

    def put(self, item: T, block : bool = True) -> None:
        try:
            if self.use_simple_queue:
                return self._queue.put(item)
            self._queue.put(item, block=block)
        except Exception as e:
            logger.error(f"failed to put item in queue: {e}")

    def close(self):
        if self.use_pytorch_mp and not self.use_simple_queue:
            self._queue.close()
    
    def join_thread(self):
        if self.use_pytorch_mp and not self.use_simple_queue:
            self._queue.join_thread()

    def cancel_join_thread(self):
        if self.use_pytorch_mp and not self.use_simple_queue:
            self._queue.cancel_join_thread()

class AgentManager(AgentProcessor):
    """
    An AgentManager is an AgentProcessor that also holds a single trajectory and policy queue.
    Note: this leaves room for adding AgentProcessors that publish multiple trajectory queues.
    """

    def __init__(
        self,
        policy: Policy,
        behavior_id: str,
        stats_reporter: StatsReporter,
        torch_settings : TorchSettings,
        max_trajectory_length: int = sys.maxsize,
        threaded: bool = True,
        process_trajectory_on_termination: bool = False,
        use_pytorch_mp: bool = False,
    ):
        super().__init__(policy, behavior_id, stats_reporter, max_trajectory_length, process_trajectory_on_termination)
        # trajectory_queue_len = 20 if threaded or use_pytorch_mp else 0
        # For nsys profiling we want to allow infinite trajectory queue size and start profiling 
        # after hitting a certain number of trajectories dynamically
        # Otherwise we want as many traj as it takes to completley rewrite the buffer
        # For Supertrack, assuming 1 traj = 48 steps / entries, 150000 buffer size, 3072*48 ~= 150k  
        trajectory_queue_len = 0 if torch_settings.profile else 3072
        self.trajectory_queue: AgentManagerQueue[Trajectory] = AgentManagerQueue(
            self._behavior_id, maxlen=trajectory_queue_len, use_pytorch_mp=use_pytorch_mp, name = "trajectory_queue",  use_simple_queue=False #use_simple_queue=False
        )
        # NOTE: we make policy queues of infinite length to avoid lockups of the trainers.
        # In the environment manager, we make sure to empty the policy queue before continuing to produce steps.
        self.policy_queue: AgentManagerQueue[Policy] = AgentManagerQueue(
            self._behavior_id, maxlen=0, use_pytorch_mp=use_pytorch_mp, name = "policy_queue", use_simple_queue=True
        )
        self.publish_trajectory_queue(self.trajectory_queue)

    def record_environment_stats(
        self, env_stats: EnvironmentStats, worker_id: int
    ) -> None:
        """
        Pass stats from the environment to the StatsReporter.
        Depending on the StatsAggregationMethod, either StatsReporter.add_stat or StatsReporter.set_stat is used.
        The worker_id is used to determine whether StatsReporter.set_stat should be used.

        :param env_stats:
        :param worker_id:
        :return:
        """
        for stat_name, value_list in env_stats.items():
            for val, agg_type in value_list:
                if agg_type == StatsAggregationMethod.AVERAGE:
                    self._stats_reporter.add_stat(stat_name, val, agg_type)
                elif agg_type == StatsAggregationMethod.SUM:
                    self._stats_reporter.add_stat(stat_name, val, agg_type)
                elif agg_type == StatsAggregationMethod.HISTOGRAM:
                    self._stats_reporter.add_stat(stat_name, val, agg_type)
                elif agg_type == StatsAggregationMethod.MOST_RECENT:
                    # In order to prevent conflicts between multiple environments,
                    # only stats from the first environment are recorded.
                    if worker_id == 0:
                        self._stats_reporter.set_stat(stat_name, val)
                else:
                    raise UnityTrainerException(
                        f"Unknown StatsAggregationMethod encountered. {agg_type}"
                    )
