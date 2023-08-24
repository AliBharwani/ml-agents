from collections import defaultdict
from typing import Dict, cast
import os
from mlagents.trainers import supertrack

from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents.trainers.supertrack_buffer import CharState, PDTargets, SuperTrackDataField

from mlagents.trainers.trajectory import ObsUtil
import attr


# from mlagents.trainers.trajectory import Trajectory

from mlagents.trainers.trajectory import Trajectory


from mlagents.trainers.torch_entities.networks import SimpleActor

from mlagents_envs.base_env import BehaviorSpec

from mlagents.trainers.policy.torch_policy import TorchPolicy

import numpy as np
from mlagents.trainers.policy.checkpoint_manager import ModelCheckpoint

from mlagents_envs.logging_util import get_logger
from mlagents_envs.timers import timed
from mlagents.trainers.buffer import RewardSignalUtil
from mlagents.trainers.policy import Policy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from mlagents.trainers.supertrack.optimizer_torch import SuperTrackPolicyNetwork, TorchSuperTrackOptimizer, SuperTrackSettings


TOTAL_ACTION_LEN = 576
CHAR_STATE_LEN = 259
NUM_BONES = 17
ENTRIES_PER_BONE = 13


class Quat():
    w, x, y, z : int
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

def extract_char_state(all_actions, idx) -> (CharState, int):
    positions = []
    rotations = []
    velocities = []
    rot_velocities = []
    for _ in range(NUM_BONES):
        positions.append(all_actions[idx:idx+3])
        idx += 3
        rotations.append(Quat(all_actions[idx], all_actions[idx + 1], all_actions[idx + 2], all_actions[idx + 3]))
        idx += 4
        velocities.append(all_actions[idx:idx+3])
        idx += 3
        rot_velocities.append(all_actions[idx:idx+3])
        idx += 3
    return CharState(positions, rotations, velocities, rot_velocities), idx 

def extract_pd_targets(all_actions, idx) -> (PDTargets, int):
    rotations = []
    rot_velocities = []
    for _ in range(NUM_BONES):
        rotations.append(Quat(all_actions[idx], all_actions[idx + 1], all_actions[idx + 2], all_actions[idx + 3]))
        idx += 4
        rot_velocities.append(all_actions[idx:idx+3])
        idx += 3
    return PDTargets(rotations, rot_velocities), idx 

def add_supertrack_data_field(agent_buffer_trajectory: AgentBuffer) -> AgentBuffer:
    supertrack_data = []
    for i in range(len(agent_buffer_trajectory)):
        actions = agent_buffer_trajectory[BufferKey.CONTINUOUS_ACTION][i]
        if (len(actions) != TOTAL_ACTION_LEN):
            raise Exception(f'Actions was of len {len(actions)} expected {TOTAL_ACTION_LEN}')
        # Extract sim char state
        idx = 0
        sim_char_state, idx = extract_char_state(actions, idx)
        # Extract kin char state
        kin_char_state, idx = extract_char_state(actions, idx)
        # Extract pre_targets
        pre_targets, idx = extract_pd_targets(actions, idx)
        # Extract post_targets
        post_targets, idx = extract_pd_targets(actions, idx)
        supertrack_data.append(
            SuperTrackDataField(
            sim_char_state=sim_char_state, 
            kin_char_state=kin_char_state,
            pre_targets=pre_targets,
            post_targets=post_targets))


    agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA] = supertrack_data
    