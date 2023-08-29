from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, cast
import os
from mlagents.torch_utils import torch
from mlagents.trainers import supertrack

from mlagents.trainers.buffer import AgentBuffer, BufferKey, ObservationKeyPrefix

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


TOTAL_OBS_LEN = 704
CHAR_STATE_LEN = 259
NUM_BONES = 17
ENTRIES_PER_BONE = 13

class Quat():
    w : int; x : int; y : int; z : int
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        
class Bone(Enum):
    ROOT = 0
    HIP = 1
    SPINE = 2

@dataclass
class CharState():
    positions : List[np.ndarray]
    rotations : List[Quat]
    velocities : List[np.ndarray]
    rot_velocities : List[np.ndarray]
    heights : List[float]
    up_dir : np.ndarray
    rotations_two_axis_form: List[np.ndarray]

@dataclass
class PDTargets():
    rotations : List[Quat]
    rot_velocities : List[np.ndarray]
    rotations_two_axis_form: List[np.ndarray]

@dataclass
class SuperTrackDataField():
    sim_char_state : CharState
    kin_char_state: CharState
    pre_targets: PDTargets
    post_targets: PDTargets
    


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
        actions = agent_buffer_trajectory[(ObservationKeyPrefix.OBSERVATION, 0)][i]
        if (len(actions) != TOTAL_OBS_LEN):
            raise Exception(f'Obs was of len {len(actions)} expected {TOTAL_OBS_LEN}')
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
    

def process_raw_observations_to_policy_input(inputs : torch.Tensor) -> torch.Tensor:
    return inputs[:, :518]