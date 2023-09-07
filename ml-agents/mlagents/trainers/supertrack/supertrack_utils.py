from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, cast
import os
import functools
from mlagents.torch_utils import torch
from mlagents.trainers import quat, supertrack

from mlagents.trainers.buffer import AgentBuffer, AgentBufferField, BufferKey, ObservationKeyPrefix
from mlagents.trainers.torch_entities.utils import ModelUtils

from mlagents.trainers.trajectory import ObsUtil
import attr

import pytorch3d.transforms as pyt
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


TOTAL_OBS_LEN = 720
CHAR_STATE_LEN = 259
NUM_BONES = 17
NUM_OUTPUT_BONES = NUM_BONES - 1
ENTRIES_PER_BONE = 13
POLICY_INPUT_LEN = 518


class Quat():
    w : int; x : int; y : int; z : int
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def inv(self) -> 'Quat':
        return Quat(self.w, -self.x, -self.y, -self.z)
    

    def vec(self):
        """
        Return the imaginary part of the quaternion
        """
        return np.asarray([self.x, self.y, self.z])
    
        
class Bone(Enum):
    ROOT = 0
    HIP = 1
    SPINE = 2

@dataclass
class CharState():
    positions : np.ndarray
    rotations : np.ndarray
    velocities : np.ndarray
    rot_velocities : np.ndarray
    heights : np.ndarray
    up_dir : np.ndarray
    # rotations_two_axis_form: np.ndarray

    def to_tensors(self):
        # add a dimension to up_dir to make it a 1x3 tensor, so that we can use similar logic 
        return torch.tensor(self.positions), torch.tensor(self.rotations), torch.tensor(self.velocities), torch.tensor(self.rot_velocities), torch.tensor(self.heights), torch.tensor(self.up_dir)#[None, :]

@dataclass
class PDTargets():
    rotations : np.ndarray
    rot_velocities : np.ndarray
    # rotations_two_axis_form: np.ndarray

    @functools.cached_property
    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        return torch.tensor(self.rotations), torch.tensor(self.rot_velocities)

@dataclass
class SuperTrackDataField():
    sim_char_state : CharState
    kin_char_state: CharState
    pre_targets: PDTargets
    post_targets: PDTargets
    

class SupertrackUtils:

    @staticmethod
    def process_raw_observations_to_policy_input(inputs : torch.Tensor) -> torch.Tensor:
        return inputs[:, :POLICY_INPUT_LEN]

    @staticmethod
    def extract_char_state(obs: np.ndarray, idx: int) -> (CharState, int):
        positions: np.ndarray = np.zeros((NUM_BONES, 3))
        rotations: np.ndarray = np.zeros((NUM_BONES, 4))
        velocities: np.ndarray = np.zeros((NUM_BONES, 3))
        rot_velocities: np.ndarray = np.zeros((NUM_BONES, 3))
        heights: np.ndarray = np.zeros(NUM_BONES)
        # rotations_two_axis_form: np.ndarray = np.zeros((NUM_BONES, 6))
        for i in range(NUM_BONES):
            positions[i] = obs[idx:idx+3]
            idx += 3
            rotations[i] = obs[idx:idx+4]
            idx += 4
            velocities[i] = obs[idx:idx+3]
            idx += 3
            rot_velocities[i] = obs[idx:idx+3]
            idx += 3
            heights[i] = obs[idx]
            idx += 1
        up_dir = obs[idx:idx+3]
        idx += 3
        return CharState(positions,
                        rotations,
                        velocities,
                        rot_velocities, 
                        heights,
                        up_dir), idx 

    @staticmethod
    def extract_pd_targets(obs, idx) -> (PDTargets, int):
        rotations = np.zeros((NUM_BONES, 4))
        rot_velocities = np.zeros((NUM_BONES, 3))
        # rotations_two_axis_form: List[np.ndarray] = np.zeros((NUM_BONES, 6))
        for i in range(NUM_BONES):
            rotations[i] = obs[idx:idx+4]
            idx += 4
            rot_velocities[i] = obs[idx:idx+3]
            idx += 3
        return PDTargets(rotations, rot_velocities), idx 

    @staticmethod
    def add_supertrack_data_field(agent_buffer_trajectory: AgentBuffer) -> AgentBuffer:
        supertrack_data = AgentBufferField()
        for i in range(agent_buffer_trajectory.num_experiences):
            obs = agent_buffer_trajectory[(ObservationKeyPrefix.OBSERVATION, 0)][i]
            if (len(obs) != TOTAL_OBS_LEN):
                raise Exception(f'Obs was of len {len(obs)} expected {TOTAL_OBS_LEN}')
            # print(f"Obs at idx {agent_buffer_trajectory[BufferKey.IDX_IN_TRAJ][i]} : {obs}")
            # Extract sim char state
            idx = 0
            sim_char_state, idx = SupertrackUtils.extract_char_state(obs, idx)
            # Extract kin char state
            kin_char_state, idx = SupertrackUtils.extract_char_state(obs, idx)
            # Extract pre_targets
            pre_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx)
            # Extract post_targets
            post_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx)
            supertrack_data.append(
                SuperTrackDataField(
                sim_char_state=sim_char_state, 
                kin_char_state=kin_char_state,
                pre_targets=pre_targets,
                post_targets=post_targets))
        agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA] = supertrack_data
    
    @staticmethod
    def split_world_model_output(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        B = x.shape[0]
        pos_a = x[:, x.shape[-1]//2:].reshape(B, -1, 3)
        rot_a = x[:, :x.shape[-1]//2].reshape(B, -1, 3)
        return pos_a, rot_a
    
    @staticmethod
    def integrate_accel(cur_state: CharState, accel: torch.Tensor, rot_accel: torch.Tensor, dt : float = 1/60) -> CharState:
        # We have global accelerations, velocities, and positions. How do I
        
        next_state = SuperTrackDataField()
        return next_state
        