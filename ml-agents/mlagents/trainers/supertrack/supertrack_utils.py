from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Union, cast
import os
import functools
import pdb
from mlagents.torch_utils import torch, nn, default_device

from mlagents.trainers.buffer import AgentBuffer, AgentBufferField, BufferKey, ObservationKeyPrefix


import pytorch3d.transforms as pyt

import numpy as np

from mlagents_envs.timers import timed

TOTAL_OBS_LEN = 720
CHAR_STATE_LEN = 259
NUM_BONES = 17
NUM_T_BONES = 16 # Number of bones that have PD Motors (T = targets)
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
    positions: Union[np.ndarray, torch.Tensor]
    rotations: Union[np.ndarray, torch.Tensor]
    velocities: Union[np.ndarray, torch.Tensor]
    rot_velocities: Union[np.ndarray, torch.Tensor]
    heights: Union[np.ndarray, torch.Tensor]
    up_dir: Union[np.ndarray, torch.Tensor]
    
    @functools.cached_property
    def as_tensors(self):
        if not torch.is_tensor(self.positions):
            return torch.tensor(self.positions, dtype=torch.float32), torch.tensor(self.rotations, dtype=torch.float32), torch.tensor(self.velocities, dtype=torch.float32), torch.tensor(self.rot_velocities, dtype=torch.float32), torch.tensor(self.heights, dtype=torch.float32), torch.tensor(self.up_dir, dtype=torch.float32)#[None, :]
        return self.positions, self.rotations, self.velocities, self.rot_velocities, self.heights, self.up_dir
    
    @functools.cached_property
    def values(self):
        return self.positions, self.rotations, self.velocities, self.rot_velocities, self.heights, self.up_dir

    
@dataclass
class PDTargets():
    rotations : Union[np.ndarray, torch.Tensor]
    rot_velocities : Union[np.ndarray, torch.Tensor]

    @functools.cached_property
    def as_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]: 
        if not torch.is_tensor(self.rotations):
            return torch.tensor(self.rotations, dtype=torch.float32), torch.tensor(self.rot_velocities, dtype=torch.float32)
        return self.rotations, self.rot_velocities
    
    @functools.cached_property
    def values(self):
        return self.rotations, self.rot_velocities


@dataclass
class SuperTrackDataField():
    sim_char_state : CharState
    kin_char_state: CharState
    pre_targets: PDTargets
    post_targets: PDTargets
    
class SupertrackUtils:

    @staticmethod
    def convert_actions_to_quat(actions: torch.Tensor, # shape: (B, num_bones, 3)
                                alpha: float = 120
                               ) -> torch.Tensor:
        """
        The PD offsets, represented by vectors, are converted to rotations 
        via the exponential map as described in Grassia (1998). These rotations
        are then multiplied by the joint rotations of the kinematic character
        to derive the final PD targets as t = exp(𝛼/2 * o) ⊗ k_t. Here, 𝛼 
        serves as a scaling factor for the offsets.
        """
        return pyt.axis_angle_to_quaternion(actions * alpha)
        
    @staticmethod
    def process_raw_observations_to_policy_input(st_data : SuperTrackDataField) -> torch.Tensor:
        # 
        """
        Take inputs directly from Unity and transform them into a form that can be used as input to the policy.
        """
        # Local sim char state
        sim_char_state = st_data.sim_char_state
        sim_inputs = [t[None, ...] for t in sim_char_state.values] # Add batch dim
        local_sim = SupertrackUtils.local(*sim_inputs)
        # Local kin char state
        kin_char_state = st_data.kin_char_state
        kin_inputs = [t[None, ...] for t in kin_char_state.values] # Add batch dim
        local_kin = SupertrackUtils.local(*kin_inputs)
        return torch.cat((*local_kin, *local_sim), dim=-1)

    @staticmethod
    def extract_char_state(obs: Union[torch.tensor, np.ndarray], idx: int, use_tensor: bool) -> (CharState, int):
        if use_tensor:
            positions: torch.Tensor = torch.zeros((NUM_BONES, 3), device=obs.device)
            rotations: torch.Tensor = torch.zeros((NUM_BONES, 4), device=obs.device)
            velocities: torch.Tensor = torch.zeros((NUM_BONES, 3), device=obs.device)
            rot_velocities: torch.Tensor = torch.zeros((NUM_BONES, 3), device=obs.device)
            heights: torch.Tensor = torch.zeros(NUM_BONES, device=obs.device)
        else:
            positions: np.ndarray = np.zeros((NUM_BONES, 3))
            rotations: np.ndarray = np.zeros((NUM_BONES, 4))
            velocities: np.ndarray = np.zeros((NUM_BONES, 3))
            rot_velocities: np.ndarray = np.zeros((NUM_BONES, 3))
            heights: np.ndarray = np.zeros(NUM_BONES)

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
    def extract_pd_targets(obs: Union[torch.tensor, np.ndarray], idx, use_tensor : bool) -> (PDTargets, int):
        if use_tensor:
            rotations = torch.zeros((NUM_BONES, 4), device=obs.device)
            rot_velocities = torch.zeros((NUM_BONES, 3), device=obs.device)
        else:
            rotations = np.zeros((NUM_BONES, 4))
            rot_velocities = np.zeros((NUM_BONES, 3))

        for i in range(NUM_BONES):
            rotations[i] = obs[idx:idx+4]
            idx += 4
            rot_velocities[i] = obs[idx:idx+3]
            idx += 3
        return PDTargets(rotations, rot_velocities), idx 
    
    
    @staticmethod
    def parse_supertrack_data_field(inputs: List[Union[torch.tensor, np.ndarray]]) -> AgentBuffer:
        if len(inputs) != 1:
            raise Exception(f"SupertrackUtils.parse_supertrack_data_field expected inputs to be of len 1, got {len(inputs)}")
        obs = inputs[0]
        idx = 0
        use_tensor = torch.is_tensor(obs) 
        sim_char_state, idx = SupertrackUtils.extract_char_state(obs, idx, use_tensor)
        # Extract kin char state
        kin_char_state, idx = SupertrackUtils.extract_char_state(obs, idx, use_tensor)
        # Extract pre_targets
        pre_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx, use_tensor)
        # Extract post_targets
        post_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx, use_tensor)
        if idx != TOTAL_OBS_LEN:
            raise Exception(f'idx was {idx} expected {TOTAL_OBS_LEN}')
        return SuperTrackDataField(
            sim_char_state=sim_char_state, 
            kin_char_state=kin_char_state,
            pre_targets=pre_targets,
            post_targets=post_targets)
    

    @staticmethod
    def add_supertrack_data_field_OLD(agent_buffer_trajectory: AgentBuffer) -> AgentBuffer:
        supertrack_data = AgentBufferField()
        for i in range(agent_buffer_trajectory.num_experiences):
            obs = agent_buffer_trajectory[(ObservationKeyPrefix.OBSERVATION, 0)][i]
            if (len(obs) != TOTAL_OBS_LEN):
                raise Exception(f'Obs was of len {len(obs)} expected {TOTAL_OBS_LEN}')
            # print(f"Obs at idx {agent_buffer_trajectory[BufferKey.IDX_IN_TRAJ][i]} : {obs}")
            supertrack_data.append(SupertrackUtils.parse_supertrack_data_field([obs]))
        agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA] = supertrack_data
    
    @staticmethod
    def split_world_model_output(x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        B = x.shape[0]
        pos_a = x[:, x.shape[-1]//2:].reshape(B, -1, 3)
        rot_a = x[:, :x.shape[-1]//2].reshape(B, -1, 3)
        return pos_a, rot_a

    @staticmethod 
    def normalize_quat(quaternions, epsilon=1e-6):
        """
        Replace quaternions with L2 norm close to zero with the identity quaternion.

        Args:
            quaternions: Input quaternions as a tensor of shape (..., 4).
            epsilon: Threshold to determine if L2 norm is close to zero.

        Returns:
            Quaternions with small L2 norm replaced by the identity quaternion.
        """
        # Compute the L2 norm squared for each quaternion
        # norm_squared = (quaternions ** 2).sum(dim=-1)
        norms = torch.norm(quaternions, p=2, dim=-1, keepdim=True)
        # Create a mask for quaternions with L2 norm close to zero
        mask = norms < epsilon
        
        # Create a tensor of identity quaternions
        identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quaternions.device)
        
        # Use torch.where to replace the quaternions with identity
        result = torch.where(mask, identity_quaternion, quaternions)
        
        return result / torch.norm(result, dim=-1, keepdim=True)
    

    @staticmethod
    @timed
    def local(cur_pos: torch.Tensor, # shape [batch_size, num_bones, 3]
            cur_rots: torch.Tensor, # shape [batch_size, num_bones, 4]
            cur_vels: torch.Tensor,   # shape [batch_size, num_bones, 3]
            cur_rot_vels: torch.Tensor, # shape [batch_size, num_bones, 3]
            cur_heights: torch.Tensor, # shape [batch_size, num_bones]
            cur_up_dir: torch.Tensor): # shape [batch_size, 3]
        B = cur_pos.shape[0] # batch_size
        root_pos = cur_pos[:, 0:1 , :] # shape [batch_size, 1, 3]
        inv_root_rots = pyt.quaternion_invert(cur_rots[:, 0:1, :]) # shape [batch_size, 1, 4]
        local_pos = pyt.quaternion_apply(inv_root_rots, cur_pos[:, 1:, :] - root_pos) # shape [batch_size, num_t_bones, 3]
        local_rots = pyt.quaternion_multiply(inv_root_rots, cur_rots[:, 1:, :]) # shape [batch_size, num_t_bones, 4]
        two_axis_rots = pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(SupertrackUtils.normalize_quat(local_rots)).reshape(-1, 3, 3)) # shape [batch_size * num_t_bones, 6]
        local_vels = pyt.quaternion_apply(inv_root_rots, cur_vels[:, 1:, :]) # shape [batch_size, num_t_bones, 3]
        local_rot_vels = pyt.quaternion_apply(inv_root_rots, cur_rot_vels[:, 1:, :]) # shape [batch_size, num_t_bones, 3]

        # return_tensors = [local_pos, two_axis_rots, local_vels, local_rot_vels, cur_heights[:, 1:], cur_up_dir]
        return_tensors = [(local_pos, 'local_pos'), (two_axis_rots, 'two_axis_rots'), (local_vels, 'local_vels'), (local_rot_vels, 'local_rot_vels'), (cur_heights[:, 1:], 'cur_heights'), (cur_up_dir, 'cur_up_dir')]
        # Have to reshape instead of view because stride can be messed up in some cases
        # return [tensor.reshape(B, -1) for tensor in return_tensors]
        # for tensor, name in return_tensors:
        #     print(f"{name} dtype: {tensor.dtype}")
        return [tensor.reshape(B, -1) for tensor, name in return_tensors]

        