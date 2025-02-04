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
from mlagents.trainers.torch_entities.utils import ModelUtils

from mlagents_envs.timers import hierarchical_timer, timed

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
    
    # @functools.cached_property
    def as_tensors(self, device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(self.positions):
            raise Exception("CharState.as_tensors called on non-tensor object")
            return torch.tensor(self.positions, dtype=torch.float32), torch.tensor(self.rotations, dtype=torch.float32), torch.tensor(self.velocities, dtype=torch.float32), torch.tensor(self.rot_velocities, dtype=torch.float32), torch.tensor(self.heights, dtype=torch.float32), torch.tensor(self.up_dir, dtype=torch.float32)#[None, :]
        return self.positions.to(device, non_blocking=True), self.rotations.to(device, non_blocking=True), self.velocities.to(device, non_blocking=True), self.rot_velocities.to(device, non_blocking=True), self.heights.to(device, non_blocking=True), self.up_dir.to(device, non_blocking=True)
    
    # @functools.cached_property
    def values(self):
        return self.positions, self.rotations, self.velocities, self.rot_velocities, self.heights, self.up_dir
    
    def to_numpy(self):
        for attr in ['positions', 'rotations', 'velocities', 'rot_velocities', 'heights', 'up_dir']:
            current_attr = getattr(self, attr)
            if not isinstance(current_attr, np.ndarray):
                setattr(self, attr, ModelUtils.to_numpy(current_attr))
        # if isinstance(self.positions, np.ndarray):
        #     return
        # self.positions = ModelUtils.to_numpy(self.positions)
        # self.rotations = ModelUtils.to_numpy(self.rotations)
        # self.velocities = ModelUtils.to_numpy(self.velocities)
        # self.rot_velocities = ModelUtils.to_numpy(self.rot_velocities)
        # self.heights = ModelUtils.to_numpy(self.heights)
        # self.up_dir = ModelUtils.to_numpy(self.up_dir)
    
    def to(self, device, clone = False):
        for attr in ['positions', 'rotations', 'velocities', 'rot_velocities', 'heights', 'up_dir']:
            current_attr = getattr(self, attr)
            # cloned = current_attr.clone()
            # setattr(self, attr, cloned.to(device))
            setattr(self, attr, current_attr.to(device, non_blocking=True))
            # if torch.is_tensor(current_attr):


    
@dataclass
class PDTargets():
    rotations : Union[np.ndarray, torch.Tensor]
    rot_velocities : Union[np.ndarray, torch.Tensor]

    # @functools.cached_property
    def as_tensors(self, device = None) -> Tuple[torch.Tensor, torch.Tensor]: 
        if not torch.is_tensor(self.rotations):
            raise Exception("PDTargets.as_tensors called on non-tensor object")
            return torch.tensor(self.rotations, dtype=torch.float32), torch.tensor(self.rot_velocities, dtype=torch.float32)
        return self.rotations.to(device, non_blocking=True), self.rot_velocities.to(device, non_blocking=True)
    
    # @functools.cached_property
    def values(self):
        return self.rotations, self.rot_velocities
    
    def to_numpy(self):
        for attr in ['rotations', 'rot_velocities']:
            current_attr = getattr(self, attr)
            if not isinstance(current_attr, np.ndarray):
                setattr(self, attr, ModelUtils.to_numpy(current_attr))
    
    def to(self, device, clone = False):
        for attr in ['rotations', 'rot_velocities']:
            current_attr = getattr(self, attr)
            # cloned = current_attr.clone()
            # setattr(self, attr, cloned.to(device))
            # if torch.is_tensor(current_attr):
            setattr(self, attr, current_attr.to(device, non_blocking=True))


@dataclass
class SuperTrackDataField():
    sim_char_state : CharState
    kin_char_state: CharState
    pre_targets: PDTargets
    post_targets: PDTargets

    def convert_to_numpy(self):
        self.sim_char_state.to_numpy()
        self.kin_char_state.to_numpy()
        self.pre_targets.to_numpy()
        self.post_targets.to_numpy()

    def to(self, device, clone = False):
        for attr in ['sim_char_state', 'kin_char_state', 'pre_targets', 'post_targets']:
            current_attr = getattr(self, attr)
            current_attr.to(device, clone=clone)
    
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
    def process_raw_observations_to_policy_input(st_data : List[SuperTrackDataField]) -> torch.Tensor:
        """
        Take inputs directly from Unity and transform them into a form that can be used as input to the policy.
        """
        # Gives us a list of tensors of shape [(pos, rots, etc) of len batch_size ]
        sim_inputs = [st_datum.sim_char_state.values() for st_datum in st_data]
        # Convert them to [batch_size, num_bones, 3] for pos, [batch_size, num_bones, 4] for rots, etc
        sim_inputs = [torch.stack(t) for t in zip(*sim_inputs)]
        # SupertrackUtils.local expects tensors in the shape [batch_size, num_bones, 3] for pos, [batch_size, num_bones, 4] for rots, etc
        local_sim = SupertrackUtils.local(*sim_inputs) # Shape is now [batch_size, local sim input size]

        kin_inputs = [st_datum.kin_char_state.values() for st_datum in st_data]
        kin_inputs = [torch.stack(t) for t in zip(*kin_inputs)]
        local_kin = SupertrackUtils.local(*kin_inputs)
        return torch.cat((*local_kin, *local_sim), dim=-1)

    @staticmethod
    def extract_char_state(obs: Union[torch.tensor, np.ndarray], # obs is of shape [batch_size, TOTAL_OBS_LEN]
                            idx: int, use_tensor: bool, pin_memory: bool = False, device = None) -> (CharState, int):
        if use_tensor:
            if device is None:
                device = obs.device
            positions: torch.Tensor = torch.zeros((NUM_BONES, 3), device=device, pin_memory=pin_memory)
            rotations: torch.Tensor = torch.zeros((NUM_BONES, 4), device=device, pin_memory=pin_memory)
            velocities: torch.Tensor = torch.zeros((NUM_BONES, 3), device=device, pin_memory=pin_memory)
            rot_velocities: torch.Tensor = torch.zeros((NUM_BONES, 3), device=device, pin_memory=pin_memory)
            heights: torch.Tensor = torch.zeros(NUM_BONES, device=device, pin_memory=pin_memory)
            if type(obs) is np.ndarray:
                obs = torch.as_tensor(np.asanyarray(obs), device=device, dtype=torch.float32)
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
    def extract_char_state_batched(obs: Union[torch.tensor, np.ndarray], # obs is of shape [batch_size, TOTAL_OBS_LEN]
                            idx: int, use_tensor: bool, device = None , pin_memory: bool = False) -> (CharState, int):
        B = obs.shape[0]
        if use_tensor:
            if device is None:
                device = obs.device
            positions: torch.Tensor = torch.zeros((B, NUM_BONES, 3), device=device, pin_memory=pin_memory)
            rotations: torch.Tensor = torch.zeros((B, NUM_BONES, 4), device=device, pin_memory=pin_memory)
            velocities: torch.Tensor = torch.zeros((B, NUM_BONES, 3), device=device, pin_memory=pin_memory)
            rot_velocities: torch.Tensor = torch.zeros((B, NUM_BONES, 3), device=device, pin_memory=pin_memory)
            heights: torch.Tensor = torch.zeros(B, NUM_BONES, device=device, pin_memory=pin_memory)
        else:
            positions: np.ndarray = np.zeros((B, NUM_BONES, 3))
            rotations: np.ndarray = np.zeros((B, NUM_BONES, 4))
            velocities: np.ndarray = np.zeros((B, NUM_BONES, 3))
            rot_velocities: np.ndarray = np.zeros((B, NUM_BONES, 3))
            heights: np.ndarray = np.zeros(B, NUM_BONES)

        for i in range(NUM_BONES):
            positions[:,i] = obs[:, idx:idx+3]
            idx += 3
            rotations[:,i] = obs[:, idx:idx+4]
            idx += 4
            velocities[:,i] = obs[:, idx:idx+3]
            idx += 3
            rot_velocities[:,i] = obs[:,idx:idx+3]
            idx += 3
            heights[:,i] = obs[:,idx]
            idx += 1
        up_dir = obs[:,idx:idx+3]
        idx += 3
        return [CharState(positions[i],
                        rotations[i],
                        velocities[i],
                        rot_velocities[i], 
                        heights[i],
                        up_dir[i]) for i in range(B)], idx 
    
    @staticmethod
    def extract_pd_targets_batched(obs: Union[torch.tensor, np.ndarray], idx, use_tensor : bool, device = None, pin_memory: bool = False) -> (PDTargets, int):
        # obs is of shape [batch_size, TOTAL_OBS_LEN]
        B = obs.shape[0]
        if use_tensor:
            if device is None:
                device = obs.device
            rotations = torch.zeros((B, NUM_BONES, 4), device=device, pin_memory=pin_memory)
            rot_velocities = torch.zeros((B, NUM_BONES, 3), device=device, pin_memory=pin_memory)
        else:
            rotations = np.zeros((B, NUM_BONES, 4))
            rot_velocities = np.zeros((B, NUM_BONES, 3))

        for i in range(NUM_BONES):
            rotations[:, i] = obs[:, idx:idx+4]
            idx += 4
            rot_velocities[:, i] = obs[:, idx:idx+3]
            idx += 3
        return [PDTargets(rotations[i], rot_velocities[i]) for i in range(B)], idx 

    @staticmethod
    def extract_pd_targets(obs: Union[torch.tensor, np.ndarray], idx, use_tensor : bool, pin_memory: bool = False, device = None) -> (PDTargets, int):
        if use_tensor:
            if device is None:
                device = obs.device
            rotations = torch.zeros((NUM_BONES, 4), device=device, pin_memory=pin_memory, dtype=torch.float32)
            rot_velocities = torch.zeros((NUM_BONES, 3), device=device, pin_memory=pin_memory, dtype=torch.float32)
            if type(obs) is np.ndarray:
                obs = torch.as_tensor(np.asanyarray(obs), device=device, dtype=torch.float32)
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
    def parse_supertrack_data_field_batched(inputs: Union[torch.tensor, np.ndarray], device = None, pin_memory: bool = False) -> List[SuperTrackDataField]:
        if len(inputs.shape) != 2:
            raise Exception(f"SupertrackUtils.parse_supertrack_data_field_batched expected inputs to be of len 2 (batch_dim, data), got {len(inputs)}")
        use_tensor = torch.is_tensor(inputs)  # torch.is_tensor(inputs) 
        B = inputs.shape[0]
        idx = 0
        sim_char_state, idx = SupertrackUtils.extract_char_state_batched(inputs, idx, use_tensor, device=device, pin_memory=pin_memory)
        # Extract kin char state
        kin_char_state, idx = SupertrackUtils.extract_char_state_batched(inputs, idx, use_tensor, device=device, pin_memory=pin_memory)
        # Extract pre_targets
        pre_targets, idx = SupertrackUtils.extract_pd_targets_batched(inputs, idx, use_tensor, device=device, pin_memory=pin_memory)
        # Extract post_targets
        post_targets, idx = SupertrackUtils.extract_pd_targets_batched(inputs, idx, use_tensor, device=device, pin_memory=pin_memory)
        if idx != TOTAL_OBS_LEN:
            raise Exception(f'idx was {idx} expected {TOTAL_OBS_LEN}')
        return [SuperTrackDataField(
                sim_char_state=sim_char_state[i], 
                kin_char_state=kin_char_state[i],
                pre_targets=pre_targets[i],
                post_targets=post_targets[i]) for i in range(B)]
    
    @staticmethod
    def parse_supertrack_data_field(inputs: List[Union[torch.tensor, np.ndarray]], pin_memory: bool = False, device = None, use_tensor=None) -> AgentBuffer:
        if len(inputs) != 1:
            raise Exception(f"SupertrackUtils.parse_supertrack_data_field expected inputs to be of len 1 (data), got {len(inputs)}")
        use_tensor = torch.is_tensor(inputs) if use_tensor is None else use_tensor
        obs = inputs[0]
        idx = 0
        sim_char_state, idx = SupertrackUtils.extract_char_state(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        # Extract kin char state
        kin_char_state, idx = SupertrackUtils.extract_char_state(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        # Extract pre_targets
        pre_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        # Extract post_targets
        post_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        if idx != TOTAL_OBS_LEN:
            raise Exception(f'idx was {idx} expected {TOTAL_OBS_LEN}')
        return SuperTrackDataField(
            sim_char_state=sim_char_state, 
            kin_char_state=kin_char_state,
            pre_targets=pre_targets,
            post_targets=post_targets)
    

    @staticmethod
    def add_supertrack_data_field_OLD(agent_buffer_trajectory: AgentBuffer, pin_memory: bool = False, device = None) -> AgentBuffer:
        supertrack_data = AgentBufferField()
        # s_pos = 
        for i in range(agent_buffer_trajectory.num_experiences):
            obs = agent_buffer_trajectory[(ObservationKeyPrefix.OBSERVATION, 0)][i]
            if (len(obs) != TOTAL_OBS_LEN):
                raise Exception(f'Obs was of len {len(obs)} expected {TOTAL_OBS_LEN}')
            # print(f"Obs at idx {agent_buffer_trajectory[BufferKey.IDX_IN_TRAJ][i]} : {obs}")
            st_datum = SupertrackUtils.parse_supertrack_data_field([obs], pin_memory=pin_memory, device=device, use_tensor=True)
            supertrack_data.append(st_datum)

        agent_buffer_trajectory[BufferKey.SUPERTRACK_DATA] = supertrack_data

        # agent_buffer_trajectory['s_pos'] 

    
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
            cur_up_dir: torch.Tensor, # shape [batch_size, 3]
            rots_as_twoaxis: bool = True,
            unzip_to_batchsize: bool = True,
            ): 
        B = cur_pos.shape[0] # batch_size
        with hierarchical_timer("root_pos"):
            root_pos = cur_pos[:, 0:1 , :] # shape [batch_size, 1, 3]
        with hierarchical_timer("inv_root_rots"):
            inv_root_rots = pyt.quaternion_invert(cur_rots[:, 0:1, :]) # shape [batch_size, 1, 4]
        with hierarchical_timer("local_pos"):
            local_pos = pyt.quaternion_apply(inv_root_rots, cur_pos[:, 1:, :] - root_pos) # shape [batch_size, num_t_bones, 3]
        with hierarchical_timer("local_rots"):
            local_rots = pyt.quaternion_multiply(inv_root_rots, cur_rots[:, 1:, :]) # shape [batch_size, num_t_bones, 4]
        with hierarchical_timer("rots_as_twoaxis"):
            if rots_as_twoaxis:
                return_rots = pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(SupertrackUtils.normalize_quat(local_rots)).reshape(-1, 3, 3)) # shape [batch_size * num_t_bones, 6]
            else:
                return_rots = local_rots
        # two_axis_rots = pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(SupertrackUtils.normalize_quat(local_rots)).reshape(-1, 3, 3)) # shape [batch_size * num_t_bones, 6]
        with hierarchical_timer("local_vels"):
            local_vels = pyt.quaternion_apply(inv_root_rots, cur_vels[:, 1:, :]) # shape [batch_size, num_t_bones, 3]
        with hierarchical_timer("local_rot_vels"):
            local_rot_vels = pyt.quaternion_apply(inv_root_rots, cur_rot_vels[:, 1:, :]) # shape [batch_size, num_t_bones, 3]

        return_tensors = [local_pos, return_rots, local_vels, local_rot_vels, cur_heights[:, 1:], cur_up_dir]
        # return_tensors = [(local_pos, 'local_pos'), (return_rots, 'return_rots'), (local_vels, 'local_vels'), (local_rot_vels, 'local_rot_vels'), (cur_heights[:, 1:], 'cur_heights'), (cur_up_dir, 'cur_up_dir')]
        # Have to reshape instead of view because stride can be messed up in some cases
        if unzip_to_batchsize:
            return [t.reshape(B, -1) for t in return_tensors]
        else:
            return [t for t in return_tensors]
        