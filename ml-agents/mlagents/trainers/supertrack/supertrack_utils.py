from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import enum
import pdb
from typing import List, Tuple, Union
from mlagents.torch_utils import torch

from mlagents.trainers.buffer import AgentBuffer, AgentBufferField, BufferKey, ObservationKeyPrefix

import pytorch3d.transforms as pyt

import numpy as np
from mlagents.trainers.torch_entities.utils import ModelUtils
from torch import Tensor


TOTAL_OBS_LEN = 720
# CHAR_STATE_LEN = 240
NUM_BONES = 17
NUM_T_BONES = 16 # Number of bones that have PD Motors (T = targets)
POLICY_INPUT_LEN = 480
POLICY_OUTPUT_LEN = 48
MINIMUM_TRAJ_LEN = 48

class STSingleBufferKey(enum.Enum):
    IDX_IN_TRAJ = "idx_in_traj"
    TRAJ_LEN = "traj_len"
    RAW_OBS_DEBUG = "raw_obs_debug"

class PDTargetPrefix(enum.Enum):
    PRE = "pre"
    POST = "post"

class PDTargetSuffix(enum.Enum):
    ROT = "rot"
    RVEL = "rvel"

class CharTypePrefix(enum.Enum):
    SIM = "sim"
    KIN = "kin"

class CharTypeSuffix(enum.Enum):
    POSITION = "position"
    ROTATION = "rotation"
    VEL = "vel"
    RVEL = "rvel"
    HEIGHT = "height"
    UP_DIR = "up_dir"
        
class Bone(Enum):
    ROOT = 0
    HIP = 1
    SPINE = 2

@contextmanager
def nsys_profiler(name: str, profiler_running: bool):
    if profiler_running: torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if profiler_running: torch.cuda.nvtx.range_pop()
    

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
        return self.positions.to(device, non_blocking=True), self.rotations.to(device, non_blocking=True), self.velocities.to(device, non_blocking=True), self.rot_velocities.to(device, non_blocking=True)
    
    # @functools.cached_property
    def values(self):
        return self.positions, self.rotations, self.velocities, self.rot_velocities #, self.heights, self.up_dir
    
    def to_numpy(self):
        for attr in ['positions', 'rotations', 'velocities', 'rot_velocities', 'heights', 'up_dir']:
            current_attr = getattr(self, attr)
            if not isinstance(current_attr, np.ndarray):
                setattr(self, attr, ModelUtils.to_numpy(current_attr))

    def to(self, device):
        for attr in ['positions', 'rotations', 'velocities', 'rot_velocities', 'heights', 'up_dir']:
            current_attr = getattr(self, attr)
            setattr(self, attr, current_attr.to(device, non_blocking=True))

    
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
    
    def to(self, device):
        for attr in ['rotations', 'rot_velocities']:
            current_attr = getattr(self, attr)
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

    def to(self, device):
        for attr in ['sim_char_state', 'kin_char_state', 'pre_targets', 'post_targets']:
            current_attr = getattr(self, attr)
            current_attr.to(device)

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
    def process_raw_observations_to_policy_input(st_data : List[SuperTrackDataField], include_global_data: bool) -> Tuple[List[torch.Tensor], torch.Tensor | None]:
        """
        Take inputs directly from Unity and transform them into a form that can be used as input to the policy.
        """
        # Gives us a list of tensors of shape [(pos, rots, etc) of len batch_size ]
        sim_inputs = [st_datum.sim_char_state.values() for st_datum in st_data]
        # Convert them to [batch_size, num_t_bones, 3] for pos, [batch_size, num_t_bones, 4] for rots, etc
        sim_inputs = [torch.stack(t)[:, 1:, :] for t in zip(*sim_inputs)]
        # SupertrackUtils.local expects tensors in the shape [batch_size, num_bones, 3] for pos, [batch_size, num_bones, 4] for rots, etc
        local_sim = SupertrackUtils.local(*sim_inputs) # Shape is now [batch_size, local sim input size]

        kin_inputs = [st_datum.kin_char_state.values() for st_datum in st_data]
        kin_inputs = [torch.stack(t)[:, 1:, :] for t in zip(*kin_inputs)]
        local_kin = SupertrackUtils.local(*kin_inputs)
        global_drift = None
        if include_global_data:
            sim_world_hip_pos = sim_inputs[0][:, 0, :] # sim_inputs[0] gives us positions, [:, 0, :] we select all batches, 0th bone (hip), all coords
            kin_world_hip_pos = kin_inputs[0][:, 0, :]
            global_drift = kin_world_hip_pos - sim_world_hip_pos
            # print(f"Sim world pos - {sim_world_hip_pos} Kin world pos - {kin_world_hip_pos} global_drift: {global_drift}")
            # pdb.set_trace()
        return [*local_kin, *local_sim], global_drift 
    
    @staticmethod
    def extract_char_state(obs: Union[torch.tensor, np.ndarray], # obs is of shape [batch_size, TOTAL_OBS_LEN] or [TOTAL_OBS_LEN]
                            idx: int, use_tensor: bool, device = None , pin_memory: bool = False) -> Tuple[CharState, int]:
        B = obs.shape[0] if len(obs.shape)  > 1 else 0
        batched = B > 0
        shape = [B, NUM_BONES] if batched else [NUM_BONES]
        if use_tensor:
            if device is None:
                device = obs.device
            zeros_func = lambda i : torch.zeros(i, device=device, pin_memory=pin_memory)
        else:
            zeros_func = lambda i : np.zeros(i)
        positions = zeros_func((*shape, 3))
        rotations = zeros_func((*shape, 4))
        velocities = zeros_func((*shape, 3))
        rot_velocities = zeros_func((*shape, 3))
        heights = zeros_func((*shape,))

        for i in range(NUM_BONES):
            SupertrackUtils.assign_next_n(positions, i, obs, idx, 3, batched)
            idx += 3
            SupertrackUtils.assign_next_n(rotations, i, obs, idx, 4, batched)
            idx += 4
            SupertrackUtils.assign_next_n(velocities, i, obs, idx, 3, batched)
            idx += 3
            SupertrackUtils.assign_next_n(rot_velocities, i, obs, idx, 3, batched)
            idx += 3
            SupertrackUtils.assign_next(heights, i, obs, idx, batched)
            idx += 1
        up_dir = obs[:,idx:idx+3] if batched else obs[idx:idx+3]
        idx += 3
        if batched:
            return_charstates = [CharState(positions[i],
                        rotations[i],
                        velocities[i],
                        rot_velocities[i], 
                        heights[i],
                        up_dir[i]) for i in range(B)]
        else:
            return_charstates = CharState(positions,
                        rotations,
                        velocities,
                        rot_velocities, 
                        heights,
                        up_dir)
        return return_charstates, idx 
    
    @staticmethod
    def assign_next_n(to, to_idx, tensor_from, from_idx, from_offset, batched):
        if batched:
            to[:, to_idx] = tensor_from[:, from_idx: from_idx + from_offset]
        else:
            to[to_idx] = tensor_from[from_idx: from_idx + from_offset]

    @staticmethod
    def assign_next(to, to_idx, tensor_from, from_idx, batched):
        if batched:
            to[:, to_idx] = tensor_from[:, from_idx]
        else:
            to[to_idx] = tensor_from[from_idx]

    @staticmethod
    def extract_pd_targets(obs: Union[torch.tensor, np.ndarray], idx, use_tensor : bool, pin_memory: bool = False, device = None) -> Tuple[PDTargets, int]:
        B = obs.shape[0] if len(obs.shape)  > 1 else 0
        batched = B > 0
        shape = [B, NUM_BONES] if batched else [NUM_BONES]
        if use_tensor:
            if device is None:
                device = obs.device
            zeros_func = lambda i : torch.zeros((*shape, i), device=device, pin_memory=pin_memory)
        else:
            zeros_func = lambda i : np.zeros((*shape, i))
        rotations = zeros_func(4)
        rot_velocities = zeros_func(3)

        for i in range(NUM_BONES):
            SupertrackUtils.assign_next_n(rotations, i, obs, idx, 4, batched)
            idx += 4
            SupertrackUtils.assign_next_n(rot_velocities, i, obs, idx, 3, batched)
            idx += 3
        if batched:
            return_targets = [PDTargets(rotations[i], rot_velocities[i]) for i in range(B)]
        else:
            return_targets = PDTargets(rotations, rot_velocities)
        return return_targets, idx 
    
    @staticmethod
    def parse_supertrack_data_field(obs: Union[torch.tensor, np.ndarray], pin_memory: bool = False, device = None, use_tensor=None):
        if use_tensor is None:
            use_tensor = torch.is_tensor(obs)
        elif use_tensor and type(obs) is np.ndarray:
            obs = torch.as_tensor(np.asanyarray(obs), device=device, dtype=torch.float32)
        B = obs.shape[0] if len(obs.shape)  > 1 else 0
        idx = 0
        sim_char_state, idx = SupertrackUtils.extract_char_state(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        # Extract kin char state
        kin_char_state, idx = SupertrackUtils.extract_char_state(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        # Extract pre_targets
        pre_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        # Extract post_targets
        post_targets, idx = SupertrackUtils.extract_pd_targets(obs, idx, use_tensor, pin_memory=pin_memory, device=device)
        # if idx != TOTAL_OBS_LEN:
        #     raise Exception(f'idx was {idx} expected {TOTAL_OBS_LEN}')
        if B > 0:
            return [SuperTrackDataField(
                sim_char_state=sim_char_state[i], 
                kin_char_state=kin_char_state[i],
                pre_targets=pre_targets[i],
                post_targets=post_targets[i]) for i in range(B)]
        return {
            (PDTargetPrefix.PRE, PDTargetSuffix.ROT) : pre_targets.rotations,
            (PDTargetPrefix.PRE, PDTargetSuffix.RVEL) : pre_targets.rot_velocities,
            (PDTargetPrefix.POST, PDTargetSuffix.ROT) : post_targets.rotations,
            (PDTargetPrefix.POST, PDTargetSuffix.RVEL) : post_targets.rot_velocities,
            **SupertrackUtils._st_charstate_keylist_helper(CharTypePrefix.KIN, kin_char_state),
            **SupertrackUtils._st_charstate_keylist_helper(CharTypePrefix.SIM, sim_char_state),
        }
           
    def _st_charstate_keylist_helper(prefix, char_state):
        attr_suffx_list = [('positions', CharTypeSuffix.POSITION), ('rotations', CharTypeSuffix.ROTATION), ('velocities', CharTypeSuffix.VEL), ('rot_velocities', CharTypeSuffix.RVEL), ('heights', CharTypeSuffix.HEIGHT), ('up_dir', CharTypeSuffix.UP_DIR)]
        return {(prefix, suffix): getattr(char_state, attr) for attr,suffix in attr_suffx_list}
    
    
    @staticmethod
    def split_world_model_output(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def local(cur_pos: torch.Tensor, # shape [..., num_t_bones, 3]
            cur_rots: torch.Tensor, # shape [..., num_t_bones, 4]
            cur_vels: torch.Tensor,   # shape [..., num_t_bones, 3]
            cur_rot_vels: torch.Tensor, # shape [..., num_t_bones, 3]
            include_quat_rots: bool = False,
            unzip_to_batchsize: bool = True,
            ): 
        """
        Take in character state in world space and convert to local space
        Removes root bone information as well

        Returns:
            tensors of shape [..., NUM_T_BONES * (3 or 4 or 6)] by default
                             if unzip_to_batchsize is false: [..., NUM_T_BONES, 3 or 4 or 6] 
        """
        # root_pos = cur_pos[..., 0:1 , :].detach().clone() # shape [..., 1, 3]
        # inv_root_rots = pyt.quaternion_invert(cur_rots[..., 0:1, :].detach().clone()) # shape [..., 1, 4]
        root_pos = cur_pos[..., 0:1 , :].clone() # shape [..., 1, 3]
        inv_root_rots = pyt.quaternion_invert(cur_rots[..., 0:1, :]) # shape [..., 1, 4]
        inv_root_rots = SupertrackUtils.normalize_quat(inv_root_rots)
        local_pos = pyt.quaternion_apply(inv_root_rots, cur_pos - root_pos) # shape [..., num_t_bones, 3]
        B = cur_pos.shape[:-2]
        up_dir = torch.zeros(*B, 3, device=cur_rots.device)
        # Set the Y component to 1
        up_dir[..., 1] = 1.0
        local_up_dir = pyt.quaternion_apply(inv_root_rots.squeeze(), up_dir)
        # Have to clone quat rots to avoid 
        # RuntimeError: Output 0 of UnbindBackward0 is a view and its base or another view of its base has been modified inplace. 
        # This view is the output of a function that returns multiple views. Such functions do not allow the output views to be
        # modified inplace. You should replace the inplace operation by an out-of-place one
        local_rots_quat = torch.empty_like(cur_rots)
        local_rots_quat[...,  0, :] = cur_rots[..., 0, :].clone()
        local_rots_quat[..., 1:, :] = pyt.quaternion_multiply(inv_root_rots, cur_rots[..., 1:, :].clone()) # shape [..., num_t_bones, 4]
        # local_rots_quat = cur_rots

        local_rots_6d = pyt.matrix_to_rotation_6d(pyt.quaternion_to_matrix(local_rots_quat)) # shape [..., 6]

        local_vels = pyt.quaternion_apply(inv_root_rots, cur_vels) # shape [..., num_t_bones, 3]
        local_rot_vels = pyt.quaternion_apply(inv_root_rots, cur_rot_vels) # shape [..., num_t_bones, 3]

        return_tensors = [local_pos, local_rots_6d, local_vels, local_rot_vels, local_up_dir]
        if include_quat_rots:
            return_tensors.append(local_rots_quat)
        # return_tensors = [(local_pos, 'local_pos'), (return_rots, 'return_rots'), (local_vels, 'local_vels'), (local_rot_vels, 'local_rot_vels'), (cur_heights[:, 1:], 'cur_heights'), (cur_up_dir, 'cur_up_dir')]
        # Have to reshape instead of view because stride can be messed up in some cases
        if unzip_to_batchsize:
            batch_dim = cur_pos.shape[:-2]
            return [t.reshape(*batch_dim, -1) for t in return_tensors]
        else:
            return return_tensors
        

    def integrate_through_world_model(world_model: torch.nn.Module, dtime : float, 
                                       pos: torch.Tensor, # shape [batch_size, num_t_bones, 3]
                                    rots: torch.Tensor, # shape [batch_size, num_t_bones, 4]
                                    vels: torch.Tensor,  # shape [batch_size, num_t_bones, 3]
                                    rvels: torch.Tensor, # shape [batch_size,  num_t_bones, 3]
                                    # heights: torch.Tensor, # shape [batch_size, num_t_bones, 3]
                                    # up_dir: torch.Tensor,  # shape [batch_size, 3]
                                    kin_rot_t: torch.Tensor, # shape [batch_size, num_t_bones, 6] num_t_bones = 16 
                                    kin_rvel_t: torch.Tensor, # shape [batch_size, num_t_bones, 3]
                                    local_tensors : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None,
                                    update_normalizer: bool = False,
                                    skip_dtime_scale: bool = False,
                                    debug: bool = False,
                                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Integrate a character state through the world model
        Params should be in world space and will be returned in world space.
        :param exclude_root: Whether to exclude the root bone from the output, useful for training the policy since we 
        don't want to compute loss with root bone
        """
        batch_size = pos.shape[0]
        # root_rot = SupertrackUtils.normalize_quat(rots[:, 0:1, :].detach().clone()) 
        root_rot = SupertrackUtils.normalize_quat(rots[:, 0:1, :].clone()) # To detach or not...  
        # If you detach, you're basically saying "Ok don't worry about outputting correct hip rot too much, because
        # even if it affects the next frame's local velocities and such it won't backprop gradients to correct them"
        if local_tensors is None:
            local_tensors = SupertrackUtils.local(pos, rots, vels, rvels)
            
        input = (*local_tensors,
                    kin_rot_t.reshape(batch_size, -1),
                    kin_rvel_t.reshape(batch_size, -1))
        output = world_model(*input, update_normalizer=update_normalizer)
        local_accel, local_rot_accel = SupertrackUtils.split_world_model_output(output)
        # Convert to world space
        accel = pyt.quaternion_apply(root_rot, local_accel) 
        rot_accel = pyt.quaternion_apply(root_rot, local_rot_accel)
        # print(f"Local_accel: {local_accel} \n Local rot accel: {local_rot_accel}")
        # padding_for_root_bone = torch.zeros((batch_size, 1, 3))
        # accel = torch.cat((padding_for_root_bone, accel), dim=1)
        # rot_accel = torch.cat((padding_for_root_bone, rot_accel), dim=1)
        # Integrate using Semi-Implicit Euler
        # We use semi-implicit so the model can influence position and velocity losses for the first timestep
        # Also that's what the paper does
        next_vels = vels  + accel* (1 if skip_dtime_scale else dtime)
        next_rvels = rvels  + rot_accel* (1 if skip_dtime_scale else dtime)
        next_pos = pos + next_vels*dtime
        # Don't need to standardize because pyt.quaternion_multiply does by default 
        next_rots = pyt.quaternion_multiply(pyt.axis_angle_to_quaternion(next_rvels*dtime) , rots.clone())
        return next_pos, next_rots, next_vels, next_rvels
    

    def char_state_loss(pos1, pos2, rot1, rot2, vel1, vel2, rvel1, rvel2, unbatched : bool = False):
        # Input shapes: [batch_size, window_size, NUM_T_BONES, 3 or 4]

        def l1_norm(a, b, c = None):
            diff = c if c is not None else a - b # shape: [batch_size, window_size, num_bones, 3]
            if unbatched: # for debug and testing purposes
                return diff.abs().sum() # Do not need to take average over batch size
            return diff.abs().sum(dim=(1,2,3)).mean()
            # return torch.mean(torch.sum(diff, dim=(1,2,3)))
        
        raw_pos_l = l1_norm(pos1, pos2) #torch.mean(torch.sum(torch.abs(pos1-pos2), dim =(1,2,3)))
        raw_vel_l = l1_norm(vel1, vel2) 
        raw_rvel_l = l1_norm(rvel1, rvel2)
        if rot1.shape[-1] != 4: # rots are in quat form
            raise Exception(f"Rots in unexpected shape: {rot1.shape}")

        # From Stack Overflow:
        # If you want to find a quaternion diff such that diff * q1 == q2, then you need to use the multiplicative inverse:
        # diff * q1 = q2  --->  diff = q2 * inverse(q1)
        # https://stackoverflow.com/questions/21513637/dot-product-of-two-quaternion-rotations
        # From paper: "⊖ represents quaternion difference - i.e. taking the log of the
        # result of the left-hand-side quaternion multiplied by the inverse of
        # the right-hand-side quaternion" - their left hand side was ground truth
        quat_diffs = pyt.quaternion_multiply(SupertrackUtils.normalize_quat(rot1) , pyt.quaternion_invert(SupertrackUtils.normalize_quat(rot2)))
        # quat_diffs = pyt.quaternion_multiply(SupertrackUtils.normalize_quat(rot2) , pyt.quaternion_invert(SupertrackUtils.normalize_quat(rot1) ))
        # vec_part = quat_diffs[..., 1:] # The magnitude of the vec part of a quaternion equals sin(angle/2) where angle is the angle of the quat
        norms = torch.norm(quat_diffs[..., 1:], p=2, dim=-1, keepdim=True) # The magnitude of the vec part of a quaternion equals sin(angle/2) where angle is the angle of the quat
        # scalar_part = quat_diffs[...,:1] # The real/scalar part of a quaternion equals cos(angle/2) where angle is the angle of the quat
        # We're basically breaking the quaternion down into the sin and cos values of its angle, and 
        # atan2() is a function that, given a cos and sin value for an angle, returns the angle between it and the unit vector (1, 0)
        halfangles = torch.atan2(norms, quat_diffs[..., :1])
        quat_logs = halfangles * (quat_diffs[..., 1:] / norms)
        raw_rot_l = l1_norm(None, None, c=quat_logs) #quat_logs.abs().sum(dim=(1,2,3)).mean()
        if unbatched:
            pp = lambda t : '[' + ', '.join(f"{x:.4f}" for x in t.cpu().numpy()) + ',]'
            # print(rvel1 - rvel2)
            for idx, bone in enumerate(BONENAMES):
                # pdb.set_trace()
                # print(f"{bone[5:]} \t kin: {rvel1[idx]} \t sim: {rvel2[idx]} loss: {torch.abs(rvel1[idx] - rvel2[idx]).sum()}")
                padding = '\t\t' if idx < 4 else ' \t'
                bone_loss = torch.abs(rvel1[idx] - rvel2[idx]).sum()
                print(f"{bone[5:]}{padding} kin: {pp(rvel1[idx])} {torch.norm(rvel1[idx]).item():.4f} \t sim: {pp(rvel2[idx])} {torch.norm(rvel2[idx]).item():.4f} loss: {bone_loss.item():.4f}")


            # pdb.set_trace()
            # print(quat_logs)
            # print(f"LeftForearm rot loss: {quat_logs[-2].abs().sum()}")
            # print(f"RightForearm rot loss: {quat_logs[-1].abs().sum()}")

        # batch_size, window_size, num_bones, num_entries = rot1.shape
        # quat_diffs = pyt.quaternion_multiply(rot2, pyt.quaternion_invert(rot1))
        # quat_logs = pyt.so3_log_map(pyt.quaternion_to_matrix(quat_diffs).reshape(-1, 3, 3)).reshape(batch_size, window_size, num_bones, 3)
        # raw_rot_l = l1_norm(None, None, c=quat_logs.abs())

        # d = torch.abs(torch.sum(SupertrackUtils.normalize_quat(rot1) * SupertrackUtils.normalize_quat(rot2), dim=-1))
        # d = torch.clamp(d, min=-1.0, max=1.0)
        # theta = 2 * torch.acos(d)
        # raw_rot_l = theta.sum(dim=(1,2)).mean()

        return raw_pos_l, raw_rot_l, raw_vel_l, raw_rvel_l
    

    def debug_loss_at_frame(st_data : SuperTrackDataField):
        sim_state = st_data.sim_char_state
        kin_state = st_data.kin_char_state
        remove_root_bone = lambda x: x[1:, :]
        k_pos, k_rot, k_vel, k_rvel = [remove_root_bone(k) for k in kin_state.as_tensors()]
        s_pos, s_rot, s_vel, s_rvel = [remove_root_bone(s) for s in sim_state.as_tensors()]
        #remove root bones 
        # pdb.set_trace()
        torch.set_printoptions(profile="short")

        pos_loss, rot_loss, vel_loss, rvel_loss  = SupertrackUtils.char_state_loss(k_pos, s_pos, k_rot, s_rot, k_vel, s_vel, k_rvel, s_rvel, unbatched = True)
        print(f"""\tPos_L : {pos_loss}
        Rot_L : {rot_loss}
        Vel_L : {vel_loss}
        Rvel_L : {rvel_loss}
        ====""")

    def apply_policy_action_to_pd_targets(pd_targets: torch.Tensor, # shape [NUM_T_BONES, 4]
                                           policy_action: torch.Tensor, # shape [48]
                                            ):
        policy_action = policy_action.reshape(NUM_T_BONES, 3) 
        offset_as_quat = pyt.axis_angle_to_quaternion(policy_action)
        kin_targets = SupertrackUtils.normalize_quat(pyt.quaternion_multiply(offset_as_quat, pd_targets))
        return kin_targets
    

BONENAMES = [
    "Bone_Hips", "Bone_Spine", "Bone_Spine1", "Bone_Spine2",
    "Bone_LeftUpLeg", "Bone_RightUpLeg", "Bone_LeftLeg", "Bone_RightLeg",
    "Bone_LeftFoot", "Bone_RightFoot", "Bone_LeftShoulder", "Bone_RightShoulder",
    "Bone_LeftArm", "Bone_RightArm", "Bone_LeftForeArm", "Bone_RightForeArm"
]