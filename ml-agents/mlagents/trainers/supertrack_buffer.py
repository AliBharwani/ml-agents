
from typing import BinaryIO, DefaultDict, List, Tuple, Union, Optional
from mlagents.trainers.supertrack.supertrack_utils import Quat

import numpy as np
import h5py
from dataclasses import dataclass
from enum import Enum

from mlagents_envs.exception import UnityException

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
    hieghts : List[float]
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

    