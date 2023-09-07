
import numpy as np

from mlagents.trainers.supertrack.supertrack_utils import Quat

class MathUtils(): 

    @staticmethod
    def localize(pos: np.ndarray, root_pos: np.ndarray, root_rot: Quat) -> np.ndarray:
        return MathUtils.quat_inv_mul_vec3(root_rot, pos - root_pos)
    
    @staticmethod
    def quat_inv_mul_vec3(q: Quat, vec: np.ndarray) -> np.ndarray: 
        return MathUtils.quat_mul_vec3(q.inv(), vec)
    
    @staticmethod
    def quat_mul_vec3(q: Quat, vec: np.ndarray) -> np.ndarray: 
        t = 2 * np.cross(q.vec(), vec)
        return vec + q.w * t + np.cross(q.vec(), t)
    



