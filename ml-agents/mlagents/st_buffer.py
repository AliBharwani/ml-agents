from collections import defaultdict
from collections.abc import MutableMapping
import enum
import functools
import itertools
import pdb
from typing import BinaryIO, DefaultDict, List, Tuple, Union
from mlagents.torch_utils import torch
from mlagents.trainers.supertrack.supertrack_utils import MINIMUM_TRAJ_LEN, NUM_BONES, TOTAL_OBS_LEN, SupertrackUtils, CharTypePrefix, CharTypeSuffix, PDTargetPrefix, PDTargetSuffix
from mlagents.trainers.trajectory import Trajectory

import h5py

from mlagents.torch_utils import torch, default_device

class BufferKey(enum.Enum):
    IDX_IN_TRAJ = "idx_in_traj"
    TRAJ_LEN = "traj_len"

class SuffixToNumValues:
    SUFFIX_TO_NUM_VAL = {
        PDTargetSuffix.ROT : 4,
        PDTargetSuffix.RVEL: 3,
        CharTypeSuffix.POSITION : 3,
        CharTypeSuffix.ROTATION : 4,
        CharTypeSuffix.VEL : 3,               
        CharTypeSuffix.RVEL : 3,
        CharTypeSuffix.HEIGHT : 1,
        # These should not be accessed from the table
        # CharTypeSuffix.UP_DIR : 3, 
    }


STBufferKey = Union[
    BufferKey, Tuple[PDTargetPrefix, PDTargetSuffix], Tuple[CharTypePrefix, CharTypeSuffix]
]

STBufferField = torch.tensor # Union[torch.tensor, list]

class STBuffer(MutableMapping):
    """
    STBuffer is a buffer created for SuperTrack for compatibility with AgentBuffer methods
    """

    # Whether or not to validate the types of keys at runtime
    # This should be off for training, but enabled for testing
    CHECK_KEY_TYPES_AT_RUNTIME = False

    def __init__(self, buffer_size = None):
        self._fields: DefaultDict[STBufferKey, STBufferField] = defaultdict(
            list
        )
        if buffer_size is None:
            return
        for key in STBuffer.get_all_possible_keys():
            if isinstance(key, tuple): # We only tuple data in the batch
                self[key] = torch.empty(buffer_size, *STBuffer.suffix_to_tensor_shape(key[1]))
                # print(f"Assinging {key[0].value} , {key[1].value} the following shape: {(buffer_size, *STBuffer.suffix_to_tensor_shape(key[1]))}")
            else: 
                self[key] = torch.zeros(buffer_size, dtype=torch.int32)
        # Num values in the buffer
        self._cur_idx = 0
        self._buffer_size = buffer_size
        self.hole = None


    def __str__(self):
        return ", ".join([f"'{k}' : {str(self[k])}" for k in self._fields.keys()])


    @staticmethod
    def _check_key(key):
        if isinstance(key, BufferKey):
            return
        if isinstance(key, tuple):
            key0, key1 = key
            if isinstance(key0, PDTargetPrefix):
                if isinstance(key1, PDTargetSuffix):
                    return
                raise KeyError(f"{key} has type ({type(key0)}, {type(key1)})")
            if isinstance(key0, CharTypePrefix):
                if isinstance(key1, CharTypeSuffix):
                    return
                raise KeyError(f"{key} has type ({type(key0)}, {type(key1)})")
        raise KeyError(f"{key} is a {type(key)}")

    @staticmethod
    def _encode_key(key: STBufferKey) -> str:
        """
        Convert the key to a string representation so that it can be used for serialization.
        """
        if isinstance(key, BufferKey):
            return key.value
        prefix, suffix = key
        return f"{prefix.value}:{suffix.value}"

    @staticmethod
    def _decode_key(encoded_key: str) -> STBufferKey:
        """
        Convert the string representation back to a key after serialization.
        """
        # Simple case: convert the string directly to a BufferKey
        try:
            return BufferKey(encoded_key)
        except ValueError:
            pass

        # Not a simple key, so split into two parts
        prefix_str, _, suffix_str = encoded_key.partition(":")

        # See if it's an PDTargetPrefix first
        try:
            return PDTargetPrefix(prefix_str), PDTargetSuffix(suffix_str)
        except ValueError:
            pass

        try:
            return CharTypePrefix(prefix_str), CharTypeSuffix(suffix_str)
        except ValueError:
            raise ValueError(f"Unable to convert {encoded_key} to an STBufferKey")
        

    def __getitem__(self, key: STBufferKey) -> STBufferField:
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        return self._fields[key]

    def __setitem__(self, key: STBufferKey, value: list) -> None:
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        self._fields[key] = value

    def __delitem__(self, key: STBufferKey) -> None:
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        self._fields.__delitem__(key)

    def __iter__(self):
        return self._fields.__iter__()

    def __len__(self) -> int:
        return self._fields.__len__()

    def __contains__(self, key):
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            self._check_key(key)
        return self._fields.__contains__(key)

    def check_length(self, key_list: List[STBufferKey]) -> bool:
        """
        Some methods will require that some fields have the same length.
        check_length will return true if the fields in key_list
        have the same length.
        :param key_list: The fields which length will be compared
        """
        if self.CHECK_KEY_TYPES_AT_RUNTIME:
            for k in key_list:
                self._check_key(k)

        if len(key_list) < 2:
            return True
        length = None
        for key in key_list:
            if key not in self._fields:
                return False
            if (length is not None) and (length != len(self[key])):
                return False
            length = len(self[key])
        return True


    def sample_mini_batch(self, batch_size: int, raw_window_size: int, key_list: List[STBufferKey] = None, normalize_quat = True) -> "STBuffer":
        """
        Creates a mini-batch
        """
        if key_list is None:
            key_list = [key for key in STBuffer.get_all_possible_keys() if isinstance(key, tuple)]
        # We need to add 1 when sampling because first data point is only used for setting initial state, 
        # not prediction / loss computation
        window_size = raw_window_size + 1
        mini_batch = STBuffer()
        for key in key_list:
            mini_batch[key] = torch.empty(batch_size, window_size, *STBuffer.suffix_to_tensor_shape(key[1]))
        buff_len = self.num_experiences
        # Subtract window_size from buff_len because we want to make sure there are enough entries for a full window
        # after the last start idx 
        high = (buff_len - window_size) // window_size
        start_idxes = torch.randint(0, high, (batch_size,)) * window_size
        # start_idxes = torch.from_numpy(DEBUG_start_idxes).to(device=default_device())
        # Check if any of the start_idxes falls within the hole
        if self.hole is not None:
            # If any start_idx falls within the hole, have it use the the traj immediately behind it
            # Create a boolean mask where True indicates that the condition is met
            mask = (self.hole[0] <= start_idxes) & (start_idxes <= self.hole[1])
            # This puts at immediately at the beginning of the trajectory where the hole is
            start_idxes[mask] -= self[BufferKey.IDX_IN_TRAJ][start_idxes[mask]]
            # This puts it at the end of the previous trajectory 
            start_idxes[mask] -= 1

        # Make sure for every start idx, there are enough values after this to create a full window 
        # without going into the next trajectory
        num_steps_remaning = self[BufferKey.TRAJ_LEN][start_idxes] - self[BufferKey.IDX_IN_TRAJ][start_idxes]
        num_steps_to_rewind = window_size - num_steps_remaning
        # Ensure it's not negative. Could also subtract with a mask instead, same thing
        num_steps_to_rewind = torch.clamp(num_steps_to_rewind, min=0)  
        start_idxes -= num_steps_to_rewind
        # print(f"start_idxes dtype: {start_idxes.dtype} self[BufferKey.TRAJ_LEN] dtype: {self[BufferKey.TRAJ_LEN].dtype} ")
        # print(f"num_steps_remaning dtype: {num_steps_remaning.dtype} num_steps_to_rewind dtype: {num_steps_to_rewind.dtype} ")
        window_range = torch.arange(window_size)  # Shape: (window_size,)
        # Expand and offset the range for each start index
        # Reshape start_idxes to (batch_size, 1) and add window_range (broadcasting)
        indices = start_idxes.unsqueeze(1) + window_range  # Shape: (batch_size, window_size)
        for key in key_list:
            # Use advanced indexing to extract slices
            # We want to gather along the first dimension of self[key]
            mini_batch[key] = self[key][indices]  # Shape: (batch_size, window_size, 17, 3)
            if normalize_quat and ((key[0] == PDTargetPrefix.POST or key[0] == PDTargetPrefix.PRE) and key[1] == PDTargetSuffix.ROT):
                mini_batch[key] = SupertrackUtils.normalize_quat(self[key][indices])
            else:
                mini_batch[key] = self[key][indices] 

        return mini_batch
    
    @staticmethod
    @functools.cache
    def get_all_possible_keys():
        return [*BufferKey,*itertools.product(PDTargetPrefix, PDTargetSuffix), *itertools.product(CharTypePrefix, CharTypeSuffix)]
    
    @staticmethod
    @functools.cache
    def suffix_to_tensor_shape(suffix):
        if suffix == CharTypeSuffix.HEIGHT:
            return [NUM_BONES]
        if suffix == CharTypeSuffix.UP_DIR:
            return [3]
        return [NUM_BONES, SuffixToNumValues.SUFFIX_TO_NUM_VAL[suffix]]
    

    def save_to_file(self, file_object: BinaryIO) -> None:
        """
        Saves the AgentBuffer to a file-like object.
        """
        with h5py.File(file_object, "w") as write_file:
            for key, data in self.items():
                write_file.create_dataset(
                    self._encode_key(key), data=data, dtype="f", compression="gzip"
                )

    def load_from_file(self, file_object: BinaryIO) -> None:
        """
        Loads the AgentBuffer from a file-like object.
        """
        with h5py.File(file_object, "r") as read_file:
            for key in list(read_file.keys()):
                decoded_key = self._decode_key(key)
                self[decoded_key] = STBufferField()
                # extend() will convert the numpy array's first dimension into list
                self[decoded_key].extend(read_file[key][()])

    def add_supertrack_data(
        self,
        trajectory : Trajectory,
    ) -> None:
        """
        Appends this AgentBuffer to target_buffer
        :param target_buffer: The buffer which to append the samples to.
        """
        obs = trajectory.steps[0].obs
        if len(obs) != 1:
            raise Exception("Attempting to add trajectory with more than one observation to SuperTrack buffer", len(obs), obs)
        traj_len = len(trajectory.steps)
        for step, exp in enumerate(trajectory.steps):
            self[BufferKey.IDX_IN_TRAJ][self.effective_idx] = step
            self[BufferKey.TRAJ_LEN][self.effective_idx] = traj_len
            obs = exp.obs[0]
            if (len(obs) != TOTAL_OBS_LEN):
                raise Exception(f'Obs was of len {len(obs)} expected {TOTAL_OBS_LEN}')
            st_keylist = SupertrackUtils.parse_supertrack_data_field(obs, device=default_device(), use_tensor=True, return_as_keylist=True)
            for key, value in st_keylist.items():
                self[key][self.effective_idx] = value
            self._cur_idx += 1
        # If we're in the stage of overwriting trajectories 
        if self._cur_idx > self._buffer_size: 
            # Check if the trajectory directly after the last one we added is still usuable 
            num_steps_in_next_traj = self[BufferKey.TRAJ_LEN][self.effective_idx] - self[BufferKey.IDX_IN_TRAJ][self.effective_idx]
            if num_steps_in_next_traj < MINIMUM_TRAJ_LEN:
                end_idx = min(self._buffer_size, self.effective_idx + num_steps_in_next_traj)
                self.hole = [self.effective_idx, end_idx]
            else: 
                self.hole = None

    @property
    def effective_idx(self):
        return self._cur_idx % self._buffer_size

    @property
    def num_experiences(self) -> int:
        """
        The number of agent experiences in the STBuffer, i.e. the length of the buffer.
        """
        # We let _cur_idx roll over and mod it by buffer size when indexing
        return min(self._buffer_size, self._cur_idx)