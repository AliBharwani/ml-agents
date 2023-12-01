import time
from mlagents.torch_utils import torch



class SimpleQueueWithSize:
    def __init__(self, name : str = "", maxsize : int = 0): 
        self._queue = torch.multiprocessing.SimpleQueue()
        self._size = torch.multiprocessing.Value('i', 0)
        self._maxsize = maxsize
        self._name = name

    def qsize(self): 
        return self._size.value
    
    def empty(self):
        return self._queue.empty()
    
    def get(self):
        # if self._name is "trajectory_queue":
        #     print(f"Getting from trajectory queue, size: {self._size.value}")
        try:
            item = self._queue.get()
            self._size.value -= 1
            return item
        except Exception as e:
            print(f"Failed to get from queue: {e}")
            raise e
        
    def put(self, item):
        # if self._name is "trajectory_queue":
        #     print(f"Putting to trajectory queue, size: {self._size.value}")
        try:
            # If at maxsize, block until we can put
            while self._maxsize > 0 and self._size.value >= self._maxsize:
                # Block and do not busy wait
                print(f"Queue {self._name} is full, waiting to put...")
                time.sleep(.001)
            self._queue.put(item)
            self._size.value += 1
        except Exception as e:
            print(f"Failed to put to queue: {e}")
            raise e
        