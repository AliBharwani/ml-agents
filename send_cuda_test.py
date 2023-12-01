from tabnanny import check
import time
import torch

from mlagents.simple_queue_with_size import SimpleQueueWithSize
from mlagents.trainers.torch_entities.utils import ModelUtils


# def test_producer_func(queue : torch.multiprocessing.SimpleQueue):
#     torch.set_default_device("cuda")
#     print(f"Test process started")
#     try:
#         while True:
#             time.sleep(5)
#             dummy_t = torch.ones(1, 1,1,30)
#             non_ones = torch.sum(dummy_t == 1)
#             queue.put(dummy_t)
#             print(f"===== TEST PROCESS PUT ::: {dummy_t.shape} with num non ones: {torch.sum(dummy_t == 1)} non ones before: {non_ones} ")
#     except Exception as e:
#         print(f"Exception in test process: {e}")


# def test_trainer_func(trainer):
#     torch.set_default_device("cuda")
#     queue = torch.multiprocessing.SimpleQueue()
#     test_process = torch.multiprocessing.Process(target=test_producer_func, args=(queue,))
#     test_process.start()
#     # trainer._initialize()
#     while True:
#         item = queue.get()
#         print(f"-------->>> MAIN PROCESS GOT TENSOR num non ones::: {torch.sum(item == 1)}")
#         # print(f"-------->>> MAIN PROCESS GOT::: {self.queue.get()}")
#         time.sleep(.001)

#     #     trainer.advance_consumer()


# class Trainer:
#     def _initialize(self):
#         self.queue = torch.multiprocessing.SimpleQueue()
#         test_process = torch.multiprocessing.Process(target=test_producer_func, args=(self.queue,))
#         test_process.start()

#     def advance_consumer(self):
#         if not self.queue.empty():
#             item = self.queue.get()
#             print(f"-------->>> MAIN PROCESS GOT TENSOR num non ones::: {torch.sum(item == 1)}")
#             # print(f"-------->>> MAIN PROCESS GOT::: {self.queue.get()}")
#         else:
#             time.sleep(.001)



# if __name__ == '__main__':
#     # Create a child SimpleQueue and a child process
#     torch.set_default_device("cuda")
#     # trainer = Trainer()
#     # trainer_process = torch.multiprocessing.Process(target=test_trainer_func, args=(trainer,))
#     # trainer_process.start()
#     queue = torch.multiprocessing.SimpleQueue()
#     test_process = torch.multiprocessing.Process(target=test_producer_func, args=(queue,))
#     test_process.start()
#     # trainer._initialize()
#     while True:
#         item = queue.get()
#         print(f"-------->>> MAIN PROCESS GOT TENSOR num non ones::: {torch.sum(item == 1)}")
#         # print(f"-------->>> MAIN PROCESS GOT::: {self.queue.get()}")
#         time.sleep(.001)


# def test_producer_func(queue : torch.multiprocessing.SimpleQueue):
#     torch.set_default_device("cuda")
#     torch.cuda.init()
#     while True:
#         dummy_t = torch.ones(1,30)
#         torch.cuda.synchronize()
#         print(f"===== TEST PROCESS PUT TENSOR with num ones: {torch.sum(dummy_t == 1)} tensor: {dummy_t}")
#         queue.put(dummy_t)
#         time.sleep(3)


# if __name__ == '__main__':
#     torch.set_default_device("cuda")
#     torch.cuda.init()
#     queue = torch.multiprocessing.SimpleQueue()
#     producer_process = torch.multiprocessing.Process(target=test_producer_func, args=(queue,))
#     producer_process.start()
#     while True:
#         if not queue.empty():
#             item = queue.get()
#             print(f"-------->>> MAIN PROCESS GOT TENSOR num ones::: {torch.sum(item == 1)} item: {item}")
#         else:
#             time.sleep(.001)


def test_producer_func(queue : torch.multiprocessing.SimpleQueue):
    torch.set_default_device("cuda")
    while True:
        if not queue.empty():
            item = queue.get()
            print(f"-------->>> test_producer_func TENSOR num ones::: {torch.sum(item == 1)}")
        else:
            time.sleep(.001)
        # dummy_t = torch.ones(1, 1,1,30)
        # non_ones = torch.sum(dummy_t == 1)
        # queue.put(dummy_t)
        # print(f"===== TEST PROCESS PUT ::: {dummy_t.shape} with num non ones: {torch.sum(dummy_t == 1)} non ones before: {non_ones} ")
        # time.sleep(3)


if __name__ == '__main__':
    torch.set_default_device("cuda")
    queue = torch.multiprocessing.SimpleQueue()
    producer_process = torch.multiprocessing.Process(target=test_producer_func, args=(queue,))
    producer_process.start()
    while True:
        # if not queue.empty():
        #     item = queue.get()
        #     print(f"-------->>> MAIN PROCESS GOT TENSOR num non ones::: {torch.sum(item == 1)}")
        # else:
        #     time.sleep(.001)
        dummy_t = torch.ones(1, 1,1,30)
        non_ones = torch.sum(dummy_t == 1)
        queue.put(dummy_t)
        print(f"===== __main__ PROCESS PUT ::: {dummy_t.shape} with num  ones: {torch.sum(dummy_t == 1)}  ones before: {non_ones} ")
        time.sleep(3)

