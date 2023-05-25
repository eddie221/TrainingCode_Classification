import torch
from contextlib import contextmanager

@contextmanager
def main_process_first(local_rank):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def info_log(message, rank = -1, type = ["std"]):
    if rank in [-1, 0]:
        if "std" in type:
            print(message)
        if "log" in type:
            logging.info(message)