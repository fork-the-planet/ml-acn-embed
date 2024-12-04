"""
System(or organization)-dependent entry points with generic implementations
"""

import logging
import os

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def can_save_checkpoint():
    return is_local_leader()


# pylint: disable=unused-argument
def dist_barrier(timeout_seconds: int):
    # We ignore timeout_seconds because monitored_barrier() is currently not supported for NCCL
    torch.distributed.barrier()


def dist_gather(dtype, x):
    if isinstance(x, torch.Tensor):
        x = x.cuda()
    else:
        x = torch.tensor(x, dtype=dtype).cuda()
    _world_size = torch.distributed.get_world_size()
    if _world_size == 0:
        return [x]
    tensor_list = [x.clone().detach().cuda() for _ in range(_world_size)]
    torch.distributed.all_gather(tensor_list, x)
    return tensor_list


def dist_get_cuda_num():
    return int(os.getenv("LOCAL_RANK", "0"))


def dist_get_summed(*args):
    newvars = []
    for var in args:
        if isinstance(var, torch.Tensor):
            var = var.cuda()
        else:
            var = torch.tensor(var).cuda()
        torch.distributed.all_reduce(var, op=torch.distributed.ReduceOp.SUM)
        newvars.append(var)
    return newvars


def dist_init():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def dist_is_initialized():
    return torch.distributed.is_initialized()


def dist_shutdown():
    torch.distributed.destroy_process_group()


def get_dist_model(model, optimizer):
    ddp_model = DistributedDataParallel(model)
    return ddp_model, ddp_model.module, optimizer


def get_rank():
    return torch.distributed.get_rank()


def infinite_dataloader(dataloader):
    while True:
        yield from dataloader


def is_leader():
    return int(os.getenv("RANK", "0")) == 0


def is_local_leader():
    return int(os.getenv("LOCAL_RANK", "0")) == 0


def log_leader(level, *msg):
    """
    Log a message if in a rank 0 distributed process
    """
    if is_leader():
        LOGGER.log(getattr(logging, level.upper()), *msg)


def log_leader_metrics(metrics: dict, epoch: int = None):
    """
    Send status to a system-dependent metrics tracking system if the leader
    """
    # pylint: disable=unused-argument
    return


def log_local_leader(level, *msg):
    """
    Log a message if in a "local" rank 0 distributed process.
    By default, is the same as log_global_once
    """
    if is_local_leader():
        LOGGER.log(getattr(logging, level.upper()), *msg)


def sync_checkpoint():
    """
    Synchronization function to be called after writing a checkpoint file.
    """
    return


def world_size():
    return torch.distributed.get_world_size()
