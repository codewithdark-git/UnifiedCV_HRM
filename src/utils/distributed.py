"""
Distributed Training Utilities

DDP, FSDP, and DeepSpeed setup helpers.
"""

import os
import contextlib
import torch
import torch.distributed as dist
from typing import Optional, Any, List


def setup_distributed(backend: str = "nccl") -> bool:
    """
    Initialize distributed process group.

    Args:
        backend: "nccl" | "gloo" | "mpi"

    Returns:
        True if distributed initialized
    """
    # Check if already initialized
    if dist.is_available() and dist.is_initialized():
        return True

    # Get rank/world_size from environment (torchrun, SLURM, etc.)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=os.environ.get("MASTER_ADDR", "tcp://localhost:29500"),
            world_size=world_size,
            rank=rank,
        )

        # Set device
        torch.cuda.set_device(local_rank)

        print(f"Distributed initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return True

    return False


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get local rank (GPU index)."""
    if dist.is_available() and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0


def is_distributed() -> bool:
    """Check if distributed training is active."""
    return dist.is_available() and dist.is_initialized()


# ─────────────────────────────────────────────────────────────────
# Tensor Operations
# ─────────────────────────────────────────────────────────────────

def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """
    Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: "mean" | "sum" | "min" | "max"

    Returns:
        Reduced tensor (on rank 0, empty on others)
    """
    if not is_distributed():
        return tensor

    rt = tensor.clone()
    dist_op = getattr(dist.ReduceOp, op.upper())
    dist.all_reduce(rt, op=dist_op)

    if op == "mean":
        rt /= get_world_size()

    return rt


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """All-reduce tensor across all processes."""
    if not is_distributed():
        return tensor

    rt = tensor.clone()
    dist_op = getattr(dist.ReduceOp, op.upper())
    dist.all_reduce(rt, op=dist_op)
    return rt


def gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensor from all processes.

    Returns:
        List of tensors from all ranks
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return gather_list


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast object from source rank to all."""
    if not is_distributed():
        return obj

    obj_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def synchronize():
    """Barrier synchronization."""
    if is_distributed():
        dist.barrier()


# ─────────────────────────────────────────────────────────────────
# Model Wrapping
# ─────────────────────────────────────────────────────────────────

def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: Optional[List[int]] = None,
    find_unused: bool = False,
) -> torch.nn.Module:
    """Wrap model with DDP."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = model.cuda()
    return DDP(
        model,
        device_ids=device_ids or [get_local_rank()],
        find_unused_parameters=find_unused,
    )


def wrap_model_fsdp(
    model: torch.nn.Module,
    sharding_strategy: str = "FULL_SHARD",
) -> torch.nn.Module:
    """Wrap model with FSDP (experimental)."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp import ShardingStrategy

    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }

    model = model.cuda()
    return FSDP(
        model,
        sharding_strategy=strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD),
        auto_wrap_policy=transformer_auto_wrap_policy,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/FSDP model."""
    if hasattr(model, "module"):
        return model.module
    return model


# ─────────────────────────────────────────────────────────────────
# Distributed Sampler
# ─────────────────────────────────────────────────────────────────

def get_distributed_sampler(dataset, shuffle: bool = True, drop_last: bool = True):
    """Get distributed sampler for dataset."""
    if is_distributed():
        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return None


# ─────────────────────────────────────────────────────────────────
# Gradient Sync (for gradient accumulation in DDP)
# ─────────────────────────────────────────────────────────────────

def sync_gradients(model: torch.nn.Module):
    """Synchronize gradients across DDP processes."""
    if is_distributed():
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= get_world_size()


# ─────────────────────────────────────────────────────────────────
# Context Managers
# ─────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def no_sync(model: torch.nn.Module):
    """Context manager to disable gradient synchronization."""
    if hasattr(model, "no_sync"):
        with model.no_sync():
            yield
    else:
        yield


import contextlib