"""Fixed rank-order FP32 gradient averaging with bounded temporary CUDA storage."""
import torch
import torch.distributed as dist

CHUNK_ELEMENTS = 8 * 1024 * 1024  # 32MiB FP32 per rank, independent of embedding size.


def rank_order_average(process_group, bucket):
    """Gather exact chunks, sum rank0..N-1; return only the original bucket.

    Synchronous collectives avoid retaining all-gather outputs through DDP's
    pending bucket futures until the optimizer step. CUDA stream dependencies
    still order each gather before its additions; no full-model gather survives.
    """
    group = process_group if process_group is not None else dist.group.WORLD
    value = bucket.buffer()
    if value.dtype != torch.float32:
        raise RuntimeError('Deterministic gradient averaging requires FP32 buckets')
    world = dist.get_world_size(group)
    for part in value.split(CHUNK_ELEMENTS):
        gathered = [torch.empty_like(part) for _ in range(world)]
        dist.all_gather(gathered, part, group=group)
        part.copy_(gathered[0])
        for rank in range(1, world):
            part.add_(gathered[rank])
        part.div_(world)
        del gathered
    future = torch.futures.Future()
    future.set_result(value)
    return future
