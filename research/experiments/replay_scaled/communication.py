"""CPU transport for Python records; CUDA collectives remain for gradients."""
from datetime import timedelta

PROTOCOL = 'gloo-records-nccl-gradients-v1'


def gather_records(rows, dist, group):
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, rows, group=group)
    return [row for part in values for row in part]


def transport_probe(dist, torch, group, device):
    """Exercise unequal multi-MB result lists and repeated gradient-size reductions."""
    rank, world = dist.get_rank(), dist.get_world_size()
    rows = [dict(rank=rank, sample=i, text=str(i) + 'x' * (8192 + rank * 7)) for i in range(512)]
    received = gather_records(rows, dist, group)
    assert len(received) == 512 * world
    for r in received:
        assert r['text'] == str(r['sample']) + 'x' * (8192 + r['rank'] * 7)
    for _ in range(16):
        tensor = torch.full((262144,), float(rank + 1), device=device)
        dist.all_reduce(tensor)
        expected=world * (world + 1) / 2
        if not torch.all(tensor == expected).item():
            raise RuntimeError(f'GPU reduction mismatch rank={rank} expected={expected} '
                f'min={tensor.min().item()} max={tensor.max().item()} first={tensor[:8].tolist()}')
    return dict(passed=True, protocol=PROTOCOL, world_size=world,
                records_per_rank=512, minimum_text_bytes_per_rank=512 * 8192,
                tensor_reductions=16,
                transport_environment={k:__import__('os').environ.get(k) for k in ('NCCL_P2P_DISABLE','NCCL_IB_DISABLE')})


def main():
    import json, os
    from pathlib import Path
    import torch
    import torch.distributed as dist
    from .core import write
    from experiments.shorterbetter.hardware import verify_worker
    rank = int(os.environ['LOCAL_RANK'])
    lease = json.loads(Path('worker-lease.json').read_text())
    verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], lease)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(seconds=90))
    group = dist.new_group(backend='gloo', timeout=timedelta(seconds=90))
    result = transport_probe(dist, torch, group, device)
    if rank == 0:
        write('output/communication-verification.json', result)
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
