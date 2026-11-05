"""Global SOL groups and distinct samples even when a group crosses rank boundaries."""
from .objective import grouped_rewards


def gather_parts(local):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return [local], 0
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, local)
    return parts, dist.get_rank()


def rewards_for_parts(parts, rank, **settings):
    rows = [row for part in parts for row in part]
    size = settings.get('group_size', 8)
    if any(len({r[0] for r in rows[start:start+size]}) != 1 for start in range(0, len(rows), size)):
        raise ValueError('Global row order does not match TRL contiguous advantage groups')
    values, targets = grouped_rewards([r[0] for r in rows], [r[1] for r in rows],
                                     [r[2] for r in rows], **settings)
    start = sum(len(p) for p in parts[:rank])
    return values[start:start + len(parts[rank])], targets


def global_rewards(ids, correct, lengths, **settings):
    parts, rank = gather_parts(list(zip(ids, correct, lengths)))
    return rewards_for_parts(parts, rank, **settings)


def global_sampling_parameters(inputs, parameters, **settings):
    from .sampling import independent_parameters
    parts, rank = gather_parts(inputs)
    flat = [p for part in parts for p in part]
    values = independent_parameters(flat, parameters, **settings)
    start = sum(len(p) for p in parts[:rank])
    return values[start:start + len(inputs)]


def atomic_json(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
