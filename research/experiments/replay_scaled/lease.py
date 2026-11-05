"""Reload atomically published funding extensions without changing training state."""
import json
import math
from pathlib import Path


def read_lease(path, identity):
    value = json.loads(Path(path).read_text())
    for key in ('job_name', 'pod_id', 'provider', 'hardware_scope', 'gpu', 'gpu_count', 'created_at'):
        if value.get(key) != identity.get(key):
            raise RuntimeError('Live lease identity changed: ' + key)
    cap = value['max_cost_usd']; rate = value['budget_rate']
    deadline = value['hard_deadline']; created = value['created_at']
    hours = value.get('max_runtime_hours', 72)
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (cap, rate, deadline, created, hours)):
        raise RuntimeError('Nonfinite live lease bounds')
    if not (1 < cap <= 150 and rate > 0 and 0 < hours <= 72
            and created < deadline <= min(created + hours * 3600, created + (cap - 1) / rate * 3600) + 1):
        raise RuntimeError('Live lease exceeds finite replay bounds')
    return value
