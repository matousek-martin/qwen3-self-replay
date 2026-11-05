"""New, finite authorization records; historical tickets remain separate."""
import json
import math
from pathlib import Path


def public_ticket(state, name, job):
    path = Path(state) / 'authorizations' / (name + '.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('scope') != 'public-reproduction-v1':
        return None
    if value.get('authorized') is not True or value.get('job_name') != name or not value.get('source_user_instruction'):
        raise ValueError('Invalid public job authorization')
    for key in ('provider', 'entry', 'hardware_scope', 'gpu_count'):
        if job.get(key) != value.get(key):
            raise ValueError('Public job changed: ' + key)
    if (job.get('gpu') or job.get('requested_gpu')) != value['gpu']:
        raise ValueError('Public job changed GPU')
    if value['provider'] == 'vast' and any(job.get(k) != value.get(k) for k in ('offer_id', 'disk_gb')):
        raise ValueError('Public job changed Vast offer or storage')
    for key in ('max_cost_usd', 'max_runtime_hours', 'max_gpu_rate'):
        cap = value.get(key)
        if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
            raise ValueError('Nonfinite public job bound')
    if not 1 < (job.get('max_cost_usd') or 0) <= value['max_cost_usd']:
        raise ValueError('Public job exceeds approved cost')
    if not 0 < (job.get('max_runtime_hours') or 0) <= value['max_runtime_hours'] <= 72:
        raise ValueError('Public job exceeds approved runtime')
    return value
