"""A named, bounded user-authorized launch exception; other budgets stay intact."""
import json
import math
from pathlib import Path


def evaluation_account_limit(state, name, job):
    """Account-wide tripwire for one registered concurrent evaluation campaign.

    The evaluator's own timed reservation remains unchanged. Account credit
    includes training too, so compare that signal with the remaining joint cap.
    Missing or inconsistent registration falls back to the stricter job cap.
    """
    fallback = job['max_cost_usd']
    try:
        auth = ticket(state, name, job)
        if auth and name.startswith('coding-curriculum-256-'):
            return auth['account_spending_limit_usd']
        if auth and name.startswith(('coding-frontier-','coding-abc-')):
            return auth['combined_cap_usd']
        if auth and name.startswith('replay-evaluation-'):
            return auth['account_spending_limit_usd']
        if not auth or auth.get('paired_evaluation_account_guard') is not True:
            return fallback
        path = Path(auth.get('paired_campaign_path', '')).resolve()
        if not path.is_relative_to(Path(state).resolve().parent/'shorterbetter-paired'):
            return fallback
        campaign = json.loads(path.read_text())
        early = campaign.get('early_evaluations', []) + campaign.get('parallel_evaluations', [])
        names = [a['job_name'] for a in campaign['arms'].values()] + [e['job_name'] for e in early]
        names += [e['job_name'] for e in campaign.get('parallel_previous_attempts', [])]
        if campaign.get('evaluation', {}).get('job_name'):
            names.append(campaign['evaluation']['job_name'])
        if (campaign.get('authorized') is not True or campaign.get('cap_usd') != 105
                or campaign.get('evaluation_reserve_usd') != 10
                or auth.get('evaluation_only') is not True
                or auth.get('full_training_authorized') is not False
                or name not in [e['job_name'] for e in early]
                or len(names) != len(set(names))):
            return fallback
        jobs = json.loads((Path(state)/'ledger.json').read_text())['jobs']
        previous = 0.0
        for other in names:
            if other == name:
                continue
            row = jobs.get(other, {})
            if row.get('released_at') and row['released_at'] <= job['created_at']:
                previous += row['charged_estimate_usd']
            elif row.get('created_at', float('inf')) < job['created_at']:
                previous += (job['created_at']-row['created_at'])/3600*row['budget_rate']
        limit = campaign['cap_usd']-previous
        return max(0, min(105, limit)) if math.isfinite(limit) else fallback
    except (OSError, ValueError, KeyError, TypeError):
        return fallback


def ticket(state, name, job):
    from experiments.public_policy import public_ticket
    public = public_ticket(state, name, job)
    if public is not None:
        return public
    if name.startswith('coding-curriculum-256-'):
        from experiments.coding_curriculum.budget import ticket as curriculum_ticket
        return curriculum_ticket(state, name, job)
    if name.startswith('coding-abc-'):
        from experiments.coding_abc.budget import ticket as abc_ticket
        return abc_ticket(state, name, job)
    if name.startswith('coding-frontier-'):
        from experiments.coding_frontier.budget import ticket as frontier_ticket
        return frontier_ticket(state, name, job)
    if name.startswith('opencoder-mode-probe-'):
        from experiments.opencoder_sft.mode_budget import ticket as mode_ticket
        return mode_ticket(state, name, job)
    if name.startswith('opencoder-sft-768-'):
        from experiments.opencoder_sft.runtime import ticket as oc_ticket
        return oc_ticket(state, name, job)
    if name.startswith('coding-replay-768-'):
        from experiments.replay_scaled.coding_budget import ticket as coding_ticket
        return coding_ticket(state, name, job)
    if name.startswith('coding-length-probe-'):
        from experiments.replay_scaled.length_budget import ticket as probe_ticket
        return probe_ticket(state, name, job)
    if name.startswith('replay-evaluation-'):
        from experiments.replay_scaled.evaluation_budget import ticket as evaluation_ticket
        return evaluation_ticket(state, name, job)
    if name.startswith('replay-scaled-'):
        from experiments.replay_scaled.core import ticket as replay_ticket
        return replay_ticket(state, name, job)
    if not name.startswith('shorterbetter-scaled-') or not name.replace('-', '').isalnum():
        return None
    path = Path(state) / 'authorizations' / (name + '.json')
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    gpu = job.get('gpu') or job.get('requested_gpu')
    if value.get('scope') == 'shorterbetter_vast_bounded_v1':
        from .hardware import GPU_RATE_CEILINGS
        valid = (value.get('authorized') is True and value.get('job_name') == name
            and value.get('provider') == job.get('provider') == 'vast'
            and job.get('hardware_scope') == 'shorterbetter-scaled-v1'
            and gpu == value.get('gpu') and gpu in GPU_RATE_CEILINGS
            and job.get('gpu_count') == value.get('gpu_count') and job.get('gpu_count') in (1, 2, 4)
            and job.get('entry') == 'experiments/shorterbetter/run-scaled.sh'
            and not job.get('completion_budget_scope')
            and type(value.get('offer_id')) is int and value['offer_id'] > 0
            and job.get('offer_id') == value['offer_id']
            and job.get('disk_gb') == value.get('disk_gb') and 50 <= value['disk_gb'] <= 500
            and 1 < (job.get('max_cost_usd') or 0) <= (value.get('max_cost_usd') or 0)
            and 0 < (job.get('max_runtime_hours') or 0) <= value.get('max_runtime_hours', 0) <= 168
            and 0 < value.get('max_gpu_rate', 0) <= GPU_RATE_CEILINGS[gpu]
            and bool(value.get('source_user_instruction')))
        funding = value.get('incremental_funding')
        if value.get('paired_short_experiment'):
            campaign_path = Path(value.get('paired_campaign_path', ''))
            expected_root = Path(state).resolve().parent/'shorterbetter-paired'
            if not campaign_path.resolve().is_relative_to(expected_root) or not campaign_path.is_file():
                return None
            campaign = json.loads(campaign_path.read_text())
            jobs = json.loads((Path(state)/'ledger.json').read_text())['jobs']
            names = [campaign.get('arms', {}).get(arm, {}).get('job_name') for arm in ('control','accuracy')]
            if name not in names or len(set(names)) != 2:
                return None
            previous = [jobs.get(n, {}) for n in names[:names.index(name)]]
            spent = sum(jobs.get(n, {}).get('charged_estimate_usd', 0) for n in names if n != name)
            valid &= (campaign.get('authorized') is True and campaign.get('cap_usd') == 105
                      and campaign.get('evaluation_reserve_usd') == 10 and funding is None
                      and value.get('rollout_batches') == 30 and value.get('gpu_count') == 4
                      and gpu == 'NVIDIA A100-SXM4-80GB'
                      and value['max_cost_usd'] <= min(55, 95-spent)
                      and all(j.get('released_at') and j.get('backup_verified') for j in previous))
        if funding is not None:
            minimum_start = 60
            parent_name = value.get('continuation_parent_job')
            if parent_name:
                ledger = Path(state)/'ledger.json'
                parent_path = Path(state)/'authorizations'/(parent_name+'.json')
                if not ledger.exists() or not parent_path.exists():
                    return None
                parent_job = json.loads(ledger.read_text())['jobs'].get(parent_name, {})
                parent_ticket = json.loads(parent_path.read_text())
                remaining = parent_ticket.get('max_cost_usd', 0) - parent_job.get('charged_estimate_usd', 0)
                predecessors = value.get('continuation_repair_jobs', [])
                if not isinstance(predecessors, list) or len(set(predecessors)) != len(predecessors):
                    return None
                jobs = json.loads(ledger.read_text())['jobs']
                for predecessor in predecessors:
                    old = jobs.get(predecessor, {})
                    auth_path = Path(state)/'authorizations'/(predecessor+'.json')
                    if (not old.get('released_at') or not old.get('backup_verified')
                            or not auth_path.is_file()
                            or json.loads(auth_path.read_text()).get('continuation_parent_job') != parent_name):
                        return None
                    remaining -= old.get('charged_estimate_usd', 0)
                valid &= bool(parent_job.get('released_at') and parent_job.get('backup_verified')
                              and parent_ticket.get('full_training_authorized')
                              and parent_job.get('provider') == 'vast'
                              and value['max_cost_usd'] <= remaining
                              and value.get('quality_stopping_disabled_by_user') is True)
                minimum_start = 40
            valid &= (funding == {'minimum_start_credit_usd': minimum_start, 'warning_credit_usd': 30,
                                  'stop_credit_usd': 12, 'release_credit_usd': 3}
                      and value.get('full_training_authorized') is True
                      and value.get('rollout_batches') == 300 and value.get('gpu_count') == 4
                      and value.get('gpu') == 'NVIDIA A100-SXM4-80GB')
        if job.get('hard_deadline'):
            valid &= job['hard_deadline'] <= job['created_at'] + value.get('max_runtime_hours', 0)*3600 + 1
        return value if valid else None
    # Historical RunPod tickets cannot authorize spending on another provider.
    if job.get('provider', 'runpod') != 'runpod':
        return None
    if value.get('scope') == 'shorterbetter_hardware_comparison_v1':
        from .hardware_benchmark import SETUPS, ROLLOUTS, MAX_HOURS
        setup = SETUPS.get(value.get('setup'))
        if not setup:
            return None
        expected_gpu, count, cap, rate = setup
        valid = (value.get('authorized') is True and value.get('job_name') == name
            and job.get('hardware_scope') == 'shorterbetter-scaled-v1'
            and gpu == value.get('gpu') == expected_gpu
            and job.get('gpu_count') == value.get('gpu_count') == count
            and job.get('entry') == 'experiments/shorterbetter/run-scaled.sh'
            and not job.get('completion_budget_scope')
            and 0 < (job.get('max_cost_usd') or 0) <= value.get('max_cost_usd', 0) <= cap
            and value.get('max_runtime_hours') == MAX_HOURS
            and value.get('pilot_rollouts') == ROLLOUTS
            and value.get('full_training_authorized') is False
            and 0 < value.get('max_gpu_rate', 0) <= rate)
        if job.get('hard_deadline'):
            valid &= job['hard_deadline'] <= job['created_at'] + MAX_HOURS*3600 + 1
        return value if valid else None
    if value.get('scope') == 'shorterbetter_multidomain_a100x2_benchmark':
        valid = (value.get('authorized') is True and value.get('job_name') == name
            and job.get('hardware_scope') == 'shorterbetter-scaled-v1'
            and gpu == value.get('gpu') == 'NVIDIA A100-SXM4-80GB'
            and job.get('gpu_count') == value.get('gpu_count') == 2
            and job.get('entry') == 'experiments/shorterbetter/run-scaled.sh'
            and not job.get('completion_budget_scope')
            and 0 < (job.get('max_cost_usd') or 0) <= (value.get('max_cost_usd') or 0) <= 10
            and value.get('max_runtime_hours') == 2
            and value.get('pilot_rollouts') == 2
            and value.get('full_training_authorized') is False
            and 0 < value.get('max_gpu_rate', 0) <= 1.79)
        if job.get('hard_deadline'):
            valid &= job['hard_deadline'] <= job['created_at'] + 2*3600 + 1
        return value if valid else None
    if value.get('scope') == 'shorterbetter_multidomain_pilot':
        valid = (value.get('authorized') is True and value.get('job_name') == name
            and job.get('hardware_scope') == 'shorterbetter-scaled-v1'
            and gpu == value.get('gpu') == 'NVIDIA GeForce RTX 4090'
            and job.get('gpu_count') == value.get('gpu_count') == 1
            and job.get('entry') == 'experiments/shorterbetter/run-scaled.sh'
            and not job.get('completion_budget_scope')
            and 0 < (job.get('max_cost_usd') or 0) <= (value.get('max_cost_usd') or 0) <= 5
            and value.get('max_runtime_hours') == 3
            and value.get('pilot_rollouts') == 3
            and 0 < value.get('max_gpu_rate', 0) <= .89)
        if job.get('hard_deadline'):
            valid &= job['hard_deadline'] <= job['created_at'] + 3*3600 + 1
        return value if valid else None
    if value.get('scope') == 'shorterbetter_4090_verification':
        count = job.get('gpu_count')
        valid = (value.get('authorized') is True and value.get('job_name') == name
            and job.get('hardware_scope') == 'shorterbetter-scaled-v1'
            and gpu == value.get('gpu') == 'NVIDIA GeForce RTX 4090'
            and count == value.get('gpu_count') and count in (1, 2)
            and job.get('entry') == 'experiments/shorterbetter/run-scaled.sh'
            and not job.get('completion_budget_scope')
            and 0 < (job.get('max_cost_usd') or 0) <= (value.get('max_cost_usd') or 0) <= 5
            and value.get('max_runtime_hours') == 2
            and 0 < value.get('max_gpu_rate', 0) <= .89)
        if job.get('hard_deadline'):
            valid &= job['hard_deadline'] <= job['created_at'] + 2*3600 + 1
        return value if valid else None
    if (value.get('authorized') is not True or value.get('job_name') != name
            or job.get('hardware_scope') != 'shorterbetter-scaled-v1'
            or gpu != value.get('gpu') or gpu != 'NVIDIA A100-SXM4-80GB'
            or job.get('gpu_count') != 4 or value.get('gpu_count') != 4
            or job.get('entry') != 'experiments/shorterbetter/run-scaled.sh'):
        return None
    if value.get('scope') == 'finish_registered_1200_updates':
        return value if (job.get('completion_budget_scope') == value['scope']
            and value.get('optimizer_updates') == 1200
            and value.get('max_cost_usd') is None and job.get('max_cost_usd') is None
            and value.get('max_runtime_hours') is None and not job.get('hard_deadline')
            and 0 < value.get('max_gpu_rate', 0) <= 1.79) else None
    if (value.get('scope') != 'one_4xa100_training_with_checks'
            or not 0 < (job.get('max_cost_usd') or 0) <= (value.get('max_cost_usd') or 0) <= 75
            or value.get('max_runtime_hours') != 8
            or not 0 < value.get('max_gpu_rate', 0) <= 1.79):
        return None
    if job.get('hard_deadline') and job['hard_deadline'] > job['created_at'] + 8*3600 + 1:
        return None
    return value
