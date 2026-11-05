"""Use the existing cloud ledger/lock and liability formula, without a new scheduler.

API jobs have budget_rate=0, no pod, and state=api_reserved, so existing cloud
liability counts their full reservation, guard_job leaves them alone, and
potentially_live_pods does not count them. released_at means the reservation was
settled (no GPU was allocated). Uncertain requests never release the reservation.
"""
import json
import math
import time
from pathlib import Path
from experiments import cloud
from .common import ROOT, digest, durable_json, file_hash, lock


class Budget:
    def __init__(self, state_dir=None):
        self.directory = Path(state_dir or cloud.STATE)
        self.path = self.directory / 'ledger.json'

    def authorize(self, plan, approval, run_dir):
        if approval.get('approved') is not True or approval.get('scope') != 'teacher-pilot-openai':
            raise ValueError('Paid teacher generation has no recorded approval')
        if approval.get('plan_sha256') != digest(plan):
            raise ValueError('Approval must bind this exact plan')
        limit = approval.get('max_usd')
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < plan['reservation_usd']:
            raise ValueError('Approved amount does not cover the complete plan')
        evidence = ROOT / approval.get('evidence_path', '')
        if not evidence.is_file() or file_hash(evidence) != approval.get('evidence_sha256'):
            raise ValueError('Missing authorization evidence (do not manufacture approval)')
        if time.time() >= approval.get('expires_at', 0):
            raise ValueError('Approval expired or expiry missing')
        name = 'teacher-api-' + digest(plan)[:24]
        with lock(self.directory / 'lock'):
            state = json.loads(self.path.read_text())
            job = state['jobs'].get(name)
            if job:
                if job.get('plan_sha256') != digest(plan) or job.get('run_dir') != str(Path(run_dir).resolve()):
                    raise ValueError('Existing campaign reservation belongs to another run directory')
                if job.get('released_at'):
                    raise ValueError('Reservation already settled; cannot send more requests')
                if job.get('authorization_sha256') != digest(approval):
                    raise ValueError('Authorization differs from original reservation')
            else:
                request_ids = {r['request_key'] for r in plan['requests']}
                for previous in state['jobs'].values():
                    if request_ids.intersection(previous.get('request_keys', [])):
                        raise ValueError('Requests already claimed by another plan; reconcile existing evidence')
                ceiling = min(cloud.USABLE, float(state['ceiling_usd']) - (cloud.CEILING - cloud.USABLE))
                if cloud.liability(state['jobs']) + plan['reservation_usd'] > ceiling:
                    raise ValueError('Shared campaign has insufficient unreserved budget')
                state['jobs'][name] = {
                    'kind': 'api', 'provider': 'openai', 'state': 'api_reserved',
                    'created_at': time.time(), 'budget_rate': 0,
                    'reserved_usd': plan['reservation_usd'], 'safely_no_pod': True,
                    'original_reservation_usd': plan['reservation_usd'],
                    'plan_sha256': digest(plan), 'run_dir': str(Path(run_dir).resolve()),
                    'authorization_sha256': digest(approval), 'authorization': approval,
                    'request_keys': sorted(request_ids),
                    'charged_estimate_usd': 0,
                }
                durable_json(self.path, state)
        return name

    def check_before_send(self, name):
        with lock(self.directory / 'lock'):
            state = json.loads(self.path.read_text())
            job = state['jobs'][name]
            if job.get('released_at') or time.time() >= job['authorization']['expires_at']:
                raise ValueError('Reservation settled or approval expired')
            ceiling = min(cloud.USABLE, float(state['ceiling_usd']) - (cloud.CEILING - cloud.USABLE))
            if cloud.liability(state['jobs']) > ceiling:
                raise ValueError('Shared campaign liabilities exceed usable cap')

    def observe(self, name, request, cost):
        """Persist known usage even on failure; an overrun must not hide in a file."""
        amount = cost['accounted_usd']
        if not math.isfinite(amount) or amount < 0:
            raise ValueError('Invalid observed API cost')
        with lock(self.directory / 'lock'):
            state = json.loads(self.path.read_text())
            job = state['jobs'][name]
            observations = job.setdefault('usage_observations', {})
            observations[request['request_key']] = {
                'accounted_usd': amount, 'nominal_usd': cost['nominal_usd'],
                'overrun_usd': max(0, amount - request['reserved_usd'])}
            job['charged_estimate_usd'] = sum(o['accounted_usd'] for o in observations.values())
            job['reserved_usd'] = job['original_reservation_usd'] + sum(o['overrun_usd'] for o in observations.values())
            durable_json(self.path, state)

    def settle(self, name, cost, request_count):
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('Invalid cost')
        with lock(self.directory / 'lock'):
            state = json.loads(self.path.read_text())
            job = state['jobs'][name]
            # Keep even an unexpected overage in the campaign; never hide it.
            job.update(released_at=time.time(), state='api_settled', charged_estimate_usd=cost,
                       request_count=request_count, settlement='usage-priced estimate; not provider invoice')
            durable_json(self.path, state)
