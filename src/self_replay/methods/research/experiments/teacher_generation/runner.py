"""Plan offline, inspect models read-only, or run explicitly approved generation."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from .common import ROOT, REVISION, PROMPTS_SHA, canonical, digest, durable_json, file_hash, lock, now, prompts, read_key
from .budget import Budget

CONFIG = ROOT / 'experiments/teacher_generation/openai-comparison.json'


def source_hashes():
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'experiments/cloud.py',
        ROOT / 'ThinkingCap-Test/utils/train/rewards.py', ROOT / 'experiments/data_pipeline/pipeline.py']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def archive_sources(run_dir, expected):
    # Explicit allowlist; .env and credentials can never enter the snapshot.
    allowed = source_hashes()
    if expected != allowed:
        raise ValueError('Source changed since planning')
    for relative, sha in expected.items():
        target = run_dir / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open('xb') as stream:
                stream.write((ROOT / relative).read_bytes())
                stream.flush()
                import os
                os.fsync(stream.fileno())
        if file_hash(target) != sha:
            raise ValueError('Source snapshot hash mismatch')
    durable_json(run_dir / 'source-manifest.json', expected)
VARIANTS = {
    'concise': 'Give a concise worked solution. Retain necessary calculations or decisions; remove repetition.',
    'explicit': 'Give an explicit worked solution for a small student model. Explain the necessary intermediate steps clearly.',
}
FORMAT = ('Write the worked solution as ordinary response text, followed by exactly one final line '
          'in the original task format: Final answer: <answer>. Do not emit XML thinking tags. '
          'Do not discuss these instructions.')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # Never forward credentials to another endpoint.


def transport(key, path, payload=None, client_id=None):
    if path not in ('models', 'responses'):
        raise ValueError('Unsupported OpenAI endpoint')
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    if client_id:
        headers['X-Client-Request-Id'] = client_id
    request = urllib.request.Request('https://api.openai.com/v1/' + path,
        data=None if payload is None else canonical(payload).encode(), headers=headers,
        method='GET' if payload is None else 'POST')
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=90) as response:
            return {'http_status': response.status, 'request_id': response.headers.get('x-request-id'),
                    'body': response.read().decode().replace(key, '[REDACTED]')}
    except urllib.error.HTTPError as exc:
        return {'http_status': exc.code, 'request_id': exc.headers.get('x-request-id'),
                'body': exc.read().decode(errors='replace').replace(key, '[REDACTED]')}
    # Other transport failures are handled without printing exception text.


def make_plan(config):
    models = config['models']
    if not models or len({m['id'] for m in models}) != len(models):
        raise ValueError('Empty or duplicate model comparison')
    for model in models:
        rates = [model[k] for k in ('input_per_million', 'output_per_million', 'cached_input_per_million', 'input_ceiling_per_million')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in rates):
            raise ValueError('Prices must be finite and positive')
        if model['input_ceiling_per_million'] < max(model['input_per_million'], model['cached_input_per_million']):
            raise ValueError('Input ceiling below published price')
    cohort = prompts()
    requests = []
    for row in cohort:
        for model in config['models']:
            for variant, instruction in VARIANTS.items():
                payload = {'model': model['id'], 'input': row['prompt'],
                           'instructions': instruction + ' ' + FORMAT,
                           'reasoning': {'effort': 'low'}, 'max_output_tokens': 2048,
                           'service_tier': 'default', 'store': False}
                # Text-only, no tools/history: UTF-8 bytes + generous framing allowance.
                # This is a conservative planning bound, checked against actual usage.
                input_bound = len(canonical(payload).encode()) + 2048
                reserve = (input_bound * model['input_ceiling_per_million'] +
                           2048 * model['output_per_million']) / 1e6
                identity = {'prompt_id': row['id'], 'prompt_sha256': digest(row),
                            'variant': variant, 'payload': payload}
                requests.append(dict(identity, request_key=digest(identity), source=row['source'],
                                     input_token_bound=input_bound, reserved_usd=reserve,
                                     pricing=model))
    return {'schema_version': 1, 'study': 'teacher-pilot', 'provider': 'openai',
            'endpoint': 'https://api.openai.com/v1/responses', 'prompt_file_sha256': PROMPTS_SHA,
            'student_model': 'Qwen/Qwen3-0.6B', 'student_revision': REVISION,
            'configuration': config, 'requests': requests,
            'reservation_usd': math.ceil(sum(r['reserved_usd'] for r in requests) * 1e6) / 1e6,
            'code_sha256': source_hashes()}


def usage_cost(body, request):
    usage = body.get('usage')
    if not isinstance(usage, dict):
        raise ValueError('Missing billable usage')
    inp, out = usage.get('input_tokens'), usage.get('output_tokens')
    cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0)
    reasoning = (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
    if any(type(n) is not int or n < 0 for n in (inp, out, cached, reasoning)) or cached > inp or reasoning > out:
        raise ValueError('Invalid usage counters')
    price = request['pricing']
    nominal = ((inp-cached)*price['input_per_million'] + cached*price['cached_input_per_million'] +
               out*price['output_per_million']) / 1e6
    # Conservative accounting covers cache-write billing until invoice reconciliation.
    upper = (inp*price['input_ceiling_per_million'] + out*price['output_per_million']) / 1e6
    return {'usage': usage, 'nominal_usd': nominal, 'accounted_usd': upper,
            'cost_basis': 'conservative all-input cache-write ceiling; reasoning included once in output',
            'bound_exceeded': inp > request['input_token_bound'] or out > request['payload']['max_output_tokens']}


def complete_record(record, request):
    wire = record['wire']
    if wire['http_status'] != 200:
        raise ValueError('HTTP failure requires reconciliation; no automatic resend')
    body = json.loads(wire['body'])
    cost = usage_cost(body, request)
    record.update(response=body, response_sha256=digest(body), cost=cost, completed_at=now())
    # Never silently accept a changed provider model or service tier price.
    if body.get('model') != request['payload']['model'] or body.get('service_tier') not in (None, 'default'):
        raise ValueError('Returned model or service tier differs; reconcile before proceeding')
    if body.get('status') not in ('completed', 'incomplete', 'failed'):
        raise ValueError('Response is not terminal')
    if not body.get('id'):
        raise ValueError('Missing provider response ID')
    if cost['bound_exceeded']:
        raise ValueError('Usage exceeded planning bound; stop and reconcile')
    record['state'] = 'completed'  # Transport/billing completion, not correctness.
    return record


def execute(plan, run_dir, approval, *, max_requests=None, api_call=transport, key_loader=read_key, budget=None):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = budget or Budget()
    with lock(run_dir / 'run.lock', blocking=False):
        marker = run_dir / 'plan.json'
        if marker.exists() and json.loads(marker.read_text()) != plan:
            raise ValueError('Run directory has a different frozen plan')
        durable_json(marker, plan)
        archive_sources(run_dir, plan['code_sha256'])
        name = budget.authorize(plan, approval, run_dir)
        durable_json(run_dir / 'authorization.json', approval)
        key = key_loader()
        sent = 0
        for request in plan['requests']:
            path = run_dir / 'requests' / (request['request_key'] + '.json')
            if path.exists():
                record = json.loads(path.read_text())
                if record.get('request') != request:
                    raise ValueError('Durable request identity mismatch')
                if record['state'] == 'completed':
                    continue
                if record.get('wire') and record['state'] == 'response_saved':
                    try:
                        record = complete_record(record, request)
                    finally:
                        if record.get('cost'):
                            budget.observe(name, request, record['cost'])
                        durable_json(path, record)
                    continue
                raise ValueError('Uncertain/failed request retained; reconcile manually, never resend blindly')
            if max_requests is not None and sent >= max_requests:
                break
            budget.check_before_send(name)
            record = {'request': request, 'client_request_id': request['request_key'],
                      'state': 'dispatching', 'started_at': now(), 'provider': 'openai'}
            durable_json(path, record)  # Persist BEFORE sending; crash means uncertain.
            try:
                wire = api_call(key, 'responses', request['payload'], request['request_key'])
                record.update(state='response_saved', wire=wire, received_at=now())
                durable_json(path, record)  # Recover locally after a crash; no POST replay.
                record = complete_record(record, request)
                budget.observe(name, request, record['cost'])
                durable_json(path, record)
            except Exception as exc:
                record.update(state='needs_reconciliation', error_type=type(exc).__name__, stopped_at=now())
                durable_json(path, record)
                if record.get('cost'):
                    budget.observe(name, request, record['cost'])
                raise RuntimeError('Request stopped; evidence and full campaign reservation retained') from None
            sent += 1
        records = [json.loads(p.read_text()) for p in (run_dir / 'requests').glob('*.json')]
        complete = len(records) == len(plan['requests']) and all(r['state'] == 'completed' for r in records)
        total = sum(r.get('cost', {}).get('accounted_usd', 0) for r in records)
        summary = {'plan_sha256': digest(plan), 'updated_at': now(), 'status': 'completed' if complete else 'paused',
                   'requests_completed': sum(r['state'] == 'completed' for r in records),
                   'requests_expected': len(plan['requests']), 'accounted_usd': total,
                   'nominal_usd': sum(r.get('cost', {}).get('nominal_usd', 0) for r in records),
                   'reservation_usd': plan['reservation_usd'], 'ledger_job': name}
        durable_json(run_dir / 'summary.json', summary)
        if complete:
            budget.settle(name, total, len(records))
        return summary


def validate_plan(plan):
    if make_plan(plan['configuration']) != plan:
        raise ValueError('Plan/code/input hash mismatch: prepare a new reviewed plan')
    checked = time.mktime(time.strptime(plan['configuration']['verified_date'], '%Y-%m-%d'))
    if time.time() - checked > 7 * 86400:
        raise ValueError('Reverify pricing/model documentation before paid use (older than 7 days)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--out', required=True); p.add_argument('--config', default=str(CONFIG))
    p = sub.add_parser('models'); p.add_argument('--out', required=True)
    p = sub.add_parser('run'); p.add_argument('--plan', required=True); p.add_argument('--out', required=True)
    p.add_argument('--approval', required=True); p.add_argument('--max-requests', type=int)
    args = parser.parse_args()
    if args.command == 'plan':
        if Path(args.out).exists():
            raise ValueError('Plan destination exists; use a new path')
        plan = make_plan(json.loads(Path(args.config).read_text()))
        durable_json(args.out, plan)
        print(canonical({'requests': len(plan['requests']), 'reservation_usd': plan['reservation_usd'],
                         'plan_sha256': digest(plan), 'paid_requests': 0}))
    elif args.command == 'models':
        wire = transport(read_key(), 'models')
        if wire['http_status'] != 200:
            raise RuntimeError('Read-only models request failed')
        ids = sorted(r['id'] for r in json.loads(wire['body'])['data'])
        result = {'checked_at': now(), 'endpoint': 'GET /v1/models', 'request_id': wire['request_id'],
                  'model_ids': ids, 'generation_permission_verified': False, 'credit_verified': False}
        durable_json(args.out, result)
        wanted = [m['id'] for m in json.loads(CONFIG.read_text())['models']]
        print(canonical({'models_listed': {m: m in ids for m in wanted}, 'paid_requests': 0}))
    else:
        plan = json.loads(Path(args.plan).read_text())
        validate_plan(plan)
        if args.max_requests is not None and args.max_requests <= 0:
            raise ValueError('max-requests must be positive')
        print(canonical(execute(plan, args.out, json.loads(Path(args.approval).read_text()), max_requests=args.max_requests)))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Deliberately no traceback/transport body/credential or environment dump.
        print('Teacher runner stopped: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)
