"""Continue untouched review items after a recorded provider server error.

Never resend a failed/uncertain request. Its sample fails the quality gate,
its original receipt and conservative budget liability remain untouched.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
import threading

import teacher_pipeline as previous

_progress_lock=threading.Lock()
_original_progress=previous.progress


def _serial_progress(*args,**kwargs):
    with _progress_lock:
        return _original_progress(*args,**kwargs)


previous.progress=_serial_progress


def verdict(item, request, receipt, common):
    result = {'verdict': 'fail', 'evidence': 'Incomplete or malformed independent review; rejected conservatively.'}
    if receipt['state'] == 'completed':
        body = receipt['response']
        if body.get('status') == 'completed':
            try:
                parsed = json.loads(previous.response_text(body))
                if parsed.get('verdict') in ('pass', 'fail') and isinstance(parsed.get('evidence'), str) and parsed['evidence'].strip():
                    result = parsed
            except (ValueError, TypeError):
                pass
        response_hash = receipt['response_sha256']
    else:
        result['evidence'] = 'Provider HTTP server error; no valid independent review. Sample rejected, request not resent, original budget liability retained.'
        response_hash = None
    return {'request_key': item['request_key'], 'response_sha256': item['response_sha256'],
            'reviewer': 'gpt-6.1-sol independent review call', **result,
            'review_request_key': request['request_key'], 'review_response_sha256': response_hash,
            'receipt_sha256': common.digest(receipt)}


def review(items,label,runner,common):
    """Parallelize only new plans; keep legacy plans and paid identities intact."""
    out=previous.STATE/label
    results=out/'reviews.json'
    if results.exists() or (out/'registered-plan.json').exists() or len(items)<8:
        return serial_review(items,label,runner,common)
    split={'partitions':4,'sample_keys':[r['request_key'] for r in items],
           'policy':'Disjoint independent reviews; preserve attempt order when joining; no failed request replay.'}
    path=out/'parallel-plan.json'
    if path.exists() and json.loads(path.read_text())!=split:
        raise ValueError('Parallel review plan changed')
    previous.write(path,split)
    def part(i):
        return serial_review(items[i::4],label+f'-part{i}',runner,common)
    with ThreadPoolExecutor(max_workers=4) as pool:
        partitions=list(pool.map(part,range(4)))
    indexed={r['request_key']:r for rows in partitions for r in rows}
    ordered=[indexed[item['request_key']] for item in items]
    previous.write(results,ordered)
    previous.backup_api(out,label)
    return ordered


def serial_review(items, label, runner, common, depth=0):
    out = previous.STATE / label
    results = out / 'reviews.json'
    if results.exists():
        previous.backup_api(out, label)
        return json.loads(results.read_text())

    def reconcile():
        plan_path = out / 'registered-plan.json'
        if not plan_path.exists():
            return None
        plan = json.loads(plan_path.read_text())
        if len(plan['requests']) != len(items):
            raise ValueError('Review continuation coverage changed')
        done, pending, failed = [], [], []
        for item, request in zip(items, plan['requests']):
            if request['prompt_id'] != item['prompt_id'] or request['target_sha256'] != item['response_sha256']:
                raise ValueError('Review continuation identity changed')
            path = out / 'run/requests' / (request['request_key'] + '.json')
            if not path.exists():
                pending.append(item)
                continue
            receipt = json.loads(path.read_text())
            if receipt.get('request') != request:
                raise ValueError('Review receipt identity changed')
            if receipt['state'] != 'completed':
                wire = receipt.get('wire', {})
                if receipt['state'] != 'needs_reconciliation' or wire.get('http_status') not in (500, 502, 503, 504):
                    raise RuntimeError('Unclassified API failure; retain evidence for inspection')
                failed.append(request['request_key'])
            done.append(verdict(item, request, receipt, common))
        return (done, pending, failed) if failed else None

    state = reconcile()
    if state is None:
        try:
            return previous.review(items, label, runner, common)
        except RuntimeError as exc:
            if str(exc) != 'Request stopped; evidence and full campaign reservation retained':
                raise
            state = reconcile()
            if state is None:
                raise
    done, pending, failed = state
    if depth >= 3:
        raise RuntimeError('Repeated provider failures; continuation stopped without resending requests')
    continuation = label + '-continuation'
    previous.write(out / 'server-error-continuation.json', {
        'failed_request_keys': failed, 'unsent_items': len(pending), 'continuation': continuation,
        'policy': 'Reject failed samples; continue only requests never dispatched. Preserve original ledger liability.',
    })
    if pending:
        time.sleep(5)
        done += serial_review(pending, continuation, runner, common, depth + 1)
    indexed = {r['request_key']: r for r in done}
    ordered = [indexed[item['request_key']] for item in items]
    previous.write(results, ordered)
    previous.backup_api(out, label)
    return ordered
