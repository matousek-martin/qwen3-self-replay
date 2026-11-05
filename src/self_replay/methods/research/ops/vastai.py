"""Vast transport for the canonical cloud controller; never an independent allocator.

Secrets remain on the control server. Instance IDs use a provider prefix so a
guard or resumed controller cannot accidentally operate on another provider.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://console.vast.ai/api/'
GPU_NAMES = {'NVIDIA GeForce RTX 4090': 'RTX 4090',
             'NVIDIA A100-SXM4-80GB': 'A100 SXM4',
             'NVIDIA H100 80GB HBM3': 'H100 SXM'}
IMAGE = 'nvidia/cuda:12.8.1-devel-ubuntu22.04'
SSH_KEY = Path(os.environ.get('SELF_REPLAY_SSH_KEY', str(Path.home() / '.ssh/id_ed25519')) + '.pub')


def api(path, method='GET', payload=None, params=None):
    key = os.environ.get('VASTAI_API_KEY')
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix('export ')
            if line.startswith('VASTAI_API_KEY='):
                key = shlex.split(line.split('=', 1)[1], comments=True)[0]
    if not key:
        raise RuntimeError('VASTAI_API_KEY missing')
    url = BASE + path
    if params:
        url += '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, method=method,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        # Error bodies can contain request material; never log them or credentials.
        raise RuntimeError(f'Vast HTTP {exc.code}: request failed') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('Vast transport failure; mutation outcome may be uncertain') from None
    if isinstance(result, dict) and result.get('success') is False:
        raise RuntimeError('Vast rejected request; mutation outcome needs reconciliation')
    return result


def normalize(row):
    identity = str(row['id'])
    if not identity.isdecimal():
        raise ValueError('Invalid Vast instance ID')
    status = str(row.get('actual_status', 'unknown')).upper()
    # Offline / stopped instances remain billable; never count them as released.
    ssh = {}
    direct_requested = 'ssh_direc' in str(row.get('image_runtype', ''))
    if status == 'RUNNING' and row.get('ssh_host') and row.get('ssh_port') and not direct_requested:
        ssh['direct'] = {'host': row['ssh_host'], 'port': int(row['ssh_port']), 'username': 'root'}
    mappings = (row.get('ports') or {}).get('22/tcp') or []
    if status == 'RUNNING' and row.get('public_ipaddr') and mappings:
        port = int(mappings[0]['HostPort'])
        if not 0 < port < 65536:
            raise ValueError('Invalid published SSH port')
        ssh['direct'] = {'host': row['public_ipaddr'], 'port': port, 'username': 'root'}
    if (status=='RUNNING' and str(row.get('label','')).startswith(('thinkingcap-pilot-coding-frontier-','thinkingcap-pilot-coding-abc-'))
            and row.get('ssh_host') and row.get('ssh_port')):
        # Frontier hosts lost published-port reachability while Vast's SSH
        # relay remained healthy; retain both endpoints and record transport.
        ssh['published_direct']=ssh.get('direct')
        ssh['direct']={'host':row['ssh_host'],'port':int(row['ssh_port']),'username':'root'}
        ssh['transport']='vast_ssh_relay'
    return {'id': 'vast:' + identity, 'provider': 'vast', 'provider_id': identity,
            'name': row.get('label') or 'vast-unlabelled-' + identity,
            'status': status, 'ssh': ssh, 'cost': row.get('dph_total'),
            'gpu': {'name': row.get('gpu_name'), 'count': row.get('num_gpus')},
            'createdAt': row.get('start_date'), 'image': row.get('image_uuid'),
            'disk': row.get('disk_space'), 'machine_id': row.get('machine_id')}


def list_pods():
    rows, tokens, ids = [], set(), set()
    token = None
    for _ in range(100):
        page = api('v1/instances/', params={'next_token': token} if token else None)
        if (not isinstance(page, dict) or not isinstance(page.get('instances'), list)
                or type(page.get('total_instances')) is not int or 'next_token' not in page):
            raise RuntimeError('Vast listing completeness unknown')
        for row in page['instances']:
            pod = normalize(row)
            if pod['id'] in ids:
                raise RuntimeError('Vast listing repeated an instance')
            ids.add(pod['id'])
            rows.append(pod)
        token = page['next_token']
        if not token:
            if len(rows) != page['total_instances']:
                raise RuntimeError('Vast listing total changed; retry a complete read')
            return rows
        if token in tokens:
            raise RuntimeError('Vast listing repeated a page token')
        tokens.add(token)
    raise RuntimeError('Vast listing exceeded page limit')


def pod_api(path, method='GET', payload=None):
    identity = path.removeprefix('pods/vast:')
    if not path.startswith('pods/vast:') or not identity.isdecimal() or payload is not None:
        raise ValueError('Invalid Vast pod operation')
    if method == 'DELETE':
        return api('v0/instances/' + identity + '/', 'DELETE')
    if method != 'GET':
        raise ValueError('Unsupported Vast pod operation')
    # Full listing avoids interpreting an eventually consistent single GET as
    # proof of deletion, and uses the current v1 listing API.
    for pod in list_pods():
        if pod['id'] == 'vast:' + identity:
            return pod
    raise RuntimeError('Vast HTTP 404: absent from complete instance listing')


def ensure_ssh_key():
    public = SSH_KEY.read_text().strip()
    if not public.startswith(('ssh-ed25519 ', 'ssh-rsa ')):
        raise ValueError('Invalid project public SSH key')
    response = api('v0/ssh/')
    rows = response if isinstance(response, list) else response.get('ssh_keys', [])
    if not isinstance(rows, list):
        raise RuntimeError('Unexpected SSH key listing')
    identity = public.split()[:2]
    if any(str(row.get('public_key', row.get('ssh_key', row.get('key', '')))).split()[:2] == identity for row in rows):
        return {'ssh_key_registered': True, 'created': False}
    api('v0/ssh/', 'POST', {'ssh_key': public})
    return {'ssh_key_registered': True, 'created': True}


def quote(gpu, count, disk_gb=80, offer_id=None, min_duration=0, exclude_offer_ids=(), exclude_machine_ids=(), allowed_countries=()):
    if gpu not in GPU_NAMES or count not in (1, 2, 4, 8) or not 50 <= disk_gb <= 500:
        raise ValueError('Unsupported Vast hardware or disk size')
    q = {'verified': {'eq': True}, 'rentable': {'eq': True}, 'rented': {'eq': False},
         'gpu_name': {'eq': GPU_NAMES[gpu]}, 'num_gpus': {'eq': count},
         'gpu_ram': {'gte': 24000 if '4090' in gpu else 80000},
         'cuda_max_good': {'gte': 12.8}, 'disk_space': {'gte': disk_gb},
         'reliability2': {'gte': .99}, 'type': 'on-demand',
         'order': [['dph_total', 'asc']], 'limit': 1000 if offer_id is not None or allowed_countries else 20}
    # The endpoint filters exact offers by ask_contract_id, not returned id.
    # Generic search bundles interchangeable GPU slots and can omit a still-rentable ID.
    if offer_id is not None:q['ask_contract_id']={'eq':int(offer_id)}
    # Also match locally; never fall back to another authorized offer ID.
    response = api('v0/bundles/', params={'q': json.dumps(q)})
    if not isinstance(response.get('offers'), list):
        raise RuntimeError('Invalid Vast offer listing')
    candidates = []
    for row in response['offers']:
        if allowed_countries and row.get('geolocation','').rsplit(',',1)[-1].strip() not in allowed_countries:
            continue
        if (int(row['id']) in exclude_offer_ids or row.get('machine_id') in exclude_machine_ids
                or row.get('gpu_name') != GPU_NAMES[gpu] or row.get('num_gpus') != count
                or float(row.get('gpu_ram', 0)) < q['gpu_ram']['gte']
                or float(row.get('disk_space', 0)) < disk_gb
                or float(row.get('duration', 0)) < min_duration
                or not (row.get('verified') is True or row.get('verification') == 'verified')
                or row.get('rentable') is not True
                or row.get('rented') is not False
                or float(row.get('reliability2', 0)) < .99
                or float(row.get('cuda_max_good', 0)) < 12.8
                or (offer_id is not None and int(row['id']) != int(offer_id))):
            continue
        # Explicit prices are required for storage AND both transfer directions.
        values = [float(row[k]) for k in ('dph_total', 'storage_cost', 'inet_up_cost', 'inet_down_cost')]
        if not all(math.isfinite(v) and v >= 0 for v in values) or values[0] <= 0:
            continue
        # Conservatively add the full selected disk; dph_total may already contain
        # the search default disk. Double counting that small amount is deliberate.
        rate = values[0] + disk_gb * values[1] / (30 * 24)
        candidates.append({'offer_id': int(row['id']), 'gpu': gpu, 'gpu_count': count,
            'disk_gb': disk_gb, 'hourly_ceiling': rate, 'quoted_dph_total': values[0],
            'transfer_costs': {'upload': values[3], 'download': values[2]},
            'offer': row})
    return min(candidates, key=lambda x: x['hourly_ceiling']) if candidates else None


def create(quote, name, direct_ssh=False):
    result = api('v0/asks/' + str(quote['offer_id']) + '/', 'PUT',
        {'image': IMAGE, 'disk': quote['disk_gb'], 'label': name,
         'runtype': 'ssh_direc ssh_proxy' if direct_ssh else 'ssh',
         'target_state': 'running', 'cancel_unavail': True})
    identity = result.get('new_contract')
    if result.get('success') is not True or type(identity) is not int or identity <= 0:
        raise RuntimeError('Vast create response ambiguous; reconcile exact label before further action')
    # Do not perform another fallible call before returning the newly acquired ID.
    return {'id': 'vast:' + str(identity), 'name': name, 'status': 'LOADING',
            'provider': 'vast', 'provider_id': str(identity), 'cost': quote['hourly_ceiling']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'setup', 'quote'])
    parser.add_argument('--gpu', choices=GPU_NAMES, default='NVIDIA A100-SXM4-80GB')
    parser.add_argument('--gpu-count', type=int, default=4)
    parser.add_argument('--disk-gb', type=int, default=80)
    args = parser.parse_args()
    if args.action == 'setup':
        result = ensure_ssh_key()
    elif args.action == 'quote':
        result = quote(args.gpu, args.gpu_count, args.disk_gb)
    else:
        user = api('v0/users/current/')
        result = {'credit_usd': user.get('credit'), 'can_pay': user.get('can_pay'),
                  'instances': list_pods()}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
