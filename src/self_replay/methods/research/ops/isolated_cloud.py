"""Immutable payloads, canonical cloud controller and one shared ledger.

This is a source/data adapter, not another allocator. Credentials, locks,
reservations, cleanup and provider operations remain in the original controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def retry_upload(sync,pod,local,remote_path,**kwargs):
    """Retry only idempotent uploads, within their original total time budget."""
    if not kwargs.get('upload'):
        return sync(pod,local,remote_path,**kwargs)
    deadline=time.monotonic()+kwargs.get('timeout',90)
    for attempt in range(3):
        try:
            return sync(pod,local,remote_path,**dict(kwargs,timeout=max(1,deadline-time.monotonic())))
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            if isinstance(exc,subprocess.CalledProcessError) and exc.returncode not in (10,12,30,35,255):raise
            delay=5*(attempt+1)
            if attempt==2 or deadline-time.monotonic()<=delay:raise
            time.sleep(delay)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_uv_bootstrap(cloud, sync, pod, local, remote_path, script, meta, timeout=300, retry_transport=False):
    """Bound remote download, then fall back to uploading our verified local binary."""
    import shlex
    begin=time.monotonic()
    def remote(command, timeout):
        deadline=time.monotonic()+timeout
        for attempt in range(3 if retry_transport else 1):
            try:return cloud.remote(pod,command,timeout=max(1,deadline-time.monotonic()))
            except subprocess.CalledProcessError as exc:
                if not retry_transport or exc.returncode!=255 or attempt==2 or deadline-time.monotonic()<6:raise
                time.sleep(5)
    try:
        remote('timeout --kill-after=5 110 python3 -c '+shlex.quote(script),timeout=125)
        return
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
        remaining=max(1,timeout-(time.monotonic()-begin)-25)
        if retry_transport:retry_upload(sync,pod,local,remote_path,upload=True,timeout=remaining)
        else:sync(pod,local,remote_path,upload=True,timeout=remaining)
        check="""import json,hashlib,os
from pathlib import Path
m=json.loads(META);p=Path('/workspace/project/uv')
assert hashlib.sha256(p.read_bytes()).hexdigest()==m['binary_sha256']
p.chmod(0o755);os.utime(p,ns=(m['mtime'],m['mtime']))
m['method']='verified_local_binary_upload_after_download_failure'
Path('/workspace/project/output/uv-bootstrap-verification.json').write_text(json.dumps(m,indent=2))
""".replace('META',repr(json.dumps(meta)))
        remote('python3 -c '+shlex.quote(check),timeout=25)


def payload_files(root):
    roots = [root / 'experiments', root / 'ThinkingCap-Test',
             root / 'lab/experiments/data', root / 'lab/gpu-requirements.lock']
    for start in roots:
        for path in ([start] if start.is_file() else sorted(start.rglob('*'))):
            if path.is_symlink():
                raise ValueError('Payload symlinks are forbidden: ' + str(path))
            if path.is_file() and '__pycache__' not in path.parts:
                if any(p == '.git' or p.startswith(('.env', '.venv')) for p in path.relative_to(root).parts):
                    raise ValueError('Private/runtime file in payload')
                yield path


def seal(root):
    root = Path(root).resolve()
    target = root / 'payload-manifest.json'
    if target.exists():
        raise ValueError('Release already sealed; create a new release')
    from experiments.baseline_cache import stage
    stage(root,ROOT/'lab/experiments/baseline-cache')
    files = {str(p.relative_to(root)): digest(p) for p in payload_files(root)}
    target.write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
    for relative in files:
        (root / relative).chmod(0o444)
    target.chmod(0o444)
    return digest(target)


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'payload-manifest.json').read_text())
    actual = {str(p.relative_to(root)): digest(p) for p in payload_files(root)}
    if actual != manifest['files']:
        raise ValueError('Immutable payload changed; refusing launch')
    return digest(root / 'payload-manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['seal', 'verify', 'run', 'watch'])
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--name')
    parser.add_argument('--entry')
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--max-cost-usd', type=float, default=None,
                        help='Optional fixed per-job envelope inside the shared ceiling')
    parser.add_argument('--gpu', default='NVIDIA GeForce RTX 4090')
    parser.add_argument('--gpu-count', type=int, default=1)
    parser.add_argument('--hardware-scope')
    parser.add_argument('--provider', choices=('runpod', 'vast'), default='runpod')
    parser.add_argument('--vast-offer-id', type=int)
    parser.add_argument('--vast-disk-gb', type=int, default=80)
    parser.add_argument('--max-runtime-hours', type=float)
    parser.add_argument('--completion-budget-scope')
    parser.add_argument('--data-upload-timeout', type=int, default=300)
    parser.add_argument('--direct-pinned-wheels', action='store_true')
    parser.add_argument('--resume-startup',action='store_true')
    args = parser.parse_args()
    if args.action in ('seal', 'verify'):
        print((seal if args.action == 'seal' else verify)(args.release))
        return
    source_hash = verify(args.release)
    from experiments import cloud
    canonical_state = (ROOT / 'lab/experiments/cloud').resolve()
    if cloud.STATE.resolve() != canonical_state:
        raise RuntimeError('Cloud ledger is not canonical')
    if args.action == 'watch':
        cloud.watch(args.name)
        return
    from experiments.baseline_cache import check_launch
    check_launch(args.release,args.entry,ROOT/'lab/experiments/baseline-cache',args.gpu)
    if not args.name or not args.entry or Path(args.entry).is_absolute() or '..' in Path(args.entry).parts:
        raise ValueError('Unique job name and relative entry required')
    # Only the upload origin changes. The imported ops client, state directory,
    # allocation locks, GPU policy and independent guard remain canonical.
    cloud.ROOT = args.release.resolve()
    upload_ceiling = 900
    auth_path = canonical_state/'authorizations'/(args.name+'.json')
    if args.hardware_scope in ('replay-evaluation-v1','coding-length-probe-v1','coding-replay-768-v1','opencoder-sft-768-v1','opencoder-mode-probe-v1','coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1') and auth_path.is_file():
        auth=json.loads(auth_path.read_text())
        if auth.get('authorized') is True and (auth.get('evaluation_only') is True or args.hardware_scope in ('coding-replay-768-v1','opencoder-sft-768-v1','opencoder-mode-probe-v1','coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1')) and auth.get('job_name')==args.name:
            upload_ceiling=1800
    if args.name.startswith('shorterbetter-scaled-parallel-') and auth_path.is_file():
        auth = json.loads(auth_path.read_text())
        if (auth.get('authorized') is True and auth.get('job_name') == args.name
                and auth.get('evaluation_only') is True and auth.get('direct_ssh') is True):
            upload_ceiling = 1800
    if not 300 <= args.data_upload_timeout <= upload_ceiling:
        raise ValueError(f'Data staging timeout must be between 300 and {upload_ceiling} seconds')
    original_sync = cloud.sync
    def sync(pod, local, remote_path, **kwargs):
        bootstrap=cloud.ROOT/'lab/experiments/data/replay-evaluation/uv-wheel-source.json'
        if (args.hardware_scope in ('replay-evaluation-v1','coding-length-probe-v1','coding-replay-768-v1','opencoder-sft-768-v1','opencoder-mode-probe-v1','coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1') and kwargs.get('upload')
                and str(remote_path)=='/workspace/project/uv' and bootstrap.is_file()):
            import shlex
            meta=json.loads(bootstrap.read_text())
            if not meta['url'].startswith('https://files.pythonhosted.org/'):raise ValueError('Untrusted bootstrap source')
            meta.update(binary_sha256=digest(local),mtime=Path(local).stat().st_mtime_ns)
            script="""import json,urllib.request,hashlib,zipfile,io,os
from pathlib import Path
m=json.loads(META)
with urllib.request.urlopen(m['url'],timeout=90) as r:data=r.read()
assert hashlib.sha256(data).hexdigest()==m['sha256']
with zipfile.ZipFile(io.BytesIO(data)) as z:
 names=[i.filename for i in z.infolist() if i.filename.endswith('/uv') and i.file_size>10000000];assert len(names)==1;binary=z.read(names[0])
assert hashlib.sha256(binary).hexdigest()==m['binary_sha256']
p=Path('/workspace/project/uv');p.write_bytes(binary);p.chmod(0o755);os.utime(p,ns=(m['mtime'],m['mtime']))
Path('/workspace/project/output/uv-bootstrap-verification.json').write_text(json.dumps(m,indent=2))
""".replace('META',repr(json.dumps(meta)))
            if args.hardware_scope in ('coding-length-probe-v1','coding-replay-768-v1','opencoder-sft-768-v1','opencoder-mode-probe-v1','coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1') or ((args.provider=='runpod' and args.name.startswith('replay-evaluation-1024-')) or args.name.startswith('replay-evaluation-1280-')):
                probe_uv_bootstrap(cloud,original_sync,pod,local,remote_path,script,meta,kwargs.get('timeout',300),retry_transport=args.hardware_scope in ('coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1'))
            else:
                cloud.remote(pod,'python3 -c '+shlex.quote(script),timeout=180)
            return
        if kwargs.get('upload') and Path(local).resolve() == (cloud.ROOT / 'lab/experiments/data').resolve():
            if (args.hardware_scope in ('coding-frontier-768-v1','coding-curriculum-256-v1') or (args.hardware_scope=='coding-abc-768-v1' and not json.loads((cloud.ROOT/'lab/experiments/data/coding-abc/plan.json').read_text()).get('final_official_evaluation'))):
                # Inference surveys and development-only pilots skip official cache uploads.
                kwargs['exclusions']=tuple(kwargs.get('exclusions',()))+('baseline-cache',)
            if args.direct_pinned_wheels:
                if args.hardware_scope != 'shorterbetter-scaled-v1':
                    raise ValueError('Direct wheel staging is scoped to ShorterBetter')
                from ops.shorterbetter_stage_existing import FETCH
                original_sync(pod, local, remote_path, **dict(kwargs, exclusions=('wheels',), timeout=args.data_upload_timeout))
                cloud.remote(pod, 'mkdir -p /opt/thinkingcap-wheels')
                original_sync(pod, Path(local)/'wheels/SHA256SUMS', '/opt/thinkingcap-wheels/SHA256SUMS', upload=True)
                helper = cloud.STATE/args.name/'fetch-pinned-wheels.py'
                helper.write_text(FETCH)
                original_sync(pod, helper, '/opt/fetch-pinned-wheels.py', upload=True)
                cloud.remote(pod, 'python3 /opt/fetch-pinned-wheels.py > /workspace/project/output/dependency-staging.log 2>&1', timeout=1800)
                cloud.remote(pod, 'ln -s /opt/thinkingcap-wheels /workspace/project/data/wheels')
                cloud.update(args.name, heartbeat=__import__('time').time())
                return
            kwargs['timeout'] = args.data_upload_timeout
        if args.hardware_scope in ('replay-scaled-a100-v1','coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1') and kwargs.get('upload'):
            return retry_upload(original_sync,pod,local,remote_path,**kwargs)
        return original_sync(pod, local, remote_path, **kwargs)
    cloud.sync = sync
    if args.hardware_scope in ('coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1') and args.vast_offer_id is None:
        # Requote inside the allocation lock; short-lived marketplace offers
        # otherwise disappear between payload verification and rental.
        original_quote=cloud.vastai.quote
        def frontier_quote(gpu,count,disk_gb=80,offer_id=None,min_duration=0,**kwargs):
            jobs=json.loads((canonical_state/'ledger.json').read_text())['jobs']
            excluded={j.get('provider_quote',{}).get('offer',{}).get('machine_id') for j in jobs.values()
                if j.get('hardware_scope') in ('coding-frontier-768-v1','coding-abc-768-v1','coding-curriculum-256-v1') and
                    (j.get('startup_failed_before_launch') or (j.get('released_at') and j.get('absence_reconciled_at')))}
            excluded.discard(None)
            return original_quote(gpu,count,disk_gb,offer_id,min_duration,exclude_machine_ids=excluded,
                allowed_countries=('US','CA','GB','DE','FR','NL','PL','FI','SE','CZ','ES','IT','AT','RO','LT','BG','IE'))
        cloud.vastai.quote=frontier_quote
    original_capacity_rejected = cloud.capacity_rejected
    def capacity_rejected(exc):
        message = str(exc).lower()
        return original_capacity_rejected(exc) or (
            isinstance(exc, RuntimeError) and 'runpod http 400:' in message
            and 'this machine does not have the resources to deploy your pod' in message)
    # A documented-by-observation explicit rejection still goes through the
    # canonical complete-list reconciliation before any retry.
    cloud.capacity_rejected = capacity_rejected
    args.release = str(args.release.resolve())
    print(json.dumps({'release': args.release, 'payload_sha256': source_hash,
                      'shared_state': str(canonical_state), 'job': args.name}), flush=True)
    cloud.run(args)


if __name__ == '__main__':
    main()
