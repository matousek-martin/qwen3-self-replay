"""Explicit 2026-11-05 cleanup: preserve science/results, prune bulky runtime data.

Plan first; apply only the exact inventoried files. Never touch model adapters,
baseline cache, source code, raw training/evaluation responses, or live jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREFIXES = ('shorterbetter', 'replay-scaled', 'replay-evaluation', 'coding-', 'opencoder-')
AUDIT = ROOT / 'lab/experiments/storage-cleanup/20261105-conversation'


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ledger_state():
    from experiments import cloud
    jobs = read(cloud.STATE / 'ledger.json')['jobs']
    return jobs


def active_roots(jobs):
    roots = []
    for name, job in jobs.items():
        if job.get('released_at'):
            continue
        roots += [ROOT / 'lab/experiments/cloud' / name, ROOT / 'lab/isolated' / name]
        auth = ROOT / 'lab/experiments/cloud/authorizations' / (name + '.json')
        if auth.exists():
            value = read(auth)
            if value.get('release'):
                roots.append(Path(value['release']).resolve())
            if value.get('campaign_path'):
                roots.append(Path(value['campaign_path']).resolve().parent)
    return roots


def checkpoint_roots():
    result = []
    for p in (ROOT / 'lab/experiments/cloud').glob('*/output/**/complete.json'):
        s = str(p.relative_to(ROOT))
        keep = (
            ('replay-scaled-training-20261101t122440z/output/training/checkpoints/' in s
             and p.parent.name in ('group-000256', 'group-000768', 'group-000850'))
            or 'replay-scaled-runpod-resume-20261102t162212z/output/training/checkpoints/group-001024/' in s
            or ('coding-abc-' in s and '/output/training/checkpoints/group-000109/' in s)
            or ('coding-curriculum-' in s and any('/output/' + a + '/checkpoints/group-000128/' in s
                                                 for a in ('kodcode', 'taco'))))
        if keep:
            result.append(p.parent)
    if len(result) != 9:
        raise RuntimeError('Expected nine explicitly protected complete checkpoints')
    return result


def classify(p, keep):
    rel = p.relative_to(ROOT)
    s = str(rel)
    if any(p.is_relative_to(k) for k in keep):
        return None
    if p.suffix == '.whl' and 'shorterbetter' in s:
        return 'redownloadable_shorterbetter_installers'
    scoped = ((s.startswith('lab/experiments/cloud/') and rel.parts[3].startswith(PREFIXES))
              or (s.startswith('lab/isolated/') and rel.parts[2].startswith(PREFIXES)))
    if scoped and p.suffix == '.pt' and p.stat().st_size > 1024 * 1024:
        return 'nonessential_optimizer_or_technical_test_state'
    if (s.startswith(('lab/experiments/coding-frontier/data-v1/',
                      'lab/experiments/coding-frontier/data-v2/'))
            and (p.suffix == '.jsonl' or p.name.endswith('.jsonl.gz'))):
        return 'superseded_frontier_preparation_data'
    if s.startswith('lab/experiments/coding-frontier/sources/') and p.suffix == '.parquet':
        return 'redownloadable_public_dataset_cache'
    if (s.startswith('lab/isolated/coding-frontier-taco-20261104t142707z')
            and p.name == 'questions.jsonl.gz' and p.stat().st_size > 1024 ** 3):
        return 'superseded_taco_payload_data'
    return None


def plan():
    if (AUDIT / 'plan.json').exists():
        raise RuntimeError('Plan exists; inspect it instead of overwriting')
    jobs = ledger_state()
    checkpoints = checkpoint_roots()
    keep = checkpoints + active_roots(jobs)
    protected = {}
    for root in checkpoints + [ROOT / 'lab/experiments/baseline-cache']:
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink():
                protected[str(p.relative_to(ROOT))] = digest(p)
    for root in checkpoints:
        m = read(root / 'complete.json')
        for name, h in m['files'].items():
            if digest(root / name) != h:
                raise RuntimeError('Protected checkpoint already incomplete: ' + str(root))
    files = []
    inodes = {}
    for directory, _, names in os.walk(ROOT / 'lab'):
        for name in names:
            p = Path(directory) / name
            if p.is_symlink() or not p.is_file():
                continue
            category = classify(p, keep)
            if not category:
                continue
            st = p.stat()
            if p.is_relative_to(ROOT / 'lab/experiments/cloud'):
                job = jobs.get(p.relative_to(ROOT).parts[3])
                if not job or not job.get('released_at'):
                    raise RuntimeError('Unreconciled cloud artifact: ' + str(p))
            key = (st.st_dev, st.st_ino)
            if key not in inodes:
                inodes[key] = digest(p)
            files.append(dict(path=str(p.relative_to(ROOT)), category=category,
                              size=st.st_size, blocks=st.st_blocks, device=st.st_dev,
                              inode=st.st_ino, links=st.st_nlink, mtime_ns=st.st_mtime_ns,
                              sha256=inodes[key]))
    groups = defaultdict(list)
    for f in files:
        groups[(f['device'], f['inode'])].append(f)
    freed = sum(v[0]['blocks'] * 512 for v in groups.values() if len(v) == v[0]['links'])
    obj = dict(created_at=time.time(), authorized_by='User requests cleanup of large files from experiments in this conversation, preserving smaller documentation.',
               protected_checkpoints=[str(p.relative_to(ROOT)) for p in checkpoints],
               active_roots_excluded=[str(p) for p in active_roots(jobs)],
               protected_sha256=protected, files=files, expected_reclaimed_bytes=freed,
               unique_selected_inodes=len(groups), policy='Keep every model adapter, code snapshot, config, metric, graph trace and raw response; remove only inventoried optimizer/test state, installers and superseded/public dataset caches.')
    write(AUDIT / 'plan.json', obj)
    print(json.dumps(dict(files=len(files), expected_reclaimed_gib=freed / 2**30,
                          categories=dict(Counter(f['category'] for f in files)))))


def apply():
    p = AUDIT / 'plan.json'
    obj = read(p)
    if (AUDIT / 'completion.json').exists():
        raise RuntimeError('Cleanup already applied')
    jobs = ledger_state()
    keep = checkpoint_roots() + active_roots(jobs)
    for f in obj['files']:
        path = ROOT / f['path']
        if f['path'].startswith('lab/experiments/cloud/'):
            job = jobs.get(Path(f['path']).parts[3], {})
            if not job.get('released_at'):
                raise RuntimeError('Cloud job is no longer released: ' + f['path'])
        st = path.lstat()
        if (path.is_symlink() or classify(path, keep) != f['category']
                or (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                != (f['device'], f['inode'], f['size'], f['mtime_ns'])):
            raise RuntimeError('Inventoried file changed: ' + str(path))
    for name, h in obj['protected_sha256'].items():
        if digest(ROOT / name) != h:
            raise RuntimeError('Protected input changed: ' + name)
    # Write a durable per-path removal record before each unlink. Never rewrite
    # historical successful backup or checksum records to claim a full archive.
    before = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    with (AUDIT / 'removals.jsonl').open('x') as log:
        for f in obj['files']:
            log.write(json.dumps(f) + '\n'); log.flush(); os.fsync(log.fileno())
            (ROOT / f['path']).unlink()
    assert all(not (ROOT / f['path']).exists() for f in obj['files'])
    for name, h in obj['protected_sha256'].items():
        if digest(ROOT / name) != h:
            raise RuntimeError('Protected input changed after cleanup: ' + name)
    affected = defaultdict(list)
    for f in obj['files']:
        rel = Path(f['path'])
        if str(rel).startswith('lab/experiments/cloud/'):
            affected[ROOT / Path(*rel.parts[:4])].append(f['path'])
        elif str(rel).startswith('lab/isolated/'):
            affected[ROOT / Path(*rel.parts[:3])].append(f['path'])
    for root, paths in affected.items():
        write(root / 'artifact-retention-20261105.json', dict(cleaned_at=time.time(),
              audit=str(p.relative_to(ROOT)), removed_files=paths,
              note='Historical hashes/backup attest to original archive. Listed files intentionally removed; do not assume full-state resume or sealed-payload reuse. Model adapters and measurements remain. See audit for nine retained complete checkpoints.'))
    after = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    result = dict(finished_at=time.time(), plan_sha256=digest(p), removed_files=len(obj['files']),
                  expected_reclaimed_bytes=obj['expected_reclaimed_bytes'], observed_free_bytes_change=after-before,
                  protected_files_verified=len(obj['protected_sha256']), affected_roots=len(affected))
    write(AUDIT / 'completion.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['plan', 'apply'])
    args = parser.parse_args()
    plan() if args.action == 'plan' else apply()
