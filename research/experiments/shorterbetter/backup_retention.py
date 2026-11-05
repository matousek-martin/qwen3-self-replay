"""Bound a full-run mirror after successful sync, preserving verified successors."""
import hashlib
import json
from pathlib import Path
import re
import shutil
from .distributed import atomic_json


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prune(output, remote_checkpoint):
    output = Path(output)
    removed = []
    technical = output/'technical-retention.json'
    if technical.exists():
        for relative, expected in json.loads(technical.read_text())['removed_artifact_sha256'].items():
            path = output/relative
            if (not path.resolve().is_relative_to(output.resolve())
                    or Path(relative).parts[0] not in ('verification-reference','verification-resumed')):
                raise ValueError('Unregistered technical retention path')
            if path.is_file():
                if digest(path) != expected:
                    raise RuntimeError('Technical backup changed before compaction')
                path.unlink()
                removed.append(relative)
    if remote_checkpoint:
        name, hashes = remote_checkpoint['name'], remote_checkpoint['hashes']
        if not re.fullmatch(r'checkpoint-\d+', name):
            raise ValueError('Invalid checkpoint name')
        required = {'model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json',
                    *(f'rng_state_{i}.pth' for i in range(4))}
        checkpoint = output/'training'/name
        if (set(hashes) == required and all((checkpoint/n).is_file() for n in required)
                and json.loads((checkpoint/'trainer_state.json').read_text())['global_step'] == int(name[11:])
                and all(digest(checkpoint/n) == hashes[n] for n in required)):
            for older in (output/'training').glob('checkpoint-*'):
                if re.fullmatch(r'checkpoint-\d+', older.name) and int(older.name[11:]) < int(name[11:]):
                    shutil.rmtree(older)
                    removed.append(str(older.relative_to(output)))
    if removed:
        atomic_json(output.parent/'backup-retention.json', {'removed':removed,
                    'verified_successor':remote_checkpoint, 'policy':'latest complete checkpoint plus best and final export'})
    return removed


def maintain(cloud, pod, output):
    checkpoints = list((output/'training').glob('checkpoint-*'))
    snapshot = None
    if len(checkpoints) > 1:
        code = """import hashlib,json
from pathlib import Path
base=Path('/workspace/project/output/training')
paths=[p for p in base.glob('checkpoint-*') if p.name[11:].isdigit()]
result=None
if paths:
 p=max(paths,key=lambda p:int(p.name[11:]))
 names=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(4)]
 if all((p/n).is_file() for n in names):
  hashes={}
  for n in names:
   h=hashlib.sha256()
   with (p/n).open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
   hashes[n]=h.hexdigest()
  result={'name':p.name,'hashes':hashes}
print(json.dumps(result))
"""
        snapshot = json.loads(cloud.remote(pod, "python3 - <<'PY'\n"+code+"\nPY", timeout=120))
    return prune(output, snapshot)
