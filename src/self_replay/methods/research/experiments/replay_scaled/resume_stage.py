"""Restore an immutable, fully verified continuation bundle before worker startup."""
import json
import shutil
import stat
from pathlib import Path
from .core import sha, write

def stage(data=Path('data/replay-scaled'), output=Path('output/training')):
    mode=json.loads((data/'launch-mode.json').read_text())
    group=mode.get('resume_group')
    if group not in (850,1024) or (group==1024 and mode.get('stop_after')!=1280):
        raise ValueError('Unregistered resume')
    source=data/'resume-training'
    hashes=json.loads((data/'resume-files.json').read_text())
    actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
    if actual!=hashes:raise ValueError('Resume bundle changed')
    manifest_path=source/f'checkpoints/group-{group:06d}/complete.json'
    manifest=json.loads(manifest_path.read_text())
    if (manifest['group']!=group or manifest['world_size']!=2 or manifest['pilot']
            or manifest['contract']!=sha(data/'plan.json')):raise ValueError('Resume contract mismatch')
    if output.exists():raise ValueError('Never overwrite existing training state')
    shutil.copytree(source,output)
    # Sealed input files are0444. Runtime copies must allow append/update on
    # RunPod's mounted filesystem, which enforces these bits even for root.
    for path in [output,*output.rglob('*')]:
        path.chmod(path.stat().st_mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
    for name in ('training-records.jsonl','timing.jsonl','evaluation-timing.jsonl'):
        with (output/name).open('a'):
            pass  # Fail before paid model inference if the runtime mount rejects writes.
    write(Path('output/resume-verification.json'),dict(group=group,files_verified=len(hashes),
          manifest_sha256=sha(manifest_path),world_size=2))

if __name__=='__main__':stage()
