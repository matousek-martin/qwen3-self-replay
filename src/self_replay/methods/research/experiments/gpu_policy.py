"""Fail before computation if the worker violates the user's GPU requirement."""
import subprocess
import json
from pathlib import Path

REQUIRED_GPU = 'NVIDIA GeForce RTX 4090'


def verify_gpu_names(output):
    names = [line.strip() for line in output.splitlines() if line.strip()]
    if not names or any(name != REQUIRED_GPU for name in names):
        raise RuntimeError(f'User requires {REQUIRED_GPU} only; observed {names!r}')
    return names


def main():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    lease_path = Path('worker-lease.json')
    lease = json.loads(lease_path.read_text()) if lease_path.exists() else {}
    if lease.get('hardware_scope'):
        from experiments.shorterbetter.hardware import verify_worker
        names = verify_worker([n.strip() for n in output.splitlines() if n.strip()], lease)
    else:
        names = verify_gpu_names(output)
    print('GPU requirement verified: ' + ', '.join(names), flush=True)


if __name__ == '__main__':
    main()
