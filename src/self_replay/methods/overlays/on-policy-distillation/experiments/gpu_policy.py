"""Fail before computation if the worker violates the user's GPU requirement."""
import subprocess

REQUIRED_GPU = 'NVIDIA GeForce RTX 4090'


def verify_gpu_names(output):
    names = [line.strip() for line in output.splitlines() if line.strip()]
    if not names or any(name != REQUIRED_GPU for name in names):
        raise RuntimeError(f'User requires {REQUIRED_GPU} only; observed {names!r}')
    return names


def main():
    names = verify_gpu_names(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True))
    print('GPU requirement verified: ' + ', '.join(names), flush=True)


if __name__ == '__main__':
    main()
