"""Render registered replay-selection commands, without launching any jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'lab/experiments/replay-selection-v1/protocol.json'


def load_protocol(path=None, worker=False):
    path = Path(path) if path else PROTOCOL
    registration = json.loads(path.with_name('registration.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != registration['protocol_sha256']:
        raise ValueError('Protocol changed after registration; create a new version')
    protocol = json.loads(path.read_text())
    for relative, expected in protocol['input_sha256'].items():
        if worker:
            relative = relative.replace('lab/experiments/data/', 'data/', 1)
            if relative == 'lab/gpu-requirements.lock':
                relative = 'requirements.lock'
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Registered input changed: ' + relative)
    if protocol['arms'] != {'control': 'all_correct', 'mixed': 'verified_correct'}:
        raise ValueError('Unexpected admission arms')
    return protocol


def training_command(protocol, arm, seed, data_root, output_root):
    if arm not in protocol['arms'] or seed not in protocol['training_seeds']:
        raise ValueError('Unregistered arm or seed')
    args = {**protocol['training'], 'seed': seed,
            'replay_admission': protocol['arms'][arm],
            'data': str(Path(data_root) / 'pilot.jsonl'),
            'references': str(Path(data_root) / 'shared/references.json'),
            'out': str(Path(output_root) / f'{arm}-seed{seed}')}
    command = [sys.executable, '-m', 'experiments.rl_training.train']
    for key, value in args.items():
        command.extend(['--' + key.replace('_', '-'), str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(ROOT / 'lab/experiments/data'))
    parser.add_argument('--output-root', default=str(ROOT / 'lab/experiments/replay-selection-v1/training'))
    args = parser.parse_args()
    protocol = load_protocol()
    for seed in protocol['training_seeds']:
        for arm in protocol['arms']:
            print(shlex.join(training_command(protocol, arm, seed, args.data_root, args.output_root)))


if __name__ == '__main__':
    main()
