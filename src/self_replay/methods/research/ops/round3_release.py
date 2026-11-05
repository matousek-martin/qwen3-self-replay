"""Create immutable isolated round3 workers from the verified prior training release."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from isolated_cloud import ROOT, digest, verify
from teacher_pipeline import lock_inputs_and_sources

STATE = ROOT / 'lab/experiments/teacher-round3-20261026'
BASE = ROOT / 'lab/isolated/teacher-round3-20261026'
TEMPLATE = ROOT / 'lab/isolated/completion-20261026/teacher-matched-v2'
ARMS = ['student', 'qwen3_32b', 'qwen38_27b']


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def skeleton(path):
    verify(TEMPLATE)
    path.mkdir(parents=True)
    for name in ('experiments', 'ThinkingCap-Test'):
        shutil.copytree(TEMPLATE / name, path / name,
                        ignore=shutil.ignore_patterns('__pycache__', '.git', '.venv*', '.env*'))
    for p in path.rglob('*'):
        if p.is_file(): p.chmod(0o644)
    (path / 'lab').mkdir(exist_ok=True)
    shutil.copy2(TEMPLATE / 'lab/gpu-requirements.lock', path / 'lab/gpu-requirements.lock')
    shutil.copy2(ROOT / 'ops/round3_gpu.py', path / 'experiments/round3_gpu.py')
    (path / 'lab/experiments/data/isolated').mkdir(parents=True)


def entry(path, name, command):
    (path / 'experiments' / name).write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
        'source /workspace/project/experiments/setup_gpu.sh\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n'
        + command + '\n')


def sampling(index, cohort):
    path = BASE / f'sampling-v1-batch{index}'
    if path.exists():
        verify(path)
        expected = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in cohort)
        if (path / 'lab/experiments/data/candidates.jsonl').read_text() != expected:
            raise ValueError('Existing sampling cohort changed')
        return path
    skeleton(path); data = path / 'lab/experiments/data'
    text = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in cohort)
    (data / 'candidates.jsonl').write_text(text)
    # Canonical upload preflight requires a pilot file, never trains in sampling mode.
    (data / 'pilot.jsonl').write_text(text)
    put(data / 'isolated/config.json', {'mode': 'sampling', 'batch': index, 'samples_per_question': 8})
    entry(path, 'run-round3-sample.sh', 'python -m experiments.round3_gpu sample')
    lock_inputs_and_sources(path)
    return path


def training():
    path = BASE / 'training-v2'
    if path.exists(): verify(path); return path
    skeleton(path); data = path / 'lab/experiments/data'
    for name, source in [('pilot.jsonl', 'rl-stream.jsonl'), ('development.jsonl', 'development.jsonl')]:
        shutil.copy2(STATE / source, data / name)
    shutil.copytree(STATE / 'static-v2', data / 'static')
    shutil.copy2(STATE / 'references.json', data / 'references.json')
    train = path / 'experiments/rl_training/train.py'
    text = train.read_text()
    old = 'parser.add_argument("--static-arm", choices=("student", "teacher"))'
    assert text.count(old) == 1
    train.write_text(text.replace(old, 'parser.add_argument("--static-arm", choices=("student", "qwen3_32b", "qwen38_27b"))'))
    static = path / 'experiments/rl_training/static_replay.py'
    text = static.read_text()
    text = text.replace("if arm not in ('student','teacher'):", "if arm not in ('student','qwen3_32b','qwen38_27b'):")
    text = text.replace("len(self.schedule)!=128", "len(self.schedule)!=512")
    text = text.replace('import hashlib', 'from .replay_format import validate_tokens\nimport hashlib', 1)
    anchor = '            self.targets[r[\'prompt_id\']]=(prompt,completion)'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '            validate_tokens(tokenizer, prompt, completion)\n' + anchor)
    static.write_text(text)
    shutil.copy2(ROOT / 'ops/round3_format.py', static.parent / 'replay_format.py')
    config = json.loads((TEMPLATE / 'lab/experiments/data/isolated/config.json').read_text())
    config.update(id='teacher-round3-v2', seeds=[17,29,43], hours=6, arms=ARMS)
    config['training'].update(max_steps=512, save_every=128, replay_capacity=256)
    for key in ('protocol_sha256', 'pool_manifest_sha256', 'primary_evaluation', 'sampling_extension_sha256',
                'protocol_version', 'cohort_extension_sha256', 'authorization'):
        config.pop(key, None)
    config.update(protocol_sha256=digest(STATE / 'execution-protocol-v2.json'),
                  pool_manifest_sha256=digest(data / 'static/manifest.json'),
                  authorization=json.loads((STATE / 'authorization.json').read_text()))
    put(data / 'isolated/config.json', config)
    for seed in config['seeds']:
        entry(path, f'run-isolated-seed{seed}.sh', f'python -m experiments.round3_gpu train --seed {seed}')
    lock_inputs_and_sources(path)
    return path


def preflight(path):
    """Load every sealed target through the exact worker loader before renting GPUs."""
    verify(path)
    script = '''
import json
from pathlib import Path
from transformers import AutoTokenizer
from experiments.rl_training.static_replay import StaticReplay
from experiments.rl_training.train import MODEL, REVISION, verifier
t=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
root=Path.cwd(); manifest=root/'lab/experiments/data/static/manifest.json'
config=json.loads((root/'lab/experiments/data/isolated/config.json').read_text())
counts={}
for seed in config['seeds']:
    schedules=[]
    for arm in config['arms']:
        pool=StaticReplay(manifest,arm,seed,t,verifier())
        assert len(pool.targets)==256 and len(pool.schedule)==512
        assert len({pool.sample(i)[0] for i in range(512)})==256
        counts[f'{seed}:{arm}']=len(pool.targets)
        schedules.append(pool.schedule)
    assert all(s==schedules[0] for s in schedules)
print(json.dumps({'passed':True,'target_counts':counts,'schedule_length':512,
                  'scope':'Exact sealed CPU worker loader; no GPU training claimed'}))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=path,
                            check=True, capture_output=True, text=True)
    proof = json.loads(result.stdout)
    proof['payload_sha256'] = verify(path)
    put(STATE / 'static-preflight-v2.json', proof)
