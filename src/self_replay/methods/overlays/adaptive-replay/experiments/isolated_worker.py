"""One isolated paired worker; GPU lifecycle belongs to the canonical controller."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from .gpu_policy import verify_gpu_names
from .study_eval import write_json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_command(config, arm, seed, data, output):
    values = dict(config['training'], **arm['overrides'], seed=seed,
                  out=str(output / 'training' / f"{arm['name']}-seed{seed}"))
    for key in ('data', 'references', 'static_replay', 'distill_cache'):
        if values.get(key):
            values[key] = str(data / values[key])
    cmd = [sys.executable, '-m', 'experiments.rl_training.train']
    for key, value in values.items():
        if value is not None:
            cmd += ['--' + key.replace('_', '-'), str(value)]
    return cmd


def run(*args):
    command = [sys.executable, *map(str, args)]
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def worker(seed):
    data, output = ROOT / 'data', ROOT / 'output'
    config_path = data / 'isolated/config.json'
    config = json.loads(config_path.read_text())
    if seed not in config['seeds']:
        raise ValueError('Unregistered training seed')
    for relative, expected in json.loads((data / 'isolated/inputs.json').read_text()).items():
        if sha(data / relative) != expected:
            raise ValueError('Input changed: ' + relative)
    for relative, expected in json.loads((data / 'isolated/sources.json').read_text()).items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Source changed: ' + relative)
    verify_gpu_names(subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True))
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip()
    output.mkdir(exist_ok=True)
    write_json(output / 'isolation-provenance.json', {'config': config, 'config_sha256': sha(config_path),
               'gpu': gpu, 'seed': seed, 'sources': json.loads((data / 'isolated/sources.json').read_text())})
    run('-m', 'unittest', *config['worker_tests'], '-v')
    if config.get('prepare_module'):
        run('-m', config['prepare_module'], '--config', config_path, '--out', output / 'preparation')
    stage = output / 'evaluation-inputs'
    states, names = {}, ['base']
    for arm in config['arms']:
        name = f"{arm['name']}-seed{seed}"
        command = training_command(config, arm, seed, data, output)
        subprocess.run(command, cwd=ROOT, check=True)
        target = output / 'training' / name
        status = json.loads((target / 'status.json').read_text())
        if status['status'] != 'completed' or status['completed_groups'] != config['training']['max_steps'] or status['optimizer_updates'] < 1:
            raise RuntimeError('Registered training budget incomplete')
        resolved = json.loads((target / 'config.json').read_text())
        for key, value in {**config['training'], **arm['overrides'], 'seed': seed}.items():
            if key not in ('data', 'references', 'static_replay', 'distill_cache') and value is not None and resolved.get(key) != value:
                raise RuntimeError('Resolved training recipe mismatch: ' + key)
        if resolved['training_gpu'] != 'NVIDIA GeForce RTX 4090':
            raise RuntimeError('Training GPU provenance mismatch')
        states[name] = status
        if config.get('validation', True):
            from .pilot import validate
            validate(name, target / 'adapter')
        shutil.copytree(target / 'adapter', stage / 'adapters' / name)
        names.append(name)
    shutil.copyfile(data / 'study/splits.json', stage / 'splits.json')
    evaluation = dict(config['evaluation'], candidates=names)
    write_json(stage / 'evaluation.json', evaluation)
    run('-m', 'experiments.vllm_compat', '--out', output / 'vllm-compat.json')
    run('-m', 'experiments.vllm_lora_smoke', '--config', stage / 'evaluation.json', '--out', output / 'vllm-lora-smoke.json')
    run('-m', 'experiments.study_eval', '--config', stage / 'evaluation.json', '--out', output / 'study')
    result = json.loads((output / 'study/study-result.json').read_text())
    if not result['complete'] or set(result['completed_candidates']) != set(names):
        raise RuntimeError('Paired evaluation incomplete')
    for name in names[2:]:
        run(ROOT / 'ThinkingCap-Test/compare.py', output / 'study/runs' / names[1],
            output / 'study/runs' / name, '--out', output / f'{name}-vs-{names[1]}.json')
    write_json(output / 'role-result.json', {'seed': seed, 'training': states, 'development_complete': True,
               'gpu_identity': gpu, 'config_sha256': sha(config_path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    worker(parser.parse_args().seed)
