"""Stage a new immutable evaluation release from verified final training outputs."""
import argparse
import json
from pathlib import Path
import shutil
import time

from isolated_cloud import ROOT, digest, seal
from experiments import cloud
from experiments.study_eval import check_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v1', 'v2'], default='v1')
    version = parser.parse_args().version
    release = ROOT/f'lab/isolated/full-evaluation-20261026/teacher-student-{version}'
    if release.exists():
        raise ValueError('Preserve existing release')
    original = ROOT/'lab/isolated/completion-20261026/evaluations-v2'
    for folder in ('experiments', 'ThinkingCap-Test'):
        shutil.copytree(original/folder, release/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '.git', '.env*', '.venv*', 'runs'))
    for src, dst in [(ROOT/'ops/teacher_full_eval_worker.py', release/'experiments/teacher_full_eval.py'),
                     (ROOT/'experiments/study_eval.py', release/'experiments/study_eval.py'),
                     (ROOT/'experiments/baseline_cache.py', release/'experiments/baseline_cache.py'),
                     (ROOT/'experiments/full_replay_eval.py', release/'experiments/full_replay_eval.py')]:
        shutil.copyfile(src, dst)
    data = release/'lab/experiments/data'
    data.mkdir(parents=True)
    shutil.copyfile(original/'lab/experiments/data/pilot.jsonl', data/'pilot.jsonl')
    shutil.copyfile(original/'lab/gpu-requirements.lock', release/'lab/gpu-requirements.lock')
    campaign = json.loads((ROOT/'lab/experiments/teacher-completion-v2/campaign.json').read_text())
    with cloud.ledger() as ledger:
        jobs = ledger['jobs']
        liability = cloud.liability(jobs)
        if liability + 3*6*1.3 > cloud.USABLE:
            raise ValueError('Full three-seed evaluation exceeds remaining shared budget')
    stage = data/'teacher-full';stage.mkdir()
    adapters = {}
    for seed in (17, 29, 43):
        trial = campaign['trials'][str(seed)];name = trial['job'];job = jobs[name]
        if (trial['state'] != 'complete' or job['state'] != 'complete'
                or not job.get('backup_verified') or not job.get('released_at')):
            raise ValueError('Training archive incomplete: ' + name)
        if cloud.verify_backup(name)['exit_code'] != 0:
            raise ValueError('Training backup verification failed')
        output = cloud.STATE/name/'output'
        for arm in ('student', 'teacher'):
            label = f'{arm}-seed{seed}';source = output/'training'/label/'adapter'
            hashes = check_adapter(source)
            target = stage/'adapters'/label
            shutil.copytree(source, target)
            adapters[label] = dict(path=f'adapters/{label}', files_sha256=hashes,
                original_job=name, original_adapter=str(source), training_gpu='NVIDIA GeForce RTX 4090',
                training_config_sha256=digest(source.parent/'config.json'))
            (stage/f'smoke-{label}.json').write_text(json.dumps({'candidates': ['base', label]}))
        entry = release/f'experiments/run-isolated-seed{seed}.sh'
        entry.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'source /workspace/project/experiments/setup_gpu.sh\n'
            'export NLTK_DATA=/root/thinkingcap_nltk_data\n'
            'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1\n'
            f'python -m experiments.teacher_full_eval --config data/teacher-full/evaluation.json --seed {seed} --out output/full-evaluation\n')
    repo = release/'ThinkingCap-Test'
    paths = [repo/'evaluate.py', repo/'compare.py'] + sorted((repo/'utils/eval').rglob('*.py'))
    config = dict(id=f'teacher-full-{version}', seeds=[17,29,43], hours=6, evaluation_only=True,
        samples=8, questions=2759, responses_per_model=22072, evaluation_seed=9001,
        cohort_sha256=digest(repo/'questions/questions.jsonl'),
        questions_manifest_sha256=digest(repo/'questions/manifest.json'),
        evaluator_files_sha256={str(p.relative_to(repo)):digest(p) for p in paths},
        adapters=adapters, fresh_holdout=False, same_worker_baseline=False,
        baseline_policy='gpu-model-and-protocol-cache-v1',
        seed_selection='all three training seeds; no selection by prior accuracy or length',
        training_performed=False, ceiling_usd=cloud.CEILING,
        conservative_liability_before_launch=liability, initial_total_reservation_usd=23.4,
        authorized_at=time.time(), authorization='spust prosim plnou evaluaci pro studenta i teachra',
        notifications='server log only', competition_rules_source='https://github.com/BottleCapAI/ThinkingCap-Test#rules',
        benchmark_policy='Evaluation only; no benchmark examples or transformations used for training or teacher generation.')
    (stage/'evaluation.json').write_text(json.dumps(config, indent=2)+'\n')
    (data/'isolated').mkdir()
    (data/'isolated/config.json').write_text(json.dumps(config, indent=2)+'\n')
    payload = seal(release)
    registration = ROOT/('lab/experiments/teacher-full-20261026' + ('' if version == 'v1' else '-'+version))
    registration.mkdir(parents=True, exist_ok=True)
    cloud.write(registration/'registration.json', dict(config, release=str(release), payload_sha256=payload))
    print(json.dumps({'release':str(release), 'payload_sha256':payload, 'registration':str(registration)}))


if __name__ == '__main__':
    main()
