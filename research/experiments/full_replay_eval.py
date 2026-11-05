"""Full official evaluation of the original locked replay adapter; no training."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from . import study_eval as study

COUNTS = {'mbpp': 378, 'svamp': 1000, 'ifeval': 541,
          'bfcl_simple_python': 400, 'bfcl_multiple': 200, 'bfcl_irrelevance': 240}


def inputs(config_path):
    config = json.loads(config_path.read_text())
    group = config.get('replay_scaled_checkpoint')
    if group is not None and (type(group) is not int or group not in (128,256,576,768,850,1024,1280)):
        raise ValueError('Unregistered replay checkpoint')
    candidate = f'replay-{group}' if group is not None else 'rl-replay'
    if config.get('coding_abc_branch') is not None:
        if (config['coding_abc_branch']!='expanded-v2' or group is not None or config.get('arm') not in ('a','b','c')
            or config.get('parent_group')!=768 or not 0<config.get('training_groups',0)<=config.get('target_groups',0)<=128
            or config.get('evaluation_only') is not True or config.get('training_authorized') is not False
            or config.get('baseline_cache_required') is not True or study.evaluator_hashes()!=config.get('evaluator_files_sha256')
            or config.get('coding_replay_branch') is not None or config.get('opencoder_sft_branch') is not None):
            raise ValueError('Unregistered expandedABC evaluation')
        candidate=f"coding-abc-{config['arm']}-{config['training_groups']}"
    if config.get('coding_replay_branch') is not None:
        if (config['coding_replay_branch']!='coding-replay-768-v1' or group is not None
                or config.get('parent_group')!=768 or config.get('training_groups')!=128
                or config.get('evaluation_only') is not True or config.get('training_authorized') is not False
                or config.get('baseline_cache_required') is not True):
            raise ValueError('Unregistered coding replay evaluation')
        candidate='coding-replay-768-128'
        if study.evaluator_hashes()!=config.get('evaluator_files_sha256'):
            raise ValueError('Frozen coding evaluator changed')
    if config.get('opencoder_sft_branch') is not None:
        if (config['opencoder_sft_branch']!='opencoder-sft-768-v1' or group is not None
                or config.get('coding_replay_branch') is not None or config.get('parent_group')!=768
                or config.get('optimizer_steps')!=84 or config.get('evaluation_only') is not True
                or config.get('training_authorized') is not False or config.get('baseline_cache_required') is not True
                or study.evaluator_hashes()!=config.get('evaluator_files_sha256')):
            raise ValueError('Unregistered OpenCoder SFT evaluation')
        candidate='opencoder-sft-768-84'
    if group is not None and (config.get('evaluation_only') is not True or config.get('training_authorized') is not False):
        raise ValueError('Checkpoint comparison is evaluation-only')
    if group is not None and study.evaluator_hashes()!=config.get('evaluator_files_sha256'):
        raise ValueError('Frozen checkpoint evaluator changed')
    if config['candidates'] != ['base', candidate] or config['samples'] != 8:
        raise ValueError('Full replay evaluation requires baseline/replay and eight samples')
    root = config_path.parent
    split_path = root / 'splits.json'
    source = study.REPO / 'questions'
    # Verify the original frozen cohort and its partition before using all rows.
    study.load_selection(source, split_path, 'confirmation')
    rows = study.read_jsonl(source / 'questions.jsonl')
    if dict(Counter(row['dataset'] for row in rows)) != COUNTS:
        raise ValueError('Expected all 2759 official questions')
    adapter = root / 'adapters' / candidate
    hashes = study.check_adapter(adapter)
    lock = study.check_lock(root / 'candidate-lock.json', candidate, hashes,
                            study.sha256(split_path))
    return config, source, rows, adapter, hashes, lock


def main():
    signal.signal(signal.SIGTERM, study.interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config, source, rows, adapter, hashes, lock = inputs(args.config.resolve())
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        'schema_version': 1, 'scope': 'full official benchmark; previously examined cohort, not fresh holdout',
        'full_contest_evaluation': True, 'fresh_holdout': False,
        'model': study.BASE_MODEL, 'revision': study.BASE_REVISION,
        'cohort_sha256': study.sha256(source / 'questions.jsonl'),
        'questions': len(rows), 'samples': 8, 'responses_per_model': len(rows) * 8,
        'seed': 9001, 'sampling': study.SAMPLING, 'candidate_lock': lock,
        'adapter_files_sha256': hashes, 'config': config,
        'evaluator_files_sha256': study.evaluator_hashes(),
        'runner_sha256': study.sha256(__file__),
        'matched_baseline_regenerated': None, 'same_physical_worker_required': False,
        'training_provenance': config['training_provenance'],
    }
    study.write_json(output / 'provenance.json', provenance)
    shutil.copyfile(source / 'manifest.json', output / 'questions-manifest.json')
    if args.check_only:
        print(json.dumps({'status': 'inputs_verified', 'questions': len(rows), 'samples': 8}))
        return
    study.preflight_nltk()
    gpu_command = ['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader']
    identity = subprocess.check_output(gpu_command, text=True).strip()
    study.write_json(output / 'worker.json', {'gpu_identity': identity, 'started_at': study.now()})
    for name in config['candidates']:
        if subprocess.check_output(gpu_command, text=True).strip() != identity:
            raise RuntimeError('Physical GPU changed between evaluations')
        if study.check_adapter(adapter) != hashes:
            raise ValueError('Locked adapter changed')
        run = output / 'runs' / name
        if run.exists():
            raise RuntimeError('Use a fresh output directory; partial runs are preserved')
        status = {'state': 'running', 'started_at': study.now(), 'gpu_identity': identity}
        study.write_json(output / 'status' / (name + '.json'), status)
        command = [sys.executable, study.REPO / 'evaluate.py', '--questions', source,
                   '--out', run, '--samples', 8, '--seed', 9001, '--workers', 4]
        if name != 'base':
            command += ['--adapter', adapter]
        study.run_process(command, output / 'logs' / (name + '.log'), None)
        result = study.validate_records(run, rows, 8, provenance['cohort_sha256'], 'full')
        if result['selection'] != 'full' or bool(result.get('adapter')) != (name != 'base'):
            raise ValueError('Incorrect selection or adapter identity')
        status.update(state='complete', finished_at=study.now(), responses=len(rows) * 8,
                      records_sha256=study.sha256(run / 'records.jsonl'))
        if name=='base':
            from .baseline_cache import status_fields
            status.update(status_fields(run))
        study.write_json(output / 'status' / (name + '.json'), status)
    if study.check_adapter(adapter) != hashes:
        raise ValueError('Locked adapter changed during evaluation')
    if subprocess.check_output(gpu_command, text=True).strip() != identity:
        raise RuntimeError('Physical GPU changed during evaluation')
    comparison = output / 'comparison.json'
    study.run_process([sys.executable, study.REPO / 'compare.py', output / 'runs/base',
                       output / 'runs' / config['candidates'][1], '--out', comparison],
                      output / 'logs/compare.log', 600)
    result = json.loads(comparison.read_text())
    if result['ungraded_benchmarks'] or any(result['full_evaluation'].values()):
        raise ValueError('Official comparator rejected full evaluation')
    study.write_json(output / 'completion.json', {'complete': True, 'finished_at': study.now(),
                     'responses_per_model': len(rows) * 8,
                     'comparison_sha256': study.sha256(comparison)})


if __name__ == '__main__':
    main()
