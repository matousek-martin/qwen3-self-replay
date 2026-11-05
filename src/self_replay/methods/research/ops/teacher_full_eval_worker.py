"""Full official evaluation of frozen student/teacher replay adapters; no training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import subprocess
import sys

from experiments import study_eval as study
from experiments.full_replay_eval import COUNTS


def inputs(config_path, seed):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config['evaluation_only'] is not True or config['samples'] != 8 or seed not in config['seeds']:
        raise ValueError('Expected registered evaluation-only seed and eight samples')
    data = config_path.parent
    source = study.REPO / 'questions'
    rows = study.read_jsonl(source / 'questions.jsonl')
    if (dict(Counter(r['dataset'] for r in rows)) != COUNTS
            or len({(r['dataset'], r['id']) for r in rows}) != 2759
            or study.sha256(source / 'questions.jsonl') != config['cohort_sha256']
            or study.sha256(source / 'manifest.json') != config['questions_manifest_sha256']):
        raise ValueError('Frozen full benchmark differs')
    if study.evaluator_hashes() != config['evaluator_files_sha256']:
        raise ValueError('Pinned evaluator changed')
    adapters = {}
    for arm in ('student', 'teacher'):
        name = f'{arm}-seed{seed}'
        meta = config['adapters'][name]
        path = study.inside(data, meta['path'])
        if study.check_adapter(path) != meta['files_sha256']:
            raise ValueError('Frozen adapter changed: ' + name)
        adapters[name] = path
    return config, source, rows, adapters


def compare(left, right, out):
    destination = out / f'{right}-vs-{left}.json'
    study.run_process([sys.executable, study.REPO / 'compare.py', out/'runs'/left,
                       out/'runs'/right, '--out', destination], out/'logs'/f'{right}-vs-{left}.log', 600)
    result = json.loads(destination.read_text())
    if result['ungraded_benchmarks'] or any(result['full_evaluation'].values()):
        raise ValueError('Official comparator rejected full evaluation')
    return study.sha256(destination)


def main():
    signal.signal(signal.SIGTERM, study.interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    # Child evaluator processes run from ThinkingCap-Test, not the project root.
    args.config = args.config.resolve()
    config, source, rows, adapters = inputs(args.config, args.seed)
    if args.check_only:
        print(json.dumps({'inputs_verified': True, 'seed': args.seed, 'questions': len(rows),
                          'samples': 8, 'candidates': list(adapters)}))
        return
    out = args.out.resolve()
    if out.exists():
        raise ValueError('Preserve existing output; no implicit evaluation restart')
    out.mkdir(parents=True)
    study.preflight_nltk()
    gpu_command = ['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']
    identity = subprocess.check_output(gpu_command, text=True).strip()
    provenance = dict(config, training_seed=args.seed, gpu_identity=identity,
                      matched_baseline_regenerated=None, same_physical_worker_required=False, fresh_holdout=False,
                      runner_sha256=study.sha256(__file__), started_at=study.now())
    study.write_json(out/'provenance.json', provenance)
    study.run_process([sys.executable, study.ROOT/'experiments/vllm_compat.py', '--out', out/'vllm-compat.json'],
                      out/'logs/compat.log', 600)
    for name in adapters:
        smoke_config = args.config.parent/f'smoke-{name}.json'
        study.run_process([sys.executable, study.ROOT/'experiments/vllm_lora_smoke.py', '--config', smoke_config,
                           '--out', out/f'{name}-routing.json'], out/'logs'/f'{name}-routing.log', 600)
    study.run_process([sys.executable, study.REPO/'evaluate.py', '--questions', source,
                       '--out', out/'grader-smoke', '--smoke', '--workers', 4],
                      out/'logs/grader-smoke.log', None)
    for name in ['base', *adapters]:
        inputs(args.config, args.seed)
        if subprocess.check_output(gpu_command, text=True).strip() != identity:
            raise RuntimeError('Physical GPU changed between models')
        status = dict(state='running', started_at=study.now(), gpu_identity=identity)
        study.write_json(out/'status'/f'{name}.json', status)
        run = out/'runs'/name
        command = [sys.executable, study.REPO/'evaluate.py', '--questions', source,
                   '--out', run, '--samples', 8, '--seed', 9001, '--workers', 4]
        if name != 'base':
            command += ['--adapter', adapters[name]]
        study.run_process(command, out/'logs'/f'{name}.log', None)
        result = study.validate_records(run, rows, 8, config['cohort_sha256'], 'full')
        if result['selection'] != 'full' or bool(result.get('adapter')) != (name != 'base'):
            raise ValueError('Unexpected model or evaluation subset')
        if name != 'base' and Path(result['adapter']).resolve() != adapters[name].resolve():
            raise ValueError('Evaluator used another adapter')
        status.update(state='complete', finished_at=study.now(), responses=22072,
                      records_sha256=study.sha256(run/'records.jsonl'))
        if name=='base':
            from experiments.baseline_cache import status_fields
            status.update(status_fields(run))
        study.write_json(out/'status'/f'{name}.json', status)
    inputs(args.config, args.seed)
    if subprocess.check_output(gpu_command, text=True).strip() != identity:
        raise RuntimeError('Physical GPU changed during evaluation')
    student, teacher = adapters
    comparisons = {f'{right}-vs-{left}': compare(left, right, out)
                   for left, right in [('base', student), ('base', teacher), (student, teacher)]}
    study.write_json(out.parent/'role-result.json', dict(development_complete=True,
        full_evaluation_complete=True, evaluation_only=True, seed=args.seed,
        gpu_identity=identity, responses_per_model=22072, comparisons_sha256=comparisons,
        finished_at=study.now()))


if __name__ == '__main__':
    main()
