"""Run the unchanged official evaluator on frozen study subsets and staged LoRAs.

No cloud provisioning. Configuration defaults to data/study/evaluation.json.
The official generation/grading code is reused; subset metadata is annotated
explicitly so these runs cannot be mistaken for the full contest evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'ThinkingCap-Test'
BASE_MODEL = 'Qwen/Qwen3-0.6B'
BASE_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
SAMPLES = {'tuning': 2, 'confirmation': 8}
SAMPLING = {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20}
DATASETS = {'mbpp', 'svamp', 'ifeval', 'bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_lines(rows):
    return ''.join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n'
                   for row in rows).encode('utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Study path escapes the staged study directory')
    return path


def directory_hashes(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Staged adapters must contain ordinary files, not symlinks')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def check_adapter(path):
    config = json.loads((path / 'adapter_config.json').read_text())
    if config.get('base_model_name_or_path') != BASE_MODEL or config.get('revision') != BASE_REVISION:
        raise ValueError('Adapter does not identify the pinned Qwen3-0.6B base revision')
    if config.get('peft_type') != 'LORA' or type(config.get('r')) is not int or config['r'] <= 0:
        raise ValueError('Only standard LoRA adapters are supported by this role')
    hashes = directory_hashes(path)
    if 'adapter_model.safetensors' not in hashes:
        raise ValueError('Missing adapter_model.safetensors')
    return hashes


def load_selection(source, split_path, subset):
    manifest = json.loads(split_path.read_text())
    source_manifest = json.loads((source / 'manifest.json').read_text())
    source_sha = sha256(source / 'questions.jsonl')
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported study split schema')
    if manifest.get('source_cohort_sha256') != source_sha or source_manifest.get('cohort_sha256') != source_sha:
        raise ValueError('Canonical benchmark questions do not match the frozen split')
    if manifest.get('source_manifest_sha256') != sha256(source / 'manifest.json'):
        raise ValueError('Canonical benchmark manifest differs from frozen split')
    wanted = manifest['subsets'][subset]
    keys = {(item['dataset'], item['id']): item['seed_index'] for item in wanted['keys']}
    if len(keys) != len(wanted['keys']):
        raise ValueError('Duplicate question in study split')
    full = read_jsonl(source / 'questions.jsonl')
    rows = [row for row in full if (row['dataset'], row['id']) in keys]
    if not rows or len(rows) != len(keys) or len(rows) != wanted['total']:
        raise ValueError('Missing or empty subset question selection')
    if any(row['seed_index'] != keys[(row['dataset'], row['id'])] for row in rows):
        raise ValueError('Split changed original generation seed indices')
    counts = dict(Counter(row['dataset'] for row in rows))
    if counts != wanted['counts'] or set(counts) != DATASETS:
        raise ValueError('Study subset must preserve all four domains and three BFCL subsets')
    body = canonical_lines(rows)
    if hashlib.sha256(body).hexdigest() != wanted['questions_sha256']:
        raise ValueError('Study subset content/order does not match frozen split hash')
    # Prove tuning and confirmation memberships are disjoint before any evaluation.
    tuning = {(item['dataset'], item['id']) for item in manifest['subsets']['tuning']['keys']}
    confirmation = {(item['dataset'], item['id']) for item in manifest['subsets']['confirmation']['keys']}
    if tuning & confirmation or tuning | confirmation != {(r['dataset'], r['id']) for r in full}:
        raise ValueError('Tuning and confirmation are not disjoint/exhaustive')
    return manifest, source_manifest, rows, body


def stage_questions(source, output, source_manifest, rows, body, subset, split_sha):
    expected_hash = hashlib.sha256(body).hexdigest()
    if output.exists():
        saved = json.loads((output / 'study-selection.json').read_text())
        if saved['split_manifest_sha256'] != split_sha or sha256(output / 'questions.jsonl') != expected_hash:
            raise ValueError('Existing staged questions differ')
        return
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    # copytree preserves the frozen input's 0444 mode. Replace the disposable
    # staged copy atomically instead of opening that read-only file for writing.
    staged = output / 'questions.jsonl.tmp'
    staged.write_bytes(body)
    staged.replace(output / 'questions.jsonl')
    adjusted = dict(source_manifest, cohort_sha256=expected_hash, total=len(rows),
                    counts=dict(Counter(row['dataset'] for row in rows)))
    write_json(output / 'manifest.json', adjusted)
    write_json(output / 'study-selection.json', {'subset': subset, 'scope': 'study subset, not full contest cohort',
               'split_manifest_sha256': split_sha, 'source_cohort_sha256': source_manifest['cohort_sha256'],
               'cohort_sha256': expected_hash, 'total': len(rows), 'original_seed_indices_preserved': True})


def check_lock(lock_path, name, hashes, split_sha):
    lock = json.loads(lock_path.read_text())
    if lock.get('schema_version') != 1 or lock.get('candidate') != name:
        raise ValueError('Confirmation lock candidate differs')
    if lock.get('adapter_files_sha256') != hashes or lock.get('split_manifest_sha256') != split_sha:
        raise ValueError('Confirmation candidate or split changed after locking')
    if lock.get('primary_candidate') != name:
        raise ValueError('Confirmation requires this candidate to be the predeclared primary')
    if not lock.get('locked_at'):
        raise ValueError('Confirmation lock has no timestamp')
    return lock


def validate_records(run, rows, samples, cohort_sha, subset):
    results = json.loads((run / 'results.json').read_text())
    records = read_jsonl(run / 'records.jsonl')
    expected = {(row['dataset'], row['id'], sample): 9001 + row['seed_index'] * 1000 + sample
                for row in rows for sample in range(samples)}
    seen = set()
    for record in records:
        key = (record['dataset'], record['id'], record['sample_index'])
        if key in seen or key not in expected or record['seed'] != expected[key]:
            raise ValueError('Duplicate, unexpected, or differently seeded evaluation response')
        if type(record.get('correct')) is not bool or record.get('grading_error'):
            raise ValueError('Ungraded evaluation response')
        if type(record.get('output_tokens')) is not int or record['output_tokens'] < 0:
            raise ValueError('Missing generated token accounting')
        seen.add(key)
    if seen != set(expected):
        raise ValueError('Incomplete evaluation response coverage')
    caps = {dataset: 8192 if dataset == 'mbpp' else 4096 for dataset in DATASETS}
    checks = {'samples': samples, 'seed': 9001, 'sampling': SAMPLING, 'max_tokens': caps,
              'max_model_len': 12288, 'cohort_sha256': cohort_sha, 'cohort_total': len(rows),
              'smoke': False, 'model': BASE_MODEL, 'revision': BASE_REVISION}
    for field, expected_value in checks.items():
        if results.get(field) != expected_value:
            raise ValueError('Official evaluation setting mismatch: ' + field)
    if results.get('failed_benchmarks') or set(results.get('results', {})) != DATASETS:
        raise ValueError('A required benchmark failed to grade')
    if sum(item['n'] for item in results['results'].values()) != len(expected):
        raise ValueError('Result summary response count differs')
    if results.get('selection') not in ('full', 'study-' + subset):
        raise ValueError('Unexpected question selection metadata')
    return results


def run_process(command, log_path, timeout):
    """Kill the evaluator's whole process group if its bounded run expires."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('RUN ' + ' '.join(map(str, command)), flush=True)
    with log_path.open('a') as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=REPO, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
    if result:
        raise RuntimeError(f'Evaluator exited {result}; see {log_path}')


def evaluator_hashes():
    paths = [REPO / 'evaluate.py', REPO / 'compare.py'] + sorted((REPO / 'utils/eval').rglob('*.py'))
    return {str(path.relative_to(REPO)): sha256(path) for path in paths}


def preflight_nltk():
    import nltk
    # Container umasks can create group-writable cache directories. NLTK's
    # pinned release correctly refuses to trust resources in such directories.
    cache = Path(os.environ.get('NLTK_DATA', '/root/thinkingcap_nltk_data'))
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache.chmod(0o700)
    nltk.data.path = [str(cache)]
    for name in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find('tokenizers/' + name)
        except LookupError:
            if not nltk.download(name, download_dir=str(cache), quiet=True, raise_on_error=True):
                raise RuntimeError('Required NLTK resource could not be installed: ' + name)
        nltk.data.find('tokenizers/' + name)


def interrupt(signum, frame):
    # Raising through run_process lets it stop its detached evaluator group.
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'data/study/evaluation.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'output/study')
    parser.add_argument('--check-only', action='store_true', help='Validate/stage inputs without loading a model')
    args = parser.parse_args()
    study = args.config.resolve().parent
    config = json.loads(args.config.read_text())
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported study evaluation configuration')
    subset = config['subset']
    if subset not in SAMPLES or config.get('samples') != SAMPLES[subset]:
        raise ValueError('Use two samples for tuning or eight for locked confirmation')
    samples = config['samples']
    names = config['candidates']
    if not names or len(names) != len(set(names)) or 'base' not in names:
        raise ValueError('Each study batch requires a fresh matched base candidate')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) for name in names):
        raise ValueError('Unsafe candidate name')
    names = ['base'] + [name for name in names if name != 'base']
    if subset == 'confirmation' and len(names) != 2:
        raise ValueError('A confirmation job evaluates one locked candidate plus its matched baseline')
    workers = config.get('workers', 4)
    # Zero means completion-driven; the control server owns budget enforcement.
    timeout = config.get('candidate_timeout_seconds', 0)
    if type(workers) is not int or not 1 <= workers <= 16 or type(timeout) not in (int, float) or not 0 <= timeout <= 43200:
        raise ValueError('Invalid worker count or per-candidate timeout')
    timeout = timeout or None
    if os.environ.get('THINKINGCAP_COMPLETION_DRIVEN') == '1':
        timeout = None
    source = inside(study, config['questions']) if config.get('questions') else REPO / 'questions'
    split_path = inside(study, config['split_manifest'])
    split, source_manifest, rows, body = load_selection(source, split_path, subset)
    split_sha = sha256(split_path)
    hashes = {name: check_adapter(inside(study, 'adapters/' + name)) for name in names if name != 'base'}
    lock = None
    if subset == 'confirmation':
        if not config.get('candidate_lock'):
            raise ValueError('Confirmation requires a candidate lock before generation')
        lock_path = inside(study, config['candidate_lock'])
        lock = check_lock(lock_path, names[1], hashes[names[1]], split_sha)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'schema_version': 1, 'config': config, 'config_sha256': sha256(args.config),
                  'effective_candidate_timeout_seconds': timeout,
                  'split_manifest_sha256': split_sha, 'subset': subset,
                  'scope': 'official prompts and recipe on a frozen study subset; not full contest evaluation',
                  'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
                  'adapter_files_sha256': hashes, 'candidate_lock': lock,
                  'evaluator_files_sha256': evaluator_hashes(), 'runner_sha256': sha256(__file__),
                  'cohort_sha256': hashlib.sha256(body).hexdigest(), 'cohort_total': len(rows),
                  'samples': samples, 'matched_baseline_regenerated': True,
                  'generation_backend': 'official vLLM LoRA; no merging, prompt, decoding or cap changes'}
    manifest_path = output / 'study-provenance.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != provenance:
        raise ValueError('Existing study output provenance differs; use a new directory')
    write_json(manifest_path, provenance)
    write_json(output / 'study-scope.json', {'subset': subset, 'full_contest_evaluation': False,
               'questions': len(rows), 'samples': samples, 'source_total': split.get('source_total'),
               'matched_baseline_required': True, 'confirmation_candidate_locked': lock is not None,
               'results_metadata_amendment': 'results.original.json preserves official output; only selection/study metadata are amended'})
    shutil.copyfile(split_path, output / 'split-manifest.json')
    question_root = output / 'questions'
    stage_questions(source, question_root, source_manifest, rows, body, subset, split_sha)
    if args.check_only:
        print(json.dumps({'status': 'inputs_verified', 'questions': len(rows), 'samples': samples,
                          'candidates': names, 'out': str(output)}, indent=2))
        return
    preflight_nltk()
    failures, completed = {}, []
    for name in names:
        run = output / 'runs' / name
        status = {'candidate': name, 'state': 'running', 'started_at': now(), 'subset': subset}
        status_path = output / 'status' / (name + '.json')
        write_json(status_path, status)
        try:
            if not (run / 'results.json').exists():
                if run.exists() and any(run.iterdir()):
                    raise RuntimeError('Incomplete earlier evaluation preserved; use a fresh study output to rerun')
                command = [sys.executable, REPO / 'evaluate.py', '--questions', question_root,
                           '--out', run, '--samples', samples, '--seed', 9001, '--workers', workers]
                if name != 'base':
                    command += ['--adapter', inside(study, 'adapters/' + name)]
                run_process(command, output / 'logs' / (name + '.log'), timeout)
            results = validate_records(run, rows, samples, provenance['cohort_sha256'], subset)
            if name != 'base' and check_adapter(inside(study, 'adapters/' + name)) != hashes[name]:
                raise ValueError('Staged adapter changed during evaluation')
            # Official load_suite calls any supplied cohort "full". Preserve its
            # exact output, then amend metadata only to identify this study subset.
            original = run / 'results.original.json'
            if not original.exists():
                shutil.copyfile(run / 'results.json', original)
            results['selection'] = 'study-' + subset
            results['study'] = {'subset': subset, 'split_manifest_sha256': split_sha,
                                'full_contest_evaluation': False, 'candidate': name,
                                'adapter_files_sha256': hashes.get(name)}
            write_json(run / 'results.json', results)
            status.update(state='complete', finished_at=now(), responses=len(rows) * samples,
                          records_sha256=sha256(run / 'records.jsonl'), results_sha256=sha256(run / 'results.json'))
            if name != 'base':
                comparison = output / 'comparisons' / (name + '.json')
                comparison.parent.mkdir(parents=True, exist_ok=True)
                run_process([sys.executable, REPO / 'compare.py', output / 'runs/base', run,
                             '--out', comparison], output / 'logs' / (name + '-compare.log'), 600)
                status['comparison_sha256'] = sha256(comparison)
            completed.append(name)
        except Exception as exc:
            failures[name] = str(exc)
            status.update(state='failed', finished_at=now(), error=str(exc))
            if name == 'base':
                write_json(status_path, status)
                write_json(output / 'study-result.json', {'subset': subset, 'complete': False,
                           'completed_candidates': completed, 'failures': failures})
                raise
        write_json(status_path, status)
    write_json(output / 'study-result.json', {'subset': subset, 'complete': not failures,
               'completed_candidates': completed, 'failures': failures,
               'scope': provenance['scope'], 'finished_at': now()})
    if failures:
        raise RuntimeError('Some study candidates failed; preserved outputs identify which ones')


if __name__ == '__main__':
    main()
