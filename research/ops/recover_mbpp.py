"""Regrade missing MBPP verdicts on CPU, preserving the original GPU output."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from runpod import ROOT
from run_baseline import write_json

REPO = ROOT / 'ThinkingCap-Test'
sys.path.insert(0, str(REPO))
import evaluate


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    active = json.loads((ROOT / 'lab/active-run.json').read_text())
    archive = Path(active['archive'])
    source = archive / 'runs/base'
    target = archive / 'runs/base-regraded'
    target.mkdir(exist_ok=False)
    original = evaluate.read_jsonl(source / 'records.jsonl')
    records = [dict(row) for row in original]
    questions = evaluate.read_jsonl(REPO / 'questions/questions.jsonl')
    suite = evaluate.load_suite(REPO / 'questions', False)
    old_results = json.loads((source / 'results.json').read_text())
    assert suite['cohort_sha256'] == old_results['cohort_sha256']
    assert len(records) == 22072
    assert Counter((r['id'], r['sample_index']) for r in records) == Counter(
        (q['id'], sample) for q in questions for sample in range(8))
    missing = [r for r in records if r.get('correct') is None]
    assert missing and all(r['dataset'] == 'mbpp' for r in missing)
    assert evaluate.sandbox.have_bwrap(), 'Local recovery requires filesystem/network isolation'
    for record in missing:
        record.pop('grading_error', None)
    context = {'suite': suite, 'mode': 'bwrap', 'workers': 1,
               'checkers': evaluate.load_checkers(suite, ['mbpp'])}
    print(f'Regrading {len(missing)} saved answers in an isolated CPU worker', flush=True)
    evaluate.grade_programs({q['id']: q for q in questions}, missing, context)
    assert all(isinstance(r.get('correct'), bool) and not r.get('grading_error')
               for r in missing), [{k: r.get(k) for k in ('id', 'grading_error')} for r in missing]
    grade_keys = {'correct', 'parse_error', 'grading_error'}
    for before, after in zip(original, records):
        assert {k:v for k,v in before.items() if k not in grade_keys} == {
            k:v for k,v in after.items() if k not in grade_keys}
        if before.get('correct') is not None:
            assert before == after
    scores, failed = evaluate.collect(questions, records, 8)
    assert not failed and sum(r['n'] for r in scores.values()) == 22072
    for dataset, score in old_results['results'].items():
        assert score == scores[dataset], dataset
    results = dict(old_results, results=scores, failed_benchmarks=failed)
    results['grading_recovery'] = {
        'scope': 'Only previously ungraded MBPP answers', 'rows': len(missing),
        'tasks': sorted({r['id'] for r in missing}), 'sandbox': 'bwrap',
        'memory_bytes': evaluate.sandbox.MEMORY_BYTES, 'device': 'cpu',
        'source': '../base', 'generated_answers_unchanged': True}
    evaluate.save_records(target, records)
    write_json(target / 'results.json', results)
    hashes = {name: digest(target / name) for name in ('records.jsonl', 'results.json')}
    (target / 'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
    patch = subprocess.run(['git', '-C', str(REPO), 'diff'], check=True,
                           capture_output=True, text=True).stdout
    (target / 'grader.patch').write_text(patch)
    packages = subprocess.run([str(Path('/root/.local/bin/uv')), 'pip', 'freeze',
                               '--python', sys.executable], check=True, capture_output=True, text=True)
    (target / 'packages.txt').write_text(packages.stdout)
    provenance = {
        'completed_at': datetime.now(timezone.utc).isoformat(),
        **results['grading_recovery'], 'outputs_sha256': hashes,
        'original_sha256': {name:digest(source/name) for name in ('records.jsonl','results.json')},
        'worker_sha256': digest(REPO / 'utils/eval/mbpp_worker.py'),
        'script_sha256': digest(Path(__file__)), 'additional_gpu_cost_usd': 0,
        'verification': 'All prompt/sample pairs present; generation fields and existing grades unchanged'}
    write_json(target / 'recovery.json', provenance)
    write_json(archive / 'recovery-status.json', {'phase':'graded', 'results_dir':str(target), **provenance})
    print(evaluate.report(scores, suite, failed), flush=True)
    print('Verified outputs:', target, flush=True)


if __name__ == '__main__':
    main()
