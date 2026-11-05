"""Private diagnostic regrade of saved generations; never modifies official runs."""
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'lab/isolated/full-evaluation-20261026/teacher-student-v2/ThinkingCap-Test'
OUT = ROOT / 'lab/experiments/teacher-format-diagnostic-20261026'
sys.path.insert(0, str(FROZEN))
import evaluate
from utils.eval import sandbox
from utils.eval.graders import mbpp_code
from utils.eval.stats import compare_runs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def repair(record):
    """Only a unique closing tag with no opening tag is repairable here."""
    result = dict(record)
    text = record['text']
    changed = '<think>' not in text and text.count('</think>') == 1
    if changed:
        result.update(text='<think>\n' + text, response=text.split('</think>', 1)[1], boundary_error=None)
        result.pop('grading_error', None)
        result['correct'] = None
        result['parse_error'] = None
    result.update(diagnostic_repaired=changed, official_correct=record['correct'],
                  official_parse_error=record.get('parse_error'))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    suite = evaluate.load_suite(FROZEN / 'questions')
    questions = evaluate.read_jsonl(FROZEN / 'questions/questions.jsonl')
    by_id = {q['id']: q for q in questions}
    grading = evaluate.grading_context(suite, SimpleNamespace(
        benchmarks=['svamp', 'bfcl', 'ifeval', 'mbpp'], sandbox='bwrap', workers=1))
    inputs = {str(p): digest(p) for p in [FROZEN / 'evaluate.py',
        FROZEN / 'utils/eval/graders.py', FROZEN / 'utils/eval/mbpp_worker.py',
        FROZEN / 'questions/questions.jsonl', FROZEN / 'questions/manifest.json']}
    runs, pending, items = {}, collections.defaultdict(list), {}
    for seed in (17, 29, 43):
        for arm in ('base', 'student', 'teacher'):
            name = 'base' if arm == 'base' else f'{arm}-seed{seed}'
            source = ROOT / f'lab/experiments/cloud/teacher-full-v2-seed{seed}-53649b2d9a/output/full-evaluation/runs/{name}/records.jsonl'
            inputs[str(source)] = digest(source)
            original = evaluate.read_jsonl(source)
            assert len(original) == 22072
            assert len({(r['id'], r['sample_index']) for r in original}) == len(original)
            rows = [repair(r) for r in original]
            runs[f'{seed}:{arm}'] = rows
            # Unmodified samples verify local checker compatibility with stored grades.
            checked = collections.Counter()
            for r in original:
                q = by_id[r['id']]
                if q['grader'] == 'mbpp' or checked[q['grader']] >= 25:
                    continue
                actual = evaluate.grade_one(q, r, grading['checkers'])
                assert actual == (r['correct'], r.get('parse_error')), (seed, arm, r['id'], actual)
                checked[q['grader']] += 1
            for r in rows:
                if not r['diagnostic_repaired']:
                    continue
                q = by_id[r['id']]
                if q['grader'] != 'mbpp':
                    r['correct'], r['parse_error'] = evaluate.grade_one(q, r, grading['checkers'])
                else:
                    code, error = mbpp_code(r['response'], q['entry_point'], grading['checkers']['mbpp'], r['truncated'])
                    if error:
                        r.update(correct=False, parse_error=error)
                    else:
                        key = hashlib.sha256((q['id'] + '\0' + code).encode()).hexdigest()
                        items[key] = (key, q['id'], code)
                        pending[key].append(r)
            stats = collections.Counter(r['dataset'] for r in rows if r['diagnostic_repaired'])
            print('noncode graded', seed, arm, dict(stats), flush=True)
            write(OUT / 'progress.json', {'phase': 'noncode', 'last_run': f'{seed}:{arm}', 'repairs': dict(stats)})
    manifest = dict(diagnostic_only=True, official_score=False,
        intervention='Prepend <think> plus newline only when text has no opening tag and exactly one closing tag; recover response after closing tag.',
        unchanged='Generated solution text, final answer, truncation, token counts, question/sample IDs; all unrepairable rows retain official verdicts.',
        token_note='Token counts remain measured original generation counts; synthetic tag is not a generated-token saving.',
        baseline_policy='Identical repair predicate applied to all three saved fresh baselines.',
        inputs_sha256=inputs, sandbox='bwrap', model_inference=False,
        mbpp_deduplication='Same question and sanitized code graded once, mapped back to every original sample.')
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Diagnostic inputs changed'
    else:
        write(manifest_path, manifest)
    # Exercise real sandboxed execution on known saved baseline verdicts.
    baseline = runs['17:base']
    smoke = [dict(r) for r in baseline if r['dataset'] == 'mbpp' and r['correct']][:3]
    evaluate.grade_programs(by_id, smoke, grading)
    assert all(r['correct'] is True and not r.get('grading_error') for r in smoke), smoke
    cache_path = OUT / 'mbpp-cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    tasks = collections.defaultdict(list)
    for key, item in items.items():
        if key not in cache:
            tasks[item[1]].append(item)
    task_groups = list(tasks.values())
    batches = [sum(task_groups[i:i+8], []) for i in range(0, len(task_groups), 8)]
    print('MBPP unique programs', len(items), 'batches', len(batches), flush=True)
    def execute(batch):
        return sandbox.run_batch('bwrap', suite['evalplus'], suite['mbpp'], batch, timeout=1800)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, batch) for batch in batches]
        for completed, future in enumerate(as_completed(futures), 1):
            verdicts = future.result()
            errors = {k: v for k, v in verdicts.items() if v.get('error') or v.get('base') is None or v.get('plus') is None}
            if errors:
                write(OUT / 'grading-errors.json', errors)
                raise RuntimeError('Sandbox grading errors; no complete diagnostic claim')
            cache.update(verdicts)
            write(cache_path, cache)
            write(OUT / 'progress.json', {'phase': 'mbpp', 'completed_batches': completed, 'batches': len(batches), 'cached_programs': len(cache)})
            print('MBPP batch', completed, '/', len(batches), flush=True)
    for key, rows in pending.items():
        verdict = cache[key]
        for r in rows:
            r.update(correct=verdict['base'] == 'pass' and verdict['plus'] == 'pass', parse_error=None)
    for name, rows in runs.items():
        assert all(type(r['correct']) is bool and not r.get('grading_error') for r in rows)
        path = OUT / (name.replace(':', '-') + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    comparisons = []
    for seed in (17, 29, 43):
        for arm in ('student', 'teacher'):
            left, right = runs[f'{seed}:base'], runs[f'{seed}:{arm}']
            domains = {}
            for domain in ('bfcl', 'ifeval', 'mbpp', 'svamp'):
                pairs = [(a, b) for a, b in zip(left, right) if a['dataset'].startswith(domain)]
                assert all((a['id'], a['sample_index']) == (b['id'], b['sample_index']) for a, b in pairs)
                a, b = zip(*pairs)
                domains[domain] = dict(n=len(a), base_accuracy=statistics.mean(r['correct'] for r in a),
                    official_accuracy=statistics.mean(r['official_correct'] for r in b),
                    diagnostic_accuracy=statistics.mean(r['correct'] for r in b),
                    repaired=sum(r['diagnostic_repaired'] for r in b),
                    corrected_false_to_true=sum(not r['official_correct'] and r['correct'] for r in b),
                    corrected_true_to_false=sum(r['official_correct'] and not r['correct'] for r in b),
                    token_ratio=sum(r['output_tokens'] for r in b)/sum(r['output_tokens'] for r in a),
                    diagnostic_ci95=compare_runs(a, b)['accuracy_delta'])
            comparisons.append(dict(seed=seed, arm=arm, domains=domains,
                official_accuracy_delta=statistics.mean(d['official_accuracy']-d['base_accuracy'] for d in domains.values()),
                diagnostic_accuracy_delta=statistics.mean(d['diagnostic_accuracy']-d['base_accuracy'] for d in domains.values()),
                token_reduction=1-math.exp(statistics.mean(math.log(d['token_ratio']) for d in domains.values()))))
    aggregate = {}
    for arm in ('student', 'teacher'):
        rows = [c for c in comparisons if c['arm'] == arm]
        aggregate[arm] = dict(official_accuracy_delta=statistics.mean(c['official_accuracy_delta'] for c in rows),
            diagnostic_accuracy_delta=statistics.mean(c['diagnostic_accuracy_delta'] for c in rows),
            token_reduction=1-math.exp(statistics.mean(math.log1p(-c['token_reduction']) for c in rows)),
            domains={domain: {key: statistics.mean(c['domains'][domain][key] for c in rows)
                for key in ('base_accuracy','official_accuracy','diagnostic_accuracy')} for domain in rows[0]['domains']},
            note='Descriptive equal-seed means; no joint three-seed confidence interval.')
    assert all(digest(Path(p)) == sha for p, sha in inputs.items()), 'Official source changed during audit'
    write(OUT / 'summary.json', dict(complete=True, diagnostic_only=True, official_score=False,
        comparisons=comparisons, aggregate=aggregate, official_inputs_unchanged=True))
    write(OUT / 'progress.json', {'phase': 'complete'})
    print(json.dumps(aggregate), flush=True)


if __name__ == '__main__':
    main()
