"""Prepare isolated inputs and a design for the next teacher comparison; no paid calls."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.data_pipeline.pipeline import (
    DuplicateIndex, digest, file_hash, prompt_text, read_jsonl, write_jsonl,
)

OUT = ROOT / 'lab/experiments/teacher-round3-20261026'
SOURCES = {'gsm8k': 128, 'arc_easy': 64, 'commonsense_qa': 64}


def main():
    if OUT.exists():
        raise ValueError('Refusing to replace an existing preparation')
    parent = ROOT / 'lab/experiments/data'
    manifest = json.loads((parent / 'manifest.json').read_text())
    for name in ('train.jsonl', 'val.jsonl', 'pilot.jsonl', 'sources.lock.json'):
        if file_hash(parent / name) != manifest['files'][name]:
            raise ValueError('Parent input changed: ' + name)
    benchmark = ROOT / 'ThinkingCap-Test/questions/questions.jsonl'
    if file_hash(benchmark) != manifest['benchmark_file_sha256']:
        raise ValueError('Frozen evaluation cohort changed')
    index = DuplicateIndex(0.8)
    for row in read_jsonl(benchmark):
        index.add('benchmark:' + row['id'], prompt_text(row))
    for name in ('val.jsonl', 'pilot.jsonl'):
        for row in read_jsonl(parent / name):
            index.add(name + ':' + row['id'], prompt_text(row))
    # Also exclude historical teacher-generation questions outside the pilot.
    old = parent / 'teacher-pilot-20261024/teacher-comparison-128.jsonl'
    for row in read_jsonl(old):
        index.add('historical_teacher:' + row['id'], prompt_text(row))
    rows = sorted((r for r in read_jsonl(parent / 'train.jsonl') if r['source'] in SOURCES),
                  key=lambda r: digest(['teacher-round3-v1', r['id']]))
    splits = {}; rejected = []
    # Reserve fresh development questions before assigning training candidates.
    for split, quotas in (
        ('development', {s: 64 for s in SOURCES}),
        ('primary-candidates', SOURCES),
        ('reserve-candidates', {s: n * 3 for s, n in SOURCES.items()}),
        ('rl-stream', {s: n * 2 for s, n in SOURCES.items()}),
    ):
        chosen = []; counts = Counter(); unused = []
        for row in rows:
            source = row['source']
            if counts[source] >= quotas[source]:
                unused.append(row); continue
            hit = index.match(prompt_text(row))
            if hit:
                rejected.append({'id': row['id'], 'split_attempt': split,
                                 'matched_id': hit[0], 'reason': hit[1], 'score': hit[2]})
                continue
            chosen.append(dict(row, split='development' if split == 'development' else 'train'))
            counts[source] += 1
            index.add(split + ':' + row['id'], prompt_text(row))
        if dict(counts) != quotas:
            raise ValueError('Insufficient screened candidates: ' + split)
        splits[split] = chosen; rows = unused
    all_ids = [r['id'] for rs in splits.values() for r in rs]
    assert len(all_ids) == len(set(all_ids))
    OUT.mkdir(parents=True)
    for name, rs in splits.items():
        write_jsonl(OUT / (name + '.jsonl'), rs)
    write_jsonl(OUT / 'quarantine.jsonl', rejected)
    lock = json.loads((parent / 'sources.lock.json').read_text())
    (OUT / 'sources.lock.json').write_text(json.dumps(
        {'schema_version': 1, 'sources': {s: lock['sources'][s] for s in SOURCES}}, indent=2) + '\n')
    data = {'created_at': datetime.now(timezone.utc).isoformat(),
            'parent_manifest_sha256': file_hash(parent / 'manifest.json'),
            'benchmark_sha256': file_hash(benchmark),
            'selection_code_sha256': file_hash(Path(__file__)),
            'counts': {s: dict(Counter(r['source'] for r in rs)) for s, rs in splits.items()},
            'files': {p.name: file_hash(p) for p in sorted(OUT.iterdir()) if p.is_file()},
            'screening': {'exact': True, 'five_gram_containment': 0.8,
                          'semantic_screening_completed': False,
                          'not_a_guarantee_of_no_semantic_overlap': True},
            'matched_verified_replay_pairs': 0,
            'note': '256 primary question candidates plus 768 ordered reserves; not generated targets.'}
    (OUT / 'data-manifest.json').write_text(json.dumps(data, indent=2) + '\n')
    plan = {
        'schema_version': 1, 'id': 'teacher-round3-v1', 'state': 'prepared_inputs',
        'launch_ready': False, 'paid_work_started': False,
        'authorization_scope': 'Prepare next experiment; notify on completion. No new launch recorded.',
        'hypothesis': 'Moderate teacher editing of verified student solutions improves total-token efficiency without sacrificing accuracy.',
        'student': {'model': 'Qwen/Qwen3-0.6B', 'revision': manifest['base_revision']},
        'arms': ['student', 'qwen3_32b', 'qwen38_27b'], 'seeds': [17, 29, 43],
        'training_runs': 9,
        'teachers': {'qwen3_32b': 'qwen/qwen3-32b', 'qwen38_27b': 'qwen/qwen3.8-27b'},
        'provider_candidate': 'DeepInfra', 'provider_fallback': False,
        'replay_target_count': 256, 'source_counts': SOURCES,
        'paired_data': {
            'student': 'Pinned base model; up to 8 predeclared attempts per candidate; select shortest independently verified correct complete solution.',
            'teacher': 'Edit that same student solution to about 70–80% of its student-token length; retain required reasoning and final answer.',
            'acceptance': 'All three targets independently correct and complete; target shortening is a soft goal, never excuse to drop reasoning.',
            'fallback': 'Consume source-specific reserve IDs in fixed order until quotas met; never silently reduce quotas or weaken verification.',
            'exhaustion': 'Publish insufficient yield with counts; no training until a versioned expansion supplies all 256 pairs.',
        },
        'training': {'optimizer': 'dr_grpo', 'groups': 512, 'group_size': 4,
                     'learning_rate': 1e-5, 'length_coefficient': 0.1,
                     'replay_weight': 0.1, 'lora_rank': 16, 'lora_alpha': 32,
                     'max_new_tokens': 2048, 'final_cap': 1024, 'kl_beta': 0,
                     'gpu_required': 'NVIDIA GeForce RTX 4090',
                     'schedule': 'Same RL order and replay ID schedule across arms for each seed; shuffled cyclic replay pool.',
                     'compute_caveat': 'Matched rollout groups and replay opportunities; variable target lengths mean token compute is NOT matched.'},
        'evaluation': {'development_questions': 192, 'samples_per_question': 8,
                       'fresh_base_same_physical_worker': True,
                       'selection': 'Paired development accuracy noninferiority lower 95% bound >= -0.01; among eligible arms prefer lowest total-token ratio; report each source.',
                       'if_no_eligible_arm': 'No declared winner; retain student control and report inconclusive/failed.',
                       'official': 'Unchanged official full evaluation protocol after development; repeated benchmark is not untouched holdout.'},
        'budget': {'shared_cap_usd': 200, 'existing_ledger_required': True,
                   'completion_backup_verify_release_required': True,
                   'new_cost_bound_usd': None, 'note': 'Freeze exact API and GPU bounds against live remaining budget before launch; do not infer affordability from old state.'},
        'dependencies_before_launch': [
            'Audit suspicious semantic overlaps without supplying benchmark examples to generators.',
            'Implement OpenRouter transport/accounting with fixed provider, durable receipts and no blind retries.',
            'Generate and independently verify 256 matched three-arm targets; freeze pool/schedule hashes.',
            'Extend static replay worker to three arms; pin reward references from independent development, evaluator and package lock.',
            'Freeze all remaining parameters, validation statistics code and exact spending bounds; seal separate release; verify 4090 before training.',
        ],
        'coding_followup': {'state': 'design_only', 'target_new_tasks': 128,
                            'sources': ['Pinned MIT Exercism Python', 'Independent synthetic Python tasks'],
                            'design': 'Winner versus same winner with coding data; hold training rollout budget fixed to isolate data mix; three seeds.',
                            'dependencies': ['Stage-one outcome', 'Provenance and independent contamination screen', 'Isolated code grader with independent tests'],
                            'campaign_path': 'lab/experiments/teacher-round3-coding-20261026/campaign.json'},
        'notification': {'telegram_authorized': True, 'only_verified_completion': True},
        'data_manifest_sha256': file_hash(OUT / 'data-manifest.json'),
    }
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (OUT / 'campaign.json').write_text(json.dumps({'state': 'prepared_inputs', 'trials': {},
        'plan_sha256': file_hash(OUT / 'plan.json'), 'paid_work_started': False}, indent=2) + '\n')
    print(json.dumps({'path': str(OUT), 'counts': data['counts'], 'launch_ready': False}))


if __name__ == '__main__':
    main()
