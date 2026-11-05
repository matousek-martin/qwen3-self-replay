"""Independent answer grading and pinned student-token lengths; no paid calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from .common import ROOT, REVISION, canonical, digest, durable_json, file_hash, prompts
from experiments.data_pipeline.pipeline import grader


def verify_response(body, prompt, tokenizer, grade=None):
    grade = grade or grader()
    result = {'answer_correct': False, 'accepted_answer': False,
              'explanation_review': 'pending_independent_review', 'replay_eligible': False}
    if body.get('status') != 'completed' or body.get('incomplete_details'):
        return dict(result, rejection='incomplete_or_failed')
    messages = [item for item in body.get('output', []) if item.get('type') == 'message' and item.get('role') == 'assistant']
    if len(messages) != 1 or messages[0].get('status', 'completed') != 'completed':
        return dict(result, rejection='missing_or_multiple_assistant_messages')
    content = messages[0].get('content', [])
    if not content or any(part.get('type') != 'output_text' for part in content):
        return dict(result, rejection='refusal_or_nontext')
    text = '\n'.join(part['text'] for part in content).strip()
    result['ordinary_text'] = text
    # Strict separation prevents malformed tagged material from bypassing grading.
    if '<think' in text or '</think' in text or len(re.findall(r'^Final answer:', text, re.M)) != 1:
        return dict(result, rejection='malformed_final_answer')
    lines = text.splitlines()
    if len(lines) < 2 or not lines[-1].startswith('Final answer:'):
        return dict(result, rejection='missing_worked_solution_or_final_line')
    worked = '\n'.join(lines[:-1]).strip()
    if not worked:
        return dict(result, rejection='empty_worked_solution')
    correct = bool(grade(text, prompt['answer'], prompt['task']))
    if not correct:
        return dict(result, rejection='incorrect_answer')
    # Qwen generation prefix already opens <think>; target contains its closure.
    target = worked + '\n</think>\n' + lines[-1]
    result.update(answer_correct=True, accepted_answer=True, rejection=None,
                  completion=target, completion_sha256=digest(target),
                  total_tokens=len(tokenizer.encode(target, add_special_tokens=False)),
                  ordinary_text_tokens=len(tokenizer.encode(text, add_special_tokens=False)),
                  tokenizer_revision=REVISION)
    if result['total_tokens'] > 2048:
        result.update(accepted_answer=False, rejection='student_target_over_2048')
    return result


def verify_run(run_dir, out, reviews=None):
    import importlib.metadata
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', revision=REVISION, local_files_only=True)
    cohort = {r['id']: r for r in prompts()}
    run_dir, out = Path(run_dir), Path(out)
    if out.exists():
        raise ValueError('Verification destination exists; use a fresh directory')
    plan = json.loads((run_dir / 'plan.json').read_text())
    request_map = {r['request_key']: r for r in plan['requests']}
    reviews_by_id = {}
    if reviews:
        for row in map(json.loads, Path(reviews).read_text().splitlines()):
            if row['request_key'] in reviews_by_id:
                raise ValueError('Duplicate explanation review')
            reviews_by_id[row['request_key']] = row
    results = []
    for path in sorted((run_dir / 'requests').glob('*.json')):
        record = json.loads(path.read_text())
        request = record['request']
        if request != request_map.get(request['request_key']):
            raise ValueError('Request is not in the frozen plan')
        if record['state'] != 'completed':
            continue
        body = record['response']
        if digest(body) != record['response_sha256'] or body != json.loads(record['wire']['body']):
            raise ValueError('Response evidence mismatch')
        prompt = cohort[request['prompt_id']]
        if digest(prompt) != request['prompt_sha256']:
            raise ValueError('Prompt hash mismatch')
        result = verify_response(body, prompt, tokenizer)
        review = reviews_by_id.get(request['request_key'])
        if review:
            if review.get('response_sha256') != digest(body) or not review.get('reviewer') or not review.get('evidence'):
                raise ValueError('Explanation review lacks identity or evidence')
            if review.get('verdict') not in ('pass', 'fail'):
                raise ValueError('Explanation review verdict must be pass/fail')
            result.update(explanation_review=review, replay_eligible=result['accepted_answer'] and review['verdict'] == 'pass')
        results.append(dict(result, prompt_id=prompt['id'], source=prompt['source'],
                            request_key=request['request_key'], teacher_model=body['model'],
                            variant=request['variant'], response_sha256=digest(body),
                            provider_request_id=record['wire']['request_id'], cost=record['cost']))
    if not results:
        raise ValueError('No completed responses to verify')
    unknown = set(reviews_by_id) - {r['request_key'] for r in results}
    if unknown:
        raise ValueError('Reviews reference absent responses')
    out.mkdir(parents=True)
    # All artifacts use atomic durable writes; JSON is intentionally not append-only JSONL.
    durable_json(out / 'verified.json', results)
    groups = {}
    for model in sorted({r['teacher_model'] for r in results}):
        for variant in ('concise', 'explicit'):
            for source in sorted({r['source'] for r in results}):
                rows = [r for r in results if (r['teacher_model'], r['variant'], r['source']) == (model, variant, source)]
                groups['/'.join((model, variant, source))] = {
                    'observed': len(rows), 'expected': 32,
                    'answer_verified': sum(r['accepted_answer'] for r in rows),
                    'replay_eligible': sum(r['replay_eligible'] for r in rows),
                    'rejections': dict(Counter(r['rejection'] for r in rows if r['rejection'])),
                    'student_target_tokens': [r['total_tokens'] for r in rows if r['accepted_answer']],
                    'accounted_usd': sum(r['cost']['accounted_usd'] for r in rows),
                }
    durable_json(out / 'summary.json', {'groups': groups, 'plan_sha256': digest(plan),
        'verifier_sha256': file_hash(ROOT / 'ThinkingCap-Test/utils/train/rewards.py'),
        'verification_code_sha256': file_hash(__file__), 'tokenizer_revision': REVISION,
        'tokenizer_class': type(tokenizer).__name__,
        'tokenizer_backend_sha256': digest(tokenizer.backend_tokenizer.to_str()),
        'packages': {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers', 'huggingface-hub')},
        'verified_sha256': file_hash(out / 'verified.json'),
        'reviews_sha256': file_hash(reviews) if reviews else None,
        'limitation': 'Answer correctness alone does not validate the explanation.'})
    return {'responses': len(results), 'answer_verified': sum(r['accepted_answer'] for r in results),
            'replay_eligible': sum(r['replay_eligible'] for r in results)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reviews', help='Independent explanation reviews, JSONL; response hashes required')
    args = parser.parse_args()
    print(canonical(verify_run(args.run, args.out, args.reviews)))
