"""Isolated round3 sampling, three-arm training and paired development evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def check():
    for kind in ('inputs', 'sources'):
        base = ROOT / 'data' if kind == 'inputs' else ROOT
        for name, expected in read(ROOT / f'data/isolated/{kind}.json').items():
            if sha(base / name) != expected:
                raise ValueError('Sealed input/source changed: ' + name)


def identity(training=False):
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'], text=True).strip()
    if training:
        from experiments.gpu_policy import verify_gpu_names
        verify_gpu_names(result.split(',')[0])
    return result


def generate(data, destination, samples, seed, adapter=None, sampling=False):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from experiments.rl_training.train import MODEL, REVISION, verifier, reasoning_is_complete
    cohort = rows(data); tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    grade = verifier(); close = tok.convert_tokens_to_ids('</think>')
    if destination.exists():
        raise ValueError('Do not overwrite an existing generation')
    destination.parent.mkdir(parents=True, exist_ok=True)
    engine = LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION, dtype='bfloat16',
                 max_model_len=8192, gpu_memory_utilization=.85, generation_config='vllm',
                 trust_remote_code=False, seed=seed, enable_prefix_caching=False,
                 enable_lora=bool(adapter), max_lora_rank=64)
    route = LoRARequest('candidate', 1, str(adapter.resolve())) if adapter else None
    if route:
        engine.llm_engine.add_lora(route)
        if 1 not in engine.llm_engine.list_loras():
            raise RuntimeError('Adapter not registered')
    jobs = [(r, a) for r in cohort for a in range(samples)]
    with destination.open('x', buffering=1) as stream:
        for offset in range(0, len(jobs), 24):
            batch = jobs[offset:offset + 24]
            prompts = [tok.apply_chat_template(r['prompt'], tokenize=True, add_generation_prompt=True,
                                              enable_thinking=True) for r, a in batch]
            seeds = [int(hashlib.sha256(f'round3:{seed}:{r["id"]}:{a}'.encode()).hexdigest()[:8], 16) for r, a in batch]
            outputs = engine.generate([{'prompt_token_ids': p} for p in prompts],
                [SamplingParams(n=1, temperature=.6, top_p=.95, top_k=20,
                                max_tokens=2048 if sampling else 4096, seed=s,
                                stop_token_ids=[tok.eos_token_id], include_stop_str_in_output=True) for s in seeds],
                lora_request=route, use_tqdm=False)
            for (row, attempt), prefix, s, output in zip(batch, prompts, seeds, outputs):
                actual = output.lora_request
                if (actual is None) != (route is None) or (route and actual.lora_int_id != 1):
                    raise RuntimeError('Evaluation adapter routing mismatch')
                answer = output.outputs[0]; raw_ids = list(answer.token_ids); ids = raw_ids
                if close in ids:
                    ids = ids[:ids.index(close) + 1 + 1024]
                text = tok.decode(ids, skip_special_tokens=False).replace('<|im_end|>', '').strip()
                complete = reasoning_is_complete(text)
                correct = bool(grade(text, row['answer'], row['task']))
                record = {'prompt_id': row['id'], 'source': row['source'], 'attempt': attempt,
                          'completion': text, 'correct': correct, 'seed': s,
                          'finish_reason': answer.finish_reason, 'total_tokens': len(ids),
                          'thinking_tokens': ids.index(close) if close in ids else len(ids),
                          'truncated': len(ids) != len(raw_ids) or answer.finish_reason != 'stop',
                          'accepted_answer': bool(correct and complete and answer.finish_reason == 'stop'
                              and len(ids) == len(raw_ids) and len(tok.encode(text, add_special_tokens=False)) + 1 <= 2048),
                          'model': MODEL, 'revision': REVISION,
                          'completion_sha256': hashlib.sha256(text.encode()).hexdigest()}
                if sampling:
                    record['raw_evidence'] = {'prompt_token_ids': prefix, 'generated_token_ids': raw_ids,
                                              'raw_text': answer.text}
                stream.write(json.dumps(record) + '\n')
            print(json.dumps({'generation': destination.name, 'responses': offset + len(batch), 'total': len(jobs)}), flush=True)
    write(destination.with_suffix('.summary.json'), {'complete': True, 'responses': len(jobs),
        'questions': len(cohort), 'samples': samples, 'data_sha256': sha(data), 'gpu_identity': identity(),
        'temperature': .6, 'top_p': .95, 'top_k': 20, 'max_tokens': 2048 if sampling else 4096,
        'final_cap': 1024, 'seed': seed, 'adapter': str(adapter) if adapter else None})


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['sample', 'train', 'eval'])
    p.add_argument('--seed', type=int, default=17); p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path); p.add_argument('--adapter', type=Path)
    a = p.parse_args(); check(); config = read(ROOT / 'data/isolated/config.json')
    if a.mode == 'sample':
        generate(ROOT / 'data/candidates.jsonl', ROOT / 'output/student.jsonl', 8, 20261027, sampling=True)
        write(ROOT / 'output/role-result.json', {'development_complete': True,
            'sampling_complete': True, 'responses': 8 * len(rows(ROOT / 'data/candidates.jsonl'))})
        return
    if a.mode == 'eval':
        generate(a.data.resolve(), a.out.resolve(), 8, 9001 + a.seed * 1000, a.adapter)
        return
    if a.seed not in config['seeds']:
        raise ValueError('Seed not registered')
    gpu = identity(training=True)
    for arm in config['arms']:
        target = ROOT / f'output/training/{arm}-seed{a.seed}'
        params = dict(config['training'], seed=a.seed, out=str(target),
                      data=str(ROOT / 'data/pilot.jsonl'), references=str(ROOT / 'data/references.json'),
                      static_replay=str(ROOT / 'data/static/manifest.json'),
                      static_replay_sha256=sha(ROOT / 'data/static/manifest.json'), static_arm=arm)
        command = [sys.executable, '-m', 'experiments.rl_training.train']
        for key, value in params.items():
            command.extend(['--' + key.replace('_', '-'), str(value)])
        subprocess.run(command, cwd=ROOT, check=True)
        status = read(target / 'status.json')
        if status['status'] != 'completed' or status['completed_groups'] != 512:
            raise RuntimeError('Training incomplete')
    subprocess.run([sys.executable, '-m', 'experiments.vllm_compat', '--out', str(ROOT / 'output/vllm-compat.json')], cwd=ROOT, check=True)
    for arm in ['base', *config['arms']]:
        if identity(training=True) != gpu:
            raise RuntimeError('Physical GPU changed')
        command = [sys.executable, '-m', 'experiments.round3_gpu', 'eval', '--seed', str(a.seed),
                   '--data', str(ROOT / 'data/development.jsonl'), '--out', str(ROOT / f'output/validation/{arm}.jsonl')]
        if arm != 'base':
            command += ['--adapter', str(ROOT / f'output/training/{arm}-seed{a.seed}/adapter')]
        subprocess.run(command, cwd=ROOT, check=True)
        summary = read(ROOT / f'output/validation/{arm}.summary.json')
        if summary['responses'] != 1536 or summary['gpu_identity'] != gpu:
            raise RuntimeError('Paired evaluation incomplete')
    write(ROOT / 'output/role-result.json', {'development_complete': True, 'seed': a.seed,
        'arms': config['arms'], 'gpu_identity': gpu, 'config_sha256': sha(ROOT / 'data/isolated/config.json')})


if __name__ == '__main__':
    main()
