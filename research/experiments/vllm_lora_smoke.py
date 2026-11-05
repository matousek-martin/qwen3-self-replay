"""Small real-GPU adapter routing/effect check before a full evaluation batch."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    name = next(name for name in config['candidates'] if name != 'base')
    adapter = args.config.parent / 'adapters' / name
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    model_name, revision = 'Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Compute 17 + 25. Give the answer.'}],
                                           tokenize=True, add_generation_prompt=True, enable_thinking=True)
    rank = json.loads((adapter / 'adapter_config.json').read_text())['r']
    model = LLM(model=model_name, revision=revision, tokenizer_revision=revision,
                max_model_len=256, max_num_seqs=2, gpu_memory_utilization=0.3,
                enable_lora=True, max_lora_rank=rank, enable_prefix_caching=False,
                enforce_eager=True, generation_config='vllm', seed=17)
    lora = LoRARequest('routing-smoke', 1, str(adapter.resolve()))
    model.llm_engine.add_lora(lora)
    if 1 not in model.llm_engine.list_loras():
        raise RuntimeError('Smoke adapter did not register')

    def generate(route):
        result = model.generate([{'prompt_token_ids': prompt}],
                                SamplingParams(max_tokens=8, temperature=0, logprobs=5, seed=17),
                                lora_request=route, use_tqdm=False)[0]
        actual = result.lora_request
        wrong_route = (actual is not None) if route is None else (actual is None or actual.lora_int_id != route.lora_int_id)
        if wrong_route:
            raise RuntimeError('Actual GPU output lost or changed adapter routing metadata')
        item = result.outputs[0]
        return list(item.token_ids), {str(k): v.logprob for k, v in item.logprobs[0].items()}

    base_tokens, base = generate(None)
    candidate_tokens, candidate = generate(lora)
    repeat_tokens, repeat = generate(None)
    changed = set(base) != set(candidate) or any(abs(base[k] - candidate[k]) > 1e-6 for k in base.keys() & candidate.keys())
    if not changed:
        raise RuntimeError('No measurable adapter effect on the smoke logits')
    if base_tokens != repeat_tokens or set(base) != set(repeat) or any(abs(base[k] - repeat[k]) > 1e-5 for k in base):
        raise RuntimeError('Base inference changed after the adapter request')
    args.out.write_text(json.dumps({'candidate': name, 'metadata_verified': True, 'adapter_effect_observed': True,
                                    'base_route_restored': True, 'scope': 'Synthetic routing preflight only; not accuracy evaluation'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
