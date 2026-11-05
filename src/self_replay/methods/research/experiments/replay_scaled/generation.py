"""Batched, independently seeded decoding for the online replay worker.

Only answers at the same model state are batched. Training never generates a
future question before the preceding question's optimizer/replay update.
"""
PROTOCOL = 'hf-batched-row-seeded-v1'


def settings():
    return dict(protocol=PROTOCOL, evaluation_batch_size=64,
                max_padded_tokens=262144, training_batch='rank_local_answers',
                sampling='independent_per_answer_generator', padding_side='left')


def validate_settings(value):
    if not isinstance(value, dict) or value.get('protocol') != PROTOCOL:
        raise ValueError('Batched generation needs a new registered execution contract')
    for key in ('training_batch', 'sampling', 'padding_side'):
        if value.get(key) != settings()[key]:
            raise ValueError('Unsupported generation setting: ' + key)
    if type(value.get('evaluation_batch_size')) is not int or not 1 <= value['evaluation_batch_size'] <= 64:
        raise ValueError('Evaluation batch size must be 1..64')
    if type(value.get('max_padded_tokens')) is not int or not 4096 <= value['max_padded_tokens'] <= 262144:
        raise ValueError('Invalid padded-token memory budget')
    return value


def batches(lengths, batch_size, max_padded_tokens, max_new_tokens):
    """Stable length buckets; return original indices so pairing stays explicit."""
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError('Invalid batching bounds')
    pending = []
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        length = lengths[index]
        if length < 1 or length + max_new_tokens > max_padded_tokens:
            raise ValueError('Prompt exceeds generation memory budget')
        if pending and (len(pending) == batch_size or
                        (length + max_new_tokens) * (len(pending) + 1) > max_padded_tokens):
            yield pending
            pending = []
        pending.append(index)
    if pending:
        yield pending


class RowSeededSampler:
    """HF logits processor: draw once per row, then force that sampled token.

HF's greedy dispatch selects our already sampled token; it is NOT greedy
decoding. Use HF's temperature -> top-k -> top-p filters in their usual order.
Explicit row generators isolate samples from other answers' EOS and padding.
Generators live only within a completed batch; checkpoints are between groups.
"""
    def __init__(self, seeds, device, temperature, top_k, top_p):
        import torch
        from transformers import (LogitsProcessorList, TemperatureLogitsWarper,
                                  TopKLogitsWarper, TopPLogitsWarper)
        self.generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        self.filters = LogitsProcessorList([
            TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k), TopPLogitsWarper(top_p)])

    def __call__(self, input_ids, scores):
        import torch
        if scores.shape[0] != len(self.generators):
            raise RuntimeError('Sampling batch changed unexpectedly')
        probabilities = self.filters(input_ids, scores).softmax(dim=-1)
        chosen = torch.stack([torch.multinomial(row, 1, generator=rng)
                              for row, rng in zip(probabilities, self.generators)])
        return torch.full_like(scores, -float('inf')).scatter_(1, chosen, 0.)


def trim_completion(tokens, eos_token_ids):
    """Keep first EOS, drop only subsequent batch padding (not prompt tokens)."""
    for i, token in enumerate(tokens):
        if token in eos_token_ids:
            return tokens[:i + 1]
    return tokens


def generate_batch(model, prompts, seeds, *, pad_token_id, eos_token_ids,
                   max_new_tokens, temperature=.6, top_k=20, top_p=.95):
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    if not prompts or len(prompts) != len(seeds) or any(not row for row in prompts):
        raise ValueError('Nonempty prompts and one seed per answer required')
    if pad_token_id is None or not eos_token_ids:
        raise ValueError('Explicit padding/EOS tokens required')
    device = next(model.parameters()).device
    width = max(map(len, prompts))
    # Build on CPU and transfer once, rather than issuing a GPU operation per prompt.
    ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(prompts):
        ids[i, -len(row):] = torch.tensor(row, dtype=torch.long)
        mask[i, -len(row):] = 1
    sampler = RowSeededSampler(seeds, device, temperature, top_k, top_p)
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                              pad_token_id=pad_token_id, eos_token_id=list(eos_token_ids),
                              use_cache=True, return_dict_in_generate=False)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                                    generation_config=config, use_model_defaults=False, use_cache=True,
                                    logits_processor=LogitsProcessorList([sampler]))
        return [trim_completion(row, eos_token_ids) for row in output[:, width:].tolist()]
    finally:
        model.train(was_training)
