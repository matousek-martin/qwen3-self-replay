"""Deterministic, distinct RNG streams for repeated colocated vLLM requests.

TRL 0.24 uses n=1 with duplicated prompts in colocate mode. A single explicit
SamplingParams.seed therefore repeats the SAME sample, not eight independent
samples. Keep n=1 and give every request its own reproducible seed instead.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json


def independent_parameters(prompts, parameters, *, seed, phase, step):
    if not all(isinstance(p, str) for p in prompts):
        raise ValueError('Text-only sampling adapter')
    if parameters.n != 1:
        raise ValueError('This adapter matches TRL colocate n=1 requests')
    counts, used, result = Counter(), set(), []
    for prompt in prompts:
        ordinal = counts[prompt]
        counts[prompt] += 1
        key = json.dumps([seed, phase, step if phase == 'train' else 0, prompt, ordinal])
        value = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**31 - 1)
        while value in used:
            value = (value + 1) % (2**31 - 1)
        used.add(value)
        item = deepcopy(parameters)
        item.seed = value
        result.append(item)
    return result
