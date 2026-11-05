"""Pinned vLLM 0.11.0 fix: retain accepted LoRA routing metadata in outputs.

The official evaluator's routing check remains unchanged. No model, scheduler,
sampling, tokens or grading code is modified. See the matching upstream file:
https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/engine/output_processor.py
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path

ORIGINAL_SHA = '1999e687a183ce4b7b039b4c5357b73f0ad0afd8a5b52383ef1f7d1d5613263e'
PATCHED_SHA = '5fbb95d32f478465df340b867d5bd59f0bd3f3fc0a20603534e82618d8297e3a'
CHANGES = [
    ('        self.lora_name = lora_name\n',
     '        self.lora_name = lora_name\n        self.lora_request = None\n'),
    ('        return cls(\n            request_id=request.request_id,',
     '        state = cls(\n            request_id=request.request_id,'),
    ('            log_stats=log_stats,\n        )\n\n    def make_request_output',
     '            log_stats=log_stats,\n        )\n        state.lora_request = request.lora_request\n        return state\n\n    def make_request_output'),
    ('            num_cached_tokens=self.num_cached_tokens,\n',
     '            num_cached_tokens=self.num_cached_tokens,\n            lora_request=self.lora_request,\n'),
]


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def patch_source(source):
    if digest(source) == PATCHED_SHA:
        return source
    if digest(source) != ORIGINAL_SHA:
        raise ValueError('Unrecognized vLLM output processor; refusing an unverified patch')
    for before, after in CHANGES:
        if source.count(before) != 1:
            raise ValueError('vLLM patch anchor is not unique')
        source = source.replace(before, after)
    if digest(source) != PATCHED_SHA:
        raise ValueError('Patched vLLM source hash differs')
    compile(source, 'vllm/v1/engine/output_processor.py', 'exec')
    return source


def install(out):
    package = metadata.distribution('vllm')
    if package.version != '0.11.0':
        raise ValueError('Compatibility patch requires exactly vLLM 0.11.0')
    path = Path(package.locate_file('vllm/v1/engine/output_processor.py'))
    original = path.read_text()
    patched = patch_source(original)
    if original != patched:
        backup = path.with_suffix('.py.thinkingcap-original')
        if backup.exists() and digest(backup.read_text()) != ORIGINAL_SHA:
            raise ValueError('Unexpected existing vLLM backup')
        backup.write_text(original)
        temporary = path.with_suffix('.py.thinkingcap-tmp')
        temporary.write_text(patched)
        temporary.replace(path)
    result = {'package': 'vllm', 'version': package.version, 'source': str(path),
              'original_sha256': ORIGINAL_SHA, 'patched_sha256': PATCHED_SHA,
              'scope': 'Propagate accepted request LoRA metadata into RequestOutput; generation and evaluator checks unchanged'}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    print(json.dumps(install(parser.parse_args().out)))
