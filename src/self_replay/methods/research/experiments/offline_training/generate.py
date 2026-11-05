"""Generate resumable training or validation rollouts using pinned Qwen3 weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from common import (MODEL, REVISION, Deadline, append_jsonl, file_hash, generation_seed,
                    load_model, prompt_id, prompt_tokens, read_jsonl, write_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=16, help="Prompts per vLLM batch")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="0 runs until all requested rollouts complete; positive soft cap is checked between vLLM batches; SIGTERM still requests a graceful stop")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def make_record(row, index, text, tokens, finish_reason, seed, args):
    return {
        "prompt_id": prompt_id(row), "sample_id": index, "completion": text,
        "generated_token_ids": tokens, "total_tokens": len(tokens),
        "truncated": finish_reason != "stop", "finish_reason": finish_reason,
        "model_revision": REVISION, "model": MODEL,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "generation": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "max_new_tokens": args.max_new_tokens, "seed": seed,
                       "backend": args.backend, "enable_thinking": True},
    }


def generate_vllm(rows, done, args, deadline):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    engine = LLM(model=MODEL, revision=REVISION, tokenizer_revision=REVISION,
                 dtype="bfloat16", max_model_len=args.max_model_len,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 enable_lora=bool(args.adapter), max_lora_rank=64,
                 seed=args.seed, trust_remote_code=False)
    request = LoRARequest("candidate", 1, str(args.adapter.resolve())) if args.adapter else None
    pending = [r for r in rows if any((prompt_id(r), i) not in done for i in range(args.samples))]
    for offset in range(0, len(pending), args.batch_size):
        if deadline.expired:
            break
        batch = pending[offset:offset + args.batch_size]
        inputs = [{"prompt_token_ids": prompt_tokens(tokenizer, r["prompt"])} for r in batch]
        if any(len(p["prompt_token_ids"]) + args.max_new_tokens > args.max_model_len for p in inputs):
            raise ValueError("Prompt + generation cap exceeds --max-model-len; refusing prompt truncation")
        seeds = [generation_seed(args.seed, prompt_id(r), 0) for r in batch]
        params = [SamplingParams(n=args.samples, temperature=0.6, top_p=0.95, top_k=20,
                                 max_tokens=args.max_new_tokens, seed=s) for s in seeds]
        results = engine.generate(inputs, params, lora_request=request, use_tqdm=False)
        for row, seed, result in zip(batch, seeds, results):
            for answer in result.outputs:
                key = (prompt_id(row), answer.index)
                if key not in done:
                    record = make_record(row, answer.index, answer.text, list(answer.token_ids),
                                         answer.finish_reason, seed, args)
                    append_jsonl(args.output, record)
                    done.add(key)
        write_json(args.output.with_suffix(".progress.json"),
                   {"responses": len(done), "expected": len(rows) * args.samples,
                    "elapsed_seconds": deadline.elapsed, "backend": "vllm"})
        print(f"rollouts {len(done)}/{len(rows) * args.samples}", flush=True)


def generate_transformers(rows, done, args, deadline):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class TimeStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return deadline.expired

    model, tokenizer = load_model(args.adapter)
    model.eval()
    for row in rows:
        ids = prompt_tokens(tokenizer, row["prompt"])
        if len(ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError("Prompt + generation cap exceeds --max-model-len")
        tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
        for sample_id in range(args.samples):
            key = (prompt_id(row), sample_id)
            if key in done:
                continue
            if deadline.expired:
                return
            seed = generation_seed(args.seed, prompt_id(row), sample_id)
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                        do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                        max_new_tokens=args.max_new_tokens,
                                        pad_token_id=tokenizer.pad_token_id,
                                        stopping_criteria=StoppingCriteriaList([TimeStop()]))
            tokens = output[0, len(ids):].tolist()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            reason = "stop" if tokens and tokens[-1] in eos else "deadline" if deadline.expired else "length"
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            append_jsonl(args.output, make_record(row, sample_id, text, tokens, reason, seed, args))
            done.add(key)
            print(f"rollouts {len(done)}/{len(rows) * args.samples}", flush=True)


def main():
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("samples, batch-size, and max-new-tokens must be positive")
    if args.limit < 0 or not math.isfinite(args.max_seconds) or args.max_seconds < 0:
        raise ValueError("limit must be nonnegative; max-seconds must be finite and nonnegative (0 disables the deadline)")
    rows = read_jsonl(args.prompts)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({prompt_id(r) for r in rows}) != len(rows):
        raise ValueError("Prompts must be nonempty and have unique IDs")
    if args.adapter:
        config = json.loads((args.adapter / "adapter_config.json").read_text())
        if config.get("base_model_name_or_path") != MODEL or config.get("revision") != REVISION:
            raise ValueError("Adapter must match pinned model and revision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {"model": MODEL, "revision": REVISION, "prompt_sha256": file_hash(args.prompts),
                "adapter_sha256": file_hash(args.adapter / "adapter_model.safetensors") if args.adapter else None,
                "samples": args.samples, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                "max_model_len": args.max_model_len, "backend": args.backend, "limit": args.limit,
                "temperature": 0.6, "top_p": 0.95, "top_k": 20, "enable_thinking": True}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing rollout manifest differs; use a new output path")
    if args.output.exists() and not manifest_path.exists():
        raise ValueError("Existing rollout file has no provenance manifest")
    write_json(manifest_path, manifest)
    done = set()
    allowed = {prompt_id(r) for r in rows}
    for record in read_jsonl(args.output) if args.output.exists() else []:
        key = (str(record["prompt_id"]), record["sample_id"])
        if key in done or key[0] not in allowed or not 0 <= key[1] < args.samples:
            raise ValueError("Invalid/duplicate sample in existing rollout output")
        done.add(key)
    deadline = Deadline(args.max_seconds)
    if len(done) != len(rows) * args.samples:
        (generate_vllm if args.backend == "vllm" else generate_transformers)(rows, done, args, deadline)
    versions = {}
    for package in ("torch", "transformers", "peft", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write_json(args.output.with_suffix(".progress.json"),
               {"responses": len(done), "expected": len(rows) * args.samples,
                "complete": len(done) == len(rows) * args.samples,
                "elapsed_seconds": deadline.elapsed, "versions": versions})


if __name__ == "__main__":
    main()
