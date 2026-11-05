"""Small, completion-only LoRA SFT and summed-log-probability DPO pilots."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import time

from common import (MODEL, REVISION, Deadline, append_jsonl, encode_example, file_hash,
                    load_model, read_jsonl, sequence_logprob, write_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["sft", "dpo"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--logit-chunk", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_manifest(args):
    result = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    for key in ("resume", "max_seconds", "output"):
        result.pop(key)
    result.update(model=MODEL, model_revision=REVISION, data_sha256=file_hash(args.data),
                  micro_batch_size=1, dtype="bfloat16", optimizer="AdamW", weight_decay=0.0,
                  lora_dropout=0.0, completion_only=True,
                  dpo_reference="pinned base with adapter disabled",
                  dpo_logprob_reduction="sum")
    return result


def save_checkpoint(model, tokenizer, optimizer, output, state):
    import torch

    checkpoint = output / f"checkpoint-{state['step']:06d}"
    temporary = output / f".checkpoint-{state['step']:06d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "state": state,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()},
               temporary / "trainer_state.pt")
    write_json(temporary / "state.json", state)
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    temporary.replace(checkpoint)
    write_json(output / "latest.json", {"checkpoint": checkpoint.name, **state})
    return checkpoint


def main():
    args = parse_args()
    if any(x < 1 for x in (args.steps, args.grad_accum, args.rank, args.max_seq_len,
                           args.logit_chunk, args.checkpoint_every)):
        raise ValueError("Step/count arguments must be positive")
    if args.learning_rate <= 0 or args.beta <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning-rate, beta, and max-grad-norm must be positive")
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("Training dataset is empty")
    if any(row.get("split", "train") != "train" for row in rows):
        raise ValueError("Refusing non-training rows")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = run_manifest(args)
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Run already exists; use --resume or a new output path")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume config/data differs from the original run")
    elif args.resume:
        raise ValueError("No original manifest to resume")
    write_json(manifest_path, manifest)

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(args.seed)
    deadline = Deadline(args.max_seconds)
    resume_path = None
    if args.resume:
        latest = json.loads((args.output / "latest.json").read_text())
        resume_path = args.output / latest["checkpoint"]
    model, tokenizer = load_model(resume_path, trainable=bool(resume_path))
    if resume_path is None:
        config = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                            task_type="CAUSAL_LM", bias="none", revision=REVISION)
        model = get_peft_model(model, config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    state = {"step": 0, "training_tokens": 0, "prompt_tokens": 0, "elapsed_seconds": 0.0}
    if resume_path:
        # Only our own local checkpoint is trusted; optimizer state is not accepted from external files.
        saved = torch.load(resume_path / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    previous_elapsed = state["elapsed_seconds"]
    prepared = []
    for index, row in enumerate(rows):
        try:
            if args.method == "sft":
                prepared.append(encode_example(tokenizer, row["prompt"], row["completion"], args.max_seq_len))
            else:
                if row["chosen"] == row["rejected"]:
                    raise ValueError("Identical DPO completions")
                prepared.append((encode_example(tokenizer, row["prompt"], row["chosen"], args.max_seq_len),
                                 encode_example(tokenizer, row["prompt"], row["rejected"], args.max_seq_len)))
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid training row {index}: {error}") from error
    write_json(args.output / "dataset-summary.json", {
        "rows": len(rows), "unique_prompts": len({r.get("prompt_id") for r in rows}),
        "source_counts": {source: sum(r.get("source", "unknown") == source for r in rows)
                          for source in sorted({r.get("source", "unknown") for r in rows})},
        "all_rows_encoded_without_truncation": True})
    torch.cuda.reset_peak_memory_stats()
    reference_cache = {}
    model.train()
    stop_reason = "completed"
    checkpoint = resume_path
    for step in range(state["step"], args.steps):
        if deadline.expired:
            stop_reason = "soft_deadline_or_signal"
            break
        optimizer.zero_grad(set_to_none=True)
        sampler = random.Random(args.seed + step)
        step_loss = 0.0
        step_tokens = 0
        prompt_count = 0
        margins = []
        completed_microbatches = 0
        started_step = time.monotonic()
        for microbatch in range(args.grad_accum):
            if deadline.expired:
                break
            index = sampler.randrange(len(prepared))
            example = prepared[index]
            if args.method == "sft":
                loss = -sequence_logprob(model, example, args.logit_chunk) / example["completion_tokens"]
                step_tokens += example["completion_tokens"]
                prompt_count += example["prompt_length"]
            else:
                chosen, rejected = example
                if index not in reference_cache:
                    model.eval()
                    with torch.no_grad(), model.disable_adapter():
                        ref_chosen = sequence_logprob(model, chosen, args.logit_chunk).item()
                        ref_rejected = sequence_logprob(model, rejected, args.logit_chunk).item()
                    reference_cache[index] = (ref_chosen, ref_rejected)
                    model.train()
                ref_chosen, ref_rejected = reference_cache[index]
                chosen_logprob = sequence_logprob(model, chosen, args.logit_chunk)
                rejected_logprob = sequence_logprob(model, rejected, args.logit_chunk)
                margin = (chosen_logprob - rejected_logprob) - (ref_chosen - ref_rejected)
                loss = -F.logsigmoid(args.beta * margin)
                margins.append(margin.detach().item())
                step_tokens += chosen["completion_tokens"] + rejected["completion_tokens"]
                prompt_count += chosen["prompt_length"] + rejected["prompt_length"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}; stopping before optimizer update")
            step_loss += loss.detach().item() / args.grad_accum
            (loss / args.grad_accum).backward()
            completed_microbatches += 1
        if completed_microbatches != args.grad_accum:
            optimizer.zero_grad(set_to_none=True)
            stop_reason = "soft_deadline_or_signal_partial_step_discarded"
            break
        norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        state.update(step=step + 1, training_tokens=state["training_tokens"] + step_tokens,
                     prompt_tokens=state["prompt_tokens"] + prompt_count,
                     elapsed_seconds=previous_elapsed + deadline.elapsed)
        metric = {**state, "loss": step_loss, "gradient_norm": float(norm),
                  "step_seconds": time.monotonic() - started_step,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
        if margins:
            metric["mean_dpo_logratio_margin"] = sum(margins) / len(margins)
        append_jsonl(args.output / "metrics.jsonl", metric)
        print(json.dumps(metric), flush=True)
        if state["step"] % args.checkpoint_every == 0:
            checkpoint = save_checkpoint(model, tokenizer, optimizer, args.output, state)
    state["elapsed_seconds"] = previous_elapsed + deadline.elapsed
    checkpoint = save_checkpoint(model, tokenizer, optimizer, args.output, state)
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}
    write_json(args.output / "result.json", {
        **state, "requested_steps": args.steps, "complete": state["step"] == args.steps,
        "stop_reason": stop_reason, "adapter": str(checkpoint.resolve()),
        "adapter_sha256": file_hash(checkpoint / "adapter_model.safetensors"),
        "data_sha256": manifest["data_sha256"], "versions": versions,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "validation": "Not evaluated by this command; use generate.py and score.py on held-out prompts",
    })


if __name__ == "__main__":
    main()
