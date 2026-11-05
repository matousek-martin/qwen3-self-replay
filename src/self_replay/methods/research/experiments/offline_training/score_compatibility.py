"""Exploratory frozen-student scoring of verified traces; NOT a ReDiST reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (MODEL, REVISION, Deadline, append_jsonl, encode_example, file_hash,
                    load_model, prompt_id, prompt_tokens, canonical_completion, prompt_has_open_think,
                    read_jsonl, sequence_logprob, write_json)
from score import is_task_correct, valid_reasoning


def answer_conditioned_example(tokenizer, messages, completion, answer, max_seq_len):
    # Exclude final-answer text: only prompt, reasoning, and a fixed answer header condition the gold label.
    prefix = prompt_tokens(tokenizer, messages)
    thinking = canonical_completion(completion, prompt_has_open_think(tokenizer, prefix)).split("</think>", 1)[0]
    prefix += tokenizer.encode(thinking + "</think>\n\nFinal answer:", add_special_tokens=False)
    target = tokenizer.encode(" " + str(answer), add_special_tokens=False)
    if not target or len(prefix) + len(target) > max_seq_len:
        raise ValueError("Empty answer or oversized conditional-answer sequence")
    return {"input_ids": prefix + target, "prompt_length": len(prefix), "completion_tokens": len(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--logit-chunk", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--include-completion-nll", action="store_true")
    args = parser.parse_args()
    prompts = {prompt_id(row): row for row in read_jsonl(args.prompts)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"model": MODEL, "revision": REVISION, "prompts_sha256": file_hash(args.prompts),
                "rollouts_sha256": file_hash(args.rollouts), "include_completion_nll": args.include_completion_nll,
                "method": "frozen_student_verified_answer_logprob_proxy", "reproduces_redist": False,
                "limitation": "Reasoning may already state the answer; this proxy does not establish teaching utility or causal understanding"}
    manifest_path = args.output.with_suffix(".manifest.json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing score provenance differs")
    if args.output.exists() and not manifest_path.exists():
        raise ValueError("Existing output lacks manifest")
    write_json(manifest_path, manifest)
    done = {(str(r["prompt_id"]), r["sample_id"]) for r in read_jsonl(args.output)} if args.output.exists() else set()
    deadline = Deadline(args.max_seconds)
    model, tokenizer = load_model()
    model.eval()
    import torch
    eligible = 0
    scored_this_run = 0
    for record in read_jsonl(args.rollouts):
        row = prompts[str(record["prompt_id"])]
        if record["truncated"] or not valid_reasoning(record["completion"]):
            continue
        if not is_task_correct(record["completion"], row["answer"], row["task"]):
            continue
        eligible += 1
        key = (str(record["prompt_id"]), record["sample_id"])
        if key in done:
            continue
        if deadline.expired:
            break
        example = answer_conditioned_example(tokenizer, row["prompt"], record["completion"], row["answer"], args.max_seq_len)
        with torch.no_grad():
            answer_logprob = sequence_logprob(model, example, args.logit_chunk).item()
            result = {"prompt_id": key[0], "sample_id": key[1], "total_tokens": record["total_tokens"],
                      "verified_answer_logprob": answer_logprob,
                      "verified_answer_mean_logprob": answer_logprob / example["completion_tokens"],
                      "answer_token_count": example["completion_tokens"], "model_revision": REVISION}
            if args.include_completion_nll:
                whole = encode_example(tokenizer, row["prompt"], record["completion"], args.max_seq_len)
                result["completion_mean_nll"] = -sequence_logprob(model, whole, args.logit_chunk).item() / whole["completion_tokens"]
        append_jsonl(args.output, result)
        done.add(key)
        scored_this_run += 1
    write_json(args.output.with_suffix(".progress.json"), {
        "scored_total": len(done), "scored_this_run": scored_this_run,
        "elapsed_seconds": deadline.elapsed, "stopped_by_deadline": deadline.expired,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "limitation": manifest["limitation"],
    })


if __name__ == "__main__":
    main()
