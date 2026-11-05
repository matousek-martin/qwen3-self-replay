"""Score held-out training validation rollouts, not the contest benchmarks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

from common import ROOT, file_hash, prompt_id, read_jsonl, write_json

sys.path.insert(0, str(ROOT / "ThinkingCap-Test"))
from utils.train.rewards import is_task_correct


def valid_reasoning(text):
    if text.count("</think>") != 1:
        return False
    thinking, answer = text.split("</think>", 1)
    thinking = thinking.removeprefix("<think>").strip()
    return bool(thinking and answer.strip()) and "<think>" not in thinking and "<think>" not in answer


def score_rows(prompts, rollouts, expected_samples=0):
    by_id = {prompt_id(row): row for row in prompts}
    if len(by_id) != len(prompts):
        raise ValueError("Duplicate prompt IDs")
    groups = defaultdict(list)
    seen = set()
    for record in rollouts:
        key = (str(record["prompt_id"]), record["sample_id"])
        if key in seen or key[0] not in by_id:
            raise ValueError("Duplicate or unknown rollout prompt/sample")
        if not isinstance(key[1], int) or key[1] < 0 or (expected_samples and key[1] >= expected_samples):
            raise ValueError("Unexpected sample index")
        seen.add(key)
        row = by_id[key[0]]
        text = record["completion"]
        malformed = not valid_reasoning(text)
        truncated = bool(record["truncated"])
        answer_correct = bool(is_task_correct(text, row["answer"], row["task"]))
        result = {"correct": answer_correct and not truncated and not malformed,
                  "answer_correct": answer_correct and not truncated,
                  "total_tokens": record["total_tokens"], "truncated": truncated,
                  "malformed": malformed}
        groups[row["source"]].append(result)
        groups["all"].append(result)
    counts = defaultdict(int)
    for row_id, sample in seen:
        counts[row_id] += 1
    if expected_samples and (set(counts) != set(by_id) or any(v != expected_samples for v in counts.values())):
        raise ValueError("Missing or extra samples; refusing incomplete validation score")
    summaries = {}
    for source, values in groups.items():
        n = len(values)
        correct_lengths = [v["total_tokens"] for v in values if v["correct"]]
        wrong_lengths = [v["total_tokens"] for v in values if not v["correct"]]
        summaries[source] = {
            "responses": n, "accuracy": sum(v["correct"] for v in values) / n,
            "answer_accuracy_without_thinking_format_gate": sum(v["answer_correct"] for v in values) / n,
            "mean_total_tokens": sum(v["total_tokens"] for v in values) / n,
            "truncation_rate": sum(v["truncated"] for v in values) / n,
            "malformed_reasoning_rate": sum(v["malformed"] for v in values) / n,
            "mean_correct_tokens": sum(correct_lengths) / len(correct_lengths) if correct_lengths else None,
            "mean_wrong_tokens": sum(wrong_lengths) / len(wrong_lengths) if wrong_lengths else None,
        }
    return {"scope": "training-source validation, not contest evaluation", "sources": summaries,
            "prompts_scored": len(counts), "prompts_expected": len(prompts),
            "samples_per_prompt": dict(sorted(counts.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=1)
    args = parser.parse_args()
    result = score_rows(read_jsonl(args.prompts), read_jsonl(args.rollouts), args.expected_samples)
    result.update(prompt_sha256=file_hash(args.prompts), rollout_sha256=file_hash(args.rollouts))
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
