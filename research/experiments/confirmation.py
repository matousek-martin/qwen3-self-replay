"""Freeze benchmark partitions and audit paired official evaluation outputs.

This module never trains, generates, rents GPUs, or modifies official results.
Development scores can select checkpoints; confirmation is read only after a
checkpoint hash is locked. A confidence interval crossing zero is inconclusive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ThinkingCap-Test"))
from utils.eval.recipe import SAMPLING, SEED, token_cap
from utils.eval.stats import compare_macro, compare_runs, summarize

DOMAINS = ("bfcl", "ifeval", "mbpp", "svamp")
BFCL = ("bfcl_irrelevance", "bfcl_multiple", "bfcl_simple_python")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_line(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def rows_digest(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(canonical_line(row).encode())
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value, immutable=False):
    path = Path(path)
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and immutable:
        if path.read_text() != serialized:
            raise ValueError(f"Refusing to replace frozen protocol artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized)
    temporary.replace(path)


def domain(dataset):
    result = "bfcl" if dataset in BFCL else dataset
    if result not in DOMAINS:
        raise ValueError(f"Unexpected benchmark dataset: {dataset}")
    return result


def key(row):
    return str(row["dataset"]), str(row["id"])


def allocate(total, counts):
    """Largest-remainder proportional allocation with deterministic ties."""
    denominator = sum(counts.values())
    raw = {name: total * count / denominator for name, count in counts.items()}
    result = {name: int(value) for name, value in raw.items()}
    order = sorted(raw, key=lambda name: (-(raw[name] - result[name]), name))
    for name in order[:total - sum(result.values())]:
        result[name] += 1
    if any(not 0 < result[name] < counts[name] for name in counts):
        raise ValueError("Every benchmark stratum needs at least one tuning and confirmation question")
    return result


def create_split(question_dir, tuning_per_domain=128, seed=20261023):
    question_dir = Path(question_dir)
    source = question_dir / "questions.jsonl"
    manifest = question_dir / "manifest.json"
    source_meta = json.loads(manifest.read_text())
    if digest(source) != source_meta["cohort_sha256"]:
        raise ValueError("Frozen source questions do not match their original manifest")
    questions = read_jsonl(source)
    if len({key(row) for row in questions}) != len(questions):
        raise ValueError("Duplicate source question IDs")
    if len({row["seed_index"] for row in questions}) != len(questions):
        raise ValueError("Duplicate source seed indices")
    by_domain = defaultdict(list)
    for row in questions:
        by_domain[domain(row["dataset"])].append(row)
    if set(by_domain) != set(DOMAINS):
        raise ValueError("Source must contain all four domains")
    tuning = set()
    for name in DOMAINS:
        strata = defaultdict(list)
        for row in by_domain[name]:
            strata[row["dataset"]].append(row)
        quotas = allocate(tuning_per_domain, {d: len(rows) for d, rows in strata.items()})
        for dataset, rows in strata.items():
            # Selection depends only on seed and question identity, never answers, output lengths or scores.
            ordered = sorted(rows, key=lambda row: hashlib.sha256(
                f"thinkingcap-confirmation-v1:{seed}:{dataset}:{row['id']}".encode()).hexdigest())
            tuning.update(key(row) for row in ordered[:quotas[dataset]])
    subsets = {}
    for part in ("tuning", "confirmation"):
        selected = [row for row in questions if (key(row) in tuning) == (part == "tuning")]
        subsets[part] = {
            "keys": [{"dataset": row["dataset"], "id": row["id"], "seed_index": row["seed_index"]}
                     for row in selected],
            "counts": dict(sorted(Counter(row["dataset"] for row in selected).items())),
            "domain_counts": dict(sorted(Counter(domain(row["dataset"]) for row in selected).items())),
            "total": len(selected), "questions_sha256": rows_digest(selected),
        }
    return {
        "schema_version": 1, "source_cohort_sha256": digest(source),
        "source_manifest_sha256": digest(manifest), "source_total": len(questions),
        "seed": seed, "tuning_per_domain": tuning_per_domain, "subsets": subsets,
        "policy": {
            "tuning_samples": 2, "confirmation_samples": 8,
            "strict_margin_pp": 0.0, "screening_margin_pp": 1.0,
            "screening_rule": "Macro point accuracy delta >= -1pp; CI classification reported independently",
            "strict_rule": "Paired 95% cluster CI lower accuracy bound >= 0; degenerate equality remains inconclusive",
            "confirmation": "Lock candidate hashes before inspecting any confirmation candidate outcomes; do not refine or reselect on confirmation",
            "training": "Neither partition nor its generations may be used as training examples",
            "baseline": "Generate a fresh baseline under each exact subset and sample count; do not reuse a slice of the old full run as a matched run",
            "aggregation": "BFCL pooled by response count; four-domain accuracy arithmetic mean, token ratios geometric mean",
            "serialization": "UTF-8 canonical_line: ensure_ascii=False, sort_keys=True, separators=(',',':'), LF",
        },
    }


def subset_keys(manifest, part):
    return {key(row) for row in manifest["subsets"][part]["keys"]}


def load_split(path, canonical_question_path=None):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported split schema")
    tuning, confirmation = (subset_keys(manifest, part) for part in ("tuning", "confirmation"))
    if tuning & confirmation or len(tuning | confirmation) != manifest["source_total"]:
        raise ValueError("Split overlaps or has missing source questions")
    for part in ("tuning", "confirmation"):
        info = manifest["subsets"][part]
        if len(subset_keys(manifest, part)) != info["total"] or len(info["keys"]) != info["total"]:
            raise ValueError("Split contains duplicate keys or wrong totals")
        counts = dict(sorted(Counter(row["dataset"] for row in info["keys"]).items()))
        if counts != info["counts"]:
            raise ValueError("Split stratum counts do not match keys")
    if canonical_question_path:
        source = Path(canonical_question_path)
        if digest(source) != manifest["source_cohort_sha256"]:
            raise ValueError("Source question hash changed")
        rows = read_jsonl(source)
        for part in ("tuning", "confirmation"):
            selected = [row for row in rows if key(row) in subset_keys(manifest, part)]
            if rows_digest(selected) != manifest["subsets"][part]["questions_sha256"]:
                raise ValueError("Derived subset hash mismatch")
    return manifest


def adapter_hashes(directory):
    directory = Path(directory)
    if not (directory / "adapter_config.json").is_file() or not list(directory.glob("*.safetensors")):
        raise ValueError("Adapter must contain adapter_config.json and safetensors weights")
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Locked adapters cannot contain symlinks")
    return {file.relative_to(directory).as_posix(): digest(file) for file in paths if file.is_file()}


def lock_candidate(split_path, candidate, adapter, primary_candidate=None):
    load_split(split_path)
    return {"schema_version": 1, "candidate": candidate, "primary_candidate": primary_candidate or candidate,
            "adapter_files_sha256": adapter_hashes(adapter),
            "split_manifest_sha256": digest(split_path),
            "locked_at": datetime.now(timezone.utc).isoformat()}


def preservation(delta, interval, margin_pp=0.0):
    """Statistical evidence differs from both a point screen and a guarantee."""
    threshold = -margin_pp / 100
    if interval is None:
        conclusion = "inconclusive_no_interval"
    elif interval[0] == interval[1]:
        conclusion = "inconclusive_degenerate_bootstrap"
    elif interval[0] >= threshold:
        conclusion = "supported_at_stated_margin"
    elif interval[1] < threshold:
        conclusion = "evidence_of_regression_beyond_margin"
    else:
        conclusion = "inconclusive"
    return {"margin_pp": margin_pp, "classification": conclusion,
            "point_estimate_within_margin": delta >= threshold,
            "note": "Finite-sample evidence, not a guarantee; nonsignificance does not establish preservation"}


def _index_records(rows, expected, samples):
    index = {}
    for row in rows:
        qkey = key(row)
        sample = row.get("sample_index")
        if type(sample) is not int or not 0 <= sample < samples:
            raise ValueError("Unexpected sample index")
        record_key = (*qkey, sample)
        if record_key in index:
            raise ValueError("Duplicate evaluation record")
        if qkey not in expected:
            raise ValueError("Evaluation contains questions outside the requested partition")
        if type(row.get("correct")) is not bool:
            raise ValueError("Ungraded or invalid correctness; regrade before comparing")
        if type(row.get("output_tokens")) is not int or row["output_tokens"] < 1:
            raise ValueError("Invalid generated token count")
        thinking = row.get("reasoning_tokens")
        if type(thinking) is not int or not 0 <= thinking <= row["output_tokens"]:
            raise ValueError("Missing or invalid reasoning token count")
        if type(row.get("truncated")) is not bool:
            raise ValueError("Missing truncation verdict")
        if row.get("seed") != SEED + expected[qkey] * 1000 + sample:
            raise ValueError("Record generation seed differs from the pinned recipe")
        index[record_key] = row
    wanted = {(*qkey, sample) for qkey in expected for sample in range(samples)}
    if set(index) != wanted:
        raise ValueError(f"Incomplete evaluation: {len(index)} rows, expected {len(wanted)}")
    return index


def thinking_comparison(base_mean, candidate_mean, aggregation):
    """Describe measured thinking tokens separately from total output savings."""
    return {
        "field": "reasoning_tokens", "aggregation": aggregation,
        "base_mean": base_mean, "candidate_mean": candidate_mean,
        "mean_delta": candidate_mean - base_mean,
        "reduction": 1 - candidate_mean / base_mean if base_mean else None,
        "includes_incorrect_and_truncated": True,
        "inference": "Descriptive point estimates; no thinking-token confidence interval computed",
    }


def check_runs(base_meta, candidate_meta, manifest, part, base_rows, candidate_rows):
    info = manifest["subsets"][part]
    samples = manifest["policy"][f"{part}_samples"]
    expected = {key(row): row["seed_index"] for row in info["keys"]}
    required = {
        "model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "cohort_sha256": info["questions_sha256"], "cohort_total": info["total"],
        "samples": samples, "seed": SEED, "sampling": SAMPLING,
        "max_tokens": {name: token_cap(name) for name in info["counts"]},
        "max_model_len": max(token_cap(name) for name in info["counts"]) + 4096,
    }
    for side, meta in (("base", base_meta), ("candidate", candidate_meta)):
        differences = [field for field, value in required.items() if meta.get(field) != value]
        if differences:
            raise ValueError(f"{side} metadata differs from frozen subset/recipe: {', '.join(differences)}")
        if set(meta.get("benchmarks", [])) != set(DOMAINS) or meta.get("failed_benchmarks"):
            raise ValueError(f"{side} does not have all four successfully graded domains")
        if meta.get("smoke"):
            raise ValueError("Smoke outputs cannot establish accuracy")
    if not candidate_meta.get("adapter") or base_meta.get("adapter") == candidate_meta.get("adapter"):
        raise ValueError("Expected a candidate adapter distinct from the reference checkpoint")
    for field in ("revisions", "selection", "sandbox"):
        if base_meta.get(field) != candidate_meta.get(field):
            raise ValueError(f"Run metadata mismatch: {field}")
    left = _index_records(base_rows, expected, samples)
    right = _index_records(candidate_rows, expected, samples)
    order = sorted(left)
    return [left[k] for k in order], [right[k] for k in order]


def analyze(base_dir, candidate_dir, split_path, part="tuning", resamples=2000,
            bootstrap_seed=42, lock_path=None, adapter=None):
    manifest = load_split(split_path)
    lock = None
    if part == "confirmation":
        if not lock_path or not adapter:
            raise ValueError("Confirmation analysis requires --lock and --adapter")
        lock = json.loads(Path(lock_path).read_text())
        if not lock.get("primary_candidate"):
            raise ValueError("Confirmation lock must identify the single preselected primary candidate")
        if lock["split_manifest_sha256"] != digest(split_path) or lock["adapter_files_sha256"] != adapter_hashes(adapter):
            raise ValueError("Candidate or split changed after confirmation lock")
    base_dir, candidate_dir = Path(base_dir), Path(candidate_dir)
    base_meta = json.loads((base_dir / "results.json").read_text())
    candidate_meta = json.loads((candidate_dir / "results.json").read_text())
    if part == "confirmation":
        study = candidate_meta.get("study", {})
        if study.get("candidate") != lock["candidate"] or study.get("adapter_files_sha256") != lock["adapter_files_sha256"] or study.get("split_manifest_sha256") != lock["split_manifest_sha256"]:
            raise ValueError("Candidate evaluation provenance does not match the locked adapter and split")
    left, right = check_runs(base_meta, candidate_meta, manifest, part,
                             read_jsonl(base_dir / "records.jsonl"), read_jsonl(candidate_dir / "records.jsonl"))
    groups, results = [], {}
    for name in DOMAINS:
        before = [row for row in left if domain(row["dataset"]) == name]
        after = [row for row in right if domain(row["dataset"]) == name]
        # Prefix IDs to make BFCL clustering robust if source subsets ever reuse bare IDs.
        before = [{**row, "id": row["dataset"] + ":" + row["id"]} for row in before]
        after = [{**row, "id": row["dataset"] + ":" + row["id"]} for row in after]
        groups.append((before, after))
        a, b = summarize(before), summarize(after)
        intervals = compare_runs(before, after, resamples=resamples, seed=bootstrap_seed)
        delta = b["accuracy"] - a["accuracy"]
        results[name] = {
            "base": a, "candidate": b, "accuracy_delta": delta,
            "token_reduction": 1 - b["total_output_tokens"] / a["total_output_tokens"],
            "thinking_tokens": thinking_comparison(a["mean_reasoning_tokens"], b["mean_reasoning_tokens"],
                                                   "ratio_of_domain_response_means"),
            "intervals": intervals, "strict_preservation": preservation(delta, intervals["accuracy_delta"], 0),
            "one_pp_noninferiority": preservation(delta, intervals["accuracy_delta"], 1),
            "material_point_decline_over_one_pp": delta < -0.01,
        }
    intervals = compare_macro(groups, resamples=resamples, seed=bootstrap_seed)
    delta = sum(result["accuracy_delta"] for result in results.values()) / 4
    ratios = [1 - result["token_reduction"] for result in results.values()]
    saved = 1 - math.exp(sum(math.log(ratio) for ratio in ratios) / 4)
    strict = preservation(delta, intervals["accuracy_delta"], 0)
    macro = {
        "base_accuracy": sum(result["base"]["accuracy"] for result in results.values()) / 4,
        "candidate_accuracy": sum(result["candidate"]["accuracy"] for result in results.values()) / 4,
        "accuracy_delta": delta, "token_reduction": saved, "intervals": intervals,
        "thinking_tokens": thinking_comparison(
            sum(result["base"]["mean_reasoning_tokens"] for result in results.values()) / 4,
            sum(result["candidate"]["mean_reasoning_tokens"] for result in results.values()) / 4,
            "ratio_of_equal_domain_means"),
        "strict_preservation": strict,
        "one_pp_noninferiority": preservation(delta, intervals["accuracy_delta"], 1),
        "passes_exploratory_point_screen": delta >= -0.01 and saved > 0,
    }
    return {
        "schema_version": 1, "scope": part, "samples": manifest["policy"][f"{part}_samples"],
        "question_count": manifest["subsets"][part]["total"], "domains": results, "macro": macro,
        "split_manifest_sha256": digest(split_path),
        "input_sha256": {f"{side}/{filename}": digest(path / filename)
                          for side, path in (("base", base_dir), ("candidate", candidate_dir))
                          for filename in ("results.json", "records.jsonl")},
        "candidate_lock": lock,
        "decision_flags": {
            "macro_zero_margin_supported": strict["classification"] == "supported_at_stated_margin",
            "domains_with_material_point_decline_over_one_pp": [
                name for name, result in results.items() if result["material_point_decline_over_one_pp"]],
            "domains_with_evidence_of_any_accuracy_decline": [
                name for name, result in results.items()
                if result["strict_preservation"]["classification"] == "evidence_of_regression_beyond_margin"],
            "all_domains_individually_support_zero_margin": all(
                result["strict_preservation"]["classification"] == "supported_at_stated_margin"
                for result in results.values()),
        },
        "inference_role": ("confirmatory_primary" if lock["candidate"] == lock["primary_candidate"]
                           else "exploratory_secondary_unadjusted") if lock else "exploratory_tuning",
        "comparison_reference": {"model": base_meta["model"], "adapter": base_meta.get("adapter"),
                                 "kind": "trained_control" if base_meta.get("adapter") else "unmodified_base"},
        "interpretation": {
            "token_metrics": "token_reduction remains the registered geometric-mean total-output saving. thinking_tokens separately reports measured reasoning-token means and their reduction; its macro uses equal-domain arithmetic means and does not change screening or ranking.",
            "tuning": "Exploratory; selection and repeated comparisons invalidate an unadjusted confirmatory interpretation",
            "confirmation": "Only confirmatory for a checkpoint locked before outcomes were inspected; the software cannot prove prior noninspection",
            "across_domain_preservation_claim": "Do not infer blanket preservation from the macro average; inspect all domains. Per-domain intervals are marginal, not multiplicity-adjusted",
            "full_contest_evaluation": False,
            "combining_parts": "2-sample tuning plus 8-sample confirmation is not a full official 8-sample evaluation. Regenerate tuning at 8 matched samples for a separately labeled selected-on-tuning full report",
            "guarantee": "No finite benchmark can guarantee zero future accuracy loss",
        },
    }


def estimate(split_path, baseline_dir, progress_path, hourly_rate, part="tuning", runs=1):
    manifest = load_split(split_path)
    meta = json.loads((Path(baseline_dir) / "results.json").read_text())
    if meta["cohort_sha256"] != manifest["source_cohort_sha256"]:
        raise ValueError("Timing reference is not the source full benchmark")
    rows = read_jsonl(Path(baseline_dir) / "records.jsonl")
    wanted = subset_keys(manifest, part)
    total_tokens = sum(row["output_tokens"] for row in rows)
    subset_tokens = sum(row["output_tokens"] for row in rows if key(row) in wanted)
    ratio = subset_tokens / total_tokens * manifest["policy"][f"{part}_samples"] / meta["samples"]
    observed_seconds = json.loads(Path(progress_path).read_text())["elapsed_seconds"]
    variable = observed_seconds * ratio
    lower, upper = variable + 300, variable * 2 + 900
    return {
        "scope": part, "runs_including_new_baselines": runs, "hourly_rate_usd": hourly_rate,
        "observed_full_run_elapsed_seconds": observed_seconds, "observed_full_tokens": total_tokens,
        "scaled_baseline_token_fraction": ratio, "per_run_planning_seconds": [lower, upper],
        "total_planning_gpu_hours": [runs * lower / 3600, runs * upper / 3600],
        "total_planning_usd": [runs * lower / 3600 * hourly_rate, runs * upper / 3600 * hourly_rate],
        "limitations": "Heuristic, not a quote or hard bound: old run includes setup/recovery, hardware differs, adapter/candidate lengths and coding-grader runtimes vary. Includes an assumed 5-15 minute fixed overhead and up to 2x scaled variable time. Financial guard remains authoritative.",
    }


def commands(question_dir, base_out, candidate_out, adapter, split, part, lock=None):
    def shell(parts):
        return shlex.join([str(item) for item in parts])
    samples = load_split(split)["policy"][f"{part}_samples"]
    common = ["python", "evaluate.py", "--questions", question_dir, "--samples", samples,
              "--seed", SEED, "--benchmarks", "mbpp,bfcl,svamp,ifeval", "--workers", 4]
    analysis = ["python", "../experiments/confirmation.py", "compare", "--base", base_out,
                "--candidate", candidate_out, "--split", split, "--part", part,
                "--out", str(candidate_out) + "/confirmation-analysis.json"]
    if part == "confirmation":
        if not lock:
            raise ValueError("Confirmation command requires a candidate lock")
        analysis.extend(["--lock", lock, "--adapter", adapter])
    return {"cwd": "ThinkingCap-Test", "commands": [
        shell(common + ["--out", base_out]),
        shell(common + ["--out", candidate_out, "--adapter", adapter]),
        shell(["python", "compare.py", base_out, candidate_out, "--out", str(candidate_out) + "/official-comparison.json"]),
        shell(analysis)],
        "requirements": "question_dir must be staged from the named frozen partition with vendored graders, the original seed_index, and canonical question hash; same pinned software/device/dtype/sandbox for both runs. No --max-tokens override. Original full baseline is descriptive context, not the matched subset baseline."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("split")
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tuning-per-domain", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261023)
    p = sub.add_parser("lock")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--primary-candidate", help="Single primary fixed before confirmation; defaults to this candidate")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--part", choices=["tuning", "confirmation"], default="tuning")
    p.add_argument("--lock", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("estimate")
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--hourly-rate", type=float, required=True)
    p.add_argument("--part", choices=["tuning", "confirmation"], default="tuning")
    p.add_argument("--runs", type=int, default=1)
    p = sub.add_parser("commands")
    for name in ("questions", "base", "candidate", "adapter", "split"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--part", choices=["tuning", "confirmation"], default="tuning")
    p.add_argument("--lock")
    args = parser.parse_args()
    if args.action == "split":
        result = create_split(args.questions, args.tuning_per_domain, args.seed)
        write_json(args.output, result, immutable=True)
        print(json.dumps({part: {k: v for k, v in info.items() if k != "keys"}
                          for part, info in result["subsets"].items()}, indent=2))
    elif args.action == "lock":
        if args.output.exists():
            old = json.loads(args.output.read_text())
            if old["candidate"] != args.candidate or old.get("primary_candidate") != (args.primary_candidate or args.candidate) or old["adapter_files_sha256"] != adapter_hashes(args.adapter) or old["split_manifest_sha256"] != digest(args.split):
                raise ValueError("Refusing to change an existing candidate lock")
            result = old
        else:
            result = lock_candidate(args.split, args.candidate, args.adapter, args.primary_candidate)
        write_json(args.output, result, immutable=True)
        print(json.dumps(result, indent=2))
    elif args.action == "compare":
        if args.resamples < 200:
            raise ValueError("Use at least 200 resamples; default is official 2000")
        result = analyze(args.base, args.candidate, args.split, args.part, args.resamples,
                         args.bootstrap_seed, args.lock, args.adapter)
        write_json(args.out, result)
        print(json.dumps({"scope": result["scope"], "macro": result["macro"]}, indent=2))
    elif args.action == "estimate":
        print(json.dumps(estimate(args.split, args.baseline, args.progress, args.hourly_rate,
                                  args.part, args.runs), indent=2))
    else:
        print(json.dumps(commands(args.questions, args.base, args.candidate, args.adapter,
                                  args.split, args.part, args.lock), indent=2))


if __name__ == "__main__":
    main()
