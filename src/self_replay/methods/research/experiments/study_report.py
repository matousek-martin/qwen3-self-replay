"""Read-only development-study aggregation; never reads confirmation outcomes.

All preregistered approaches, controls and refinements remain visible when
pending or failed. Rankings are exploratory, not candidate locks or claims of
preserved accuracy. Cost estimates are never relabeled as invoices.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

try:
    from experiments.confirmation import analyze, digest, write_json
    from experiments import adapter_refinement
except ModuleNotFoundError:
    from confirmation import analyze, digest, write_json
    import adapter_refinement

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = [
    ("sft-short", "sft-random"), ("sft-compatible", "sft-short"),
    ("sft-compatible", "sft-random"), ("dpo", "sft-dpo-matched"),
    ("dr-grpo", "grpo"), ("rl-fixed", "dr-grpo"), ("rl-adaptive", "dr-grpo"),
    ("rl-difficulty", "dr-grpo"), ("rl-progressive", "dr-grpo"),
    ("rl-replay", "dr-grpo"), ("rl-replay", "rl-fixed"),
]


class Inputs:
    """Snapshot small JSON/text inputs once, recording the bytes actually read."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes, self.cache = {}, {}

    def label(self, path):
        path = Path(path).resolve()
        return path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else str(path)

    def path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def text(self, path):
        path = self.path(path).resolve()
        if path not in self.cache:
            raw = path.read_bytes()
            self.hashes[self.label(path)] = hashlib.sha256(raw).hexdigest()
            self.cache[path] = raw.decode()
        return self.cache[path]

    def json(self, path, default=None):
        path = self.path(path)
        return json.loads(self.text(path)) if path.is_file() else default

    def record_hash(self, path):
        path = self.path(path)
        value = digest(path)
        self.hashes[self.label(path)] = value
        return value


def dedupe(values):
    return list(dict.fromkeys(values))


def job_from_path(path, cloud):
    path, cloud = Path(path).resolve(), Path(cloud).resolve()
    return path.relative_to(cloud).parts[0] if path.is_relative_to(cloud) else None


def runtime_fingerprint(inputs, job_name, jobs, cloud):
    if not job_name:
        return {"job": None, "gpu": None, "software_sha256": None, "warnings": ["Job provenance unavailable"]}
    folder = cloud / job_name
    job = jobs.get(job_name, {})
    gpu_path, packages = folder / "output/gpu.txt", folder / "output/packages.txt"
    runtime_gpu, driver, software_sha = None, None, None
    if gpu_path.is_file():
        text = inputs.text(gpu_path)
        found = re.search(r"\|\s*\d+\s+(.+?)\s+(?:On|Off)\s*\|", text)
        if found:
            runtime_gpu = found.group(1).strip()
        version = re.search(r"Driver Version:\s*(\S+)", text)
        driver = version.group(1) if version else None
    if packages.is_file():
        lines = sorted(line.strip() for line in inputs.text(packages).splitlines()
                       if line.strip() and not line.startswith("#"))
        software_sha = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    pod = inputs.json(folder / "pod.json", {})
    compatibility = inputs.json(folder / "output/vllm-compat.json", {})
    gpu = runtime_gpu or job.get("gpu") or (pod.get("gpu") or {}).get("id")
    warnings = []
    if not runtime_gpu:
        warnings.append("Actual GPU name unavailable; provider allocation label used where available")
    if software_sha is None:
        warnings.append("Installed package fingerprint unavailable")
    return {"job": job_name, "gpu": gpu, "gpu_name_source": "nvidia-smi" if runtime_gpu else "provider_metadata",
            "driver": driver, "software_sha256": software_sha,
            "vllm_output_processor_sha256": compatibility.get("patched_sha256"),
            "image": pod.get("image"), "warnings": warnings}


def provenance_differences(left, right, prefix):
    warnings = []
    if left.get("job") and left.get("job") == right.get("job"):
        return warnings
    for field in ("gpu", "driver", "software_sha256"):
        a, b = left.get(field), right.get(field)
        if a is None or b is None:
            warnings.append(f"{prefix}: {field} comparability unverified")
        elif a != b:
            warnings.append(f"{prefix}: different {field} ({a} vs {b})")
    if left.get('vllm_output_processor_sha256') != right.get('vllm_output_processor_sha256'):
        warnings.append(f'{prefix}: different vLLM output-metadata compatibility patch provenance')
    return warnings


def cost_report(ledger, now):
    jobs = ledger.get("jobs", {})
    released, live, reserved, liability, allocated_rate_estimate = 0.0, 0.0, 0.0, 0.0, 0.0
    unknown_rate, unestimated_released, details = [], [], []
    for name, job in jobs.items():
        if job.get("released_at"):
            if "charged_estimate_usd" not in job:
                unestimated_released.append(name)
            cost = float(job.get("charged_estimate_usd", 0.0))
            released += cost
            liability += cost
            state = "released"
            accrued = cost
            job_liability = cost
        else:
            rate = job.get("budget_rate")
            if rate is None:
                unknown_rate.append(name)
            elapsed = max(0.0, now - job.get("created_at", now))
            accrued = elapsed / 3600 * float(rate or 0)
            reservation = float(job.get("reserved_usd", 0))
            job_liability = max(reservation, accrued)
            live += accrued
            reserved += reservation
            liability += job_liability
            state = "unreleased_or_uncertain"
        if job.get("rate") is not None and not job.get("safely_no_pod"):
            end = job.get("released_at", now)
            allocated_rate_estimate += max(0, end - job.get("created_at", end)) / 3600 * float(job["rate"])
        details.append({"job": name, "state": state, "provider_state": job.get("state"),
                        "conservative_accrued_estimate_usd": accrued,
                        "reserved_or_settled_liability_usd": job_liability,
                        "budget_rate_usd_per_hour": job.get("budget_rate"),
                        "allocated_rate_usd_per_hour": job.get("rate"),
                        "backup_verified": job.get("backup_verified", False)})
    return {"budget_ceiling_usd": ledger.get("ceiling_usd", 200), "invoice_verified": False,
            "invoice_total_usd": None, "released_conservative_estimate_usd": released,
            "live_accrued_conservative_estimate_usd": live,
            "live_reservations_usd": reserved, "total_conservative_liability_usd": liability,
            "remaining_under_ceiling_after_liability_usd": ledger.get("ceiling_usd", 200) - liability,
            "allocated_rate_elapsed_estimate_usd": allocated_rate_estimate,
            "unknown_budget_rate_jobs": unknown_rate, "unestimated_released_jobs": unestimated_released,
            "cost_accounting_complete": not unknown_rate and not unestimated_released, "jobs": details,
            "note": "Conservative local ledger estimates include failed attempts and uncertain live allocations. Allocated-rate elapsed estimate excludes invoice adjustments/storage and is not an invoice. Unreleased liabilities retain max(reservation, elapsed budget-rate estimate)."}


def training_info(name, adapters, training_campaign, jobs, cloud, inputs, refinements=None):
    registration = adapters.get(name)
    if name in (refinements or {}):
        parent = training_info('dpo-32', adapters, training_campaign, jobs, cloud, inputs)
        result = dict(parent, registered_checkpoint=registration, warnings=list(parent['warnings']))
        result['warnings'].append('Post-training adapter-strength refinement; no additional optimizer updates')
        try:
            if not registration:
                raise FileNotFoundError('Refinement not staged in the checkpoint registry')
            inputs.record_hash(inputs.path(refinements[name]['manifest']))
            proof = adapter_refinement.verify_registration(inputs.root, name, registration,
                                                            adapters['dpo-32'], refinements[name])
            result.update(state='post_training_refinement', refinement=proof)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.update(state='invalid_refinement', complete=False)
            result['warnings'].append(str(exc))
        return result
    attempts = [j for j in training_campaign.get("jobs", []) if j.get("arm") == name]
    job_names = [j["job"] for j in attempts]
    job_names += [j for j in jobs if j.startswith("study-" + name + "-") and j not in job_names]
    selected_source = inputs.path(registration["source"]) if registration else None
    if selected_source:
        selected_job = registration.get("job") or job_from_path(selected_source, cloud)
        if selected_job and selected_job not in job_names:
            job_names.append(selected_job)
    else:
        selected_job = max(job_names, key=lambda j: jobs.get(j, {}).get("created_at", 0)) if job_names else None
    result = {"state": "pending", "complete": False, "job": selected_job,
              "registered_checkpoint": registration, "attempts": [
                  {"job": j, "state": jobs.get(j, {}).get("state"), "campaign": next((a for a in attempts if a["job"] == j), None)}
                  for j in job_names], "warnings": []}
    training_dir = selected_source.parent if selected_source else cloud / selected_job / "output/training" / name if selected_job else None
    if training_dir:
        status = inputs.json(training_dir / "status.json")
        summary = inputs.json(training_dir / "result.json")
        config = inputs.json(training_dir / "manifest.json") or inputs.json(training_dir / "config.json")
        checkpoint_state = inputs.json(selected_source / "state.json") if selected_source else None
        if summary is not None:
            result.update(state="complete" if summary.get("complete") else "partial", complete=bool(summary.get("complete")),
                          parent_training_result=summary, selected_checkpoint_state=checkpoint_state, config=config)
            if checkpoint_state and checkpoint_state.get("step") != summary.get("step"):
                result["state"] = "retained_earlier_checkpoint"
                result["warnings"].append("Refinement uses an intentionally earlier checkpoint; parent run totals do not equal selected checkpoint compute")
        elif status is not None:
            done = status.get("status") == "completed" and status.get("completed_groups") == 128
            result.update(state="complete" if done else status.get("status", "partial"), complete=done,
                          training_status=status, config=config)
        elif selected_job:
            result["state"] = jobs.get(selected_job, {}).get("state", "pending")
        if selected_source:
            for field, filename in (("checkpoint_sha256", "adapter_model.safetensors"), ("config_sha256", "adapter_config.json")):
                file = selected_source / filename
                if not file.is_file():
                    result.update(state="invalid_checkpoint", complete=False)
                    result["warnings"].append("Registered checkpoint file missing: " + filename)
                elif registration.get(field) and inputs.record_hash(file) != registration[field]:
                    result.update(state="invalid_checkpoint", complete=False)
                    result["warnings"].append("Registered checkpoint hash mismatch: " + filename)
    result["runtime"] = runtime_fingerprint(inputs, selected_job, jobs, cloud)
    return result


def discover_batches(campaign, jobs, cloud):
    batches = []
    for batch in campaign.get("batches", []):
        if batch.get("subset", "tuning") != "tuning":
            continue
        batches.append({**batch, "discovered_from": "evaluation_campaign"})
    seen = {batch.get("job") for batch in batches}
    for name, job in jobs.items():
        # The development controller reserves this prefix. Do not scan arbitrary
        # archived study directories or confirmation jobs looking for results.
        if name.startswith("study-eval-") and name not in seen:
            batches.append({"job": name, "output": str(cloud / name / "output/study"),
                            "candidates": [], "success": False, "discovered_from": "active_development_ledger"})
    return sorted(batches, key=lambda batch: jobs.get(batch.get("job"), {}).get("created_at", 0))


def validate_candidate(inputs, output, name, provenance, split_path):
    run = output / "runs" / name
    status = inputs.json(output / "status" / (name + ".json"), {})
    if status.get("state") != "complete":
        return {"state": status.get("state", "pending"), "error": status.get("error")}, None
    for filename, field in (("results.json", "results_sha256"), ("records.jsonl", "records_sha256")):
        file = run / filename
        if not file.is_file() or not status.get(field) or inputs.record_hash(file) != status[field]:
            raise ValueError("Complete evaluation status does not match file hash: " + filename)
    meta = inputs.json(run / "results.json")
    study = meta.get("study", {})
    wanted = provenance.get("adapter_files_sha256", {}).get(name)
    if study.get("subset") != "tuning" or study.get("candidate") != name or study.get("split_manifest_sha256") != digest(split_path):
        raise ValueError("Candidate study provenance differs from development protocol")
    if study.get("adapter_files_sha256") != wanted:
        raise ValueError("Candidate adapter hashes differ from evaluation provenance")
    return {"state": "complete", "status": status, "adapter_files_sha256": wanted,
            "run": str(run)}, run


def pareto_names(rows):
    scored = [row for row in rows if row.get("analysis")]
    answer = []
    for row in scored:
        point = row["analysis"]["macro"]
        dominated = any(
            other["name"] != row["name"]
            and other["analysis"]["macro"]["accuracy_delta"] >= point["accuracy_delta"]
            and other["analysis"]["macro"]["token_reduction"] >= point["token_reduction"]
            and (other["analysis"]["macro"]["accuracy_delta"] > point["accuracy_delta"]
                 or other["analysis"]["macro"]["token_reduction"] > point["token_reduction"])
            for other in scored)
        if not dominated:
            answer.append(row["name"])
    return answer


def build_report(root=ROOT, resamples=2000, now=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    inputs = Inputs(root)
    study, cloud = root / "lab/experiments/study", root / "lab/experiments/cloud"
    protocol = inputs.json(study / "protocol.json")
    if protocol is None:
        raise ValueError("Missing registered study protocol")
    evaluation = inputs.json(study / "evaluation-campaign.json", {})
    training = inputs.json(study / "training-campaign.json", {})
    ledger = inputs.json(cloud / "ledger.json", {"jobs": {}, "ceiling_usd": protocol["budget_total_usd"]})
    jobs = ledger.get("jobs", {})
    data = root / "lab/experiments/data/study"
    adapters = inputs.json(data / "adapters.json", {})
    split_path = inputs.path(protocol["splits_manifest"])
    split = inputs.json(split_path)
    if split is None:
        raise ValueError("Missing frozen split")
    refinements = adapter_refinement.registered_specs(protocol)
    original = dedupe(protocol["required_approaches"] + protocol.get("offline_controls", [])
                     + protocol.get("initial_refinement_candidates", []))
    planned = original + list(refinements)
    rows = {name: {"name": name, "category": "approach" if name in protocol["required_approaches"]
                  else "control" if name in protocol.get("offline_controls", []) else "refinement",
                  "training": training_info(name, adapters, training, jobs, cloud, inputs, refinements),
                  "evaluation_state": "pending", "evaluation_attempts": [], "analysis": None, "warnings": []}
            for name in planned}
    batches = discover_batches(evaluation, jobs, cloud)
    batch_reports, available = [], {}
    for batch in batches:
        output = inputs.path(batch["output"])
        job_name = batch.get("job")
        summary = {"job": job_name, "state": jobs.get(job_name, {}).get("state"),
                   "archive_backup_verified": jobs.get(job_name, {}).get("backup_verified", False),
                   "declared_candidates": batch.get("candidates", []), "output": str(output)}
        provenance = inputs.json(output / "study-provenance.json")
        if provenance is None:
            summary["status"] = "waiting_for_development_provenance"
            batch_reports.append(summary)
            continue
        if provenance.get("subset") != "tuning":
            summary["status"] = "excluded_non_development_scope_no_outcomes_read"
            batch_reports.append(summary)
            continue
        if provenance.get("split_manifest_sha256") != digest(split_path):
            summary["status"] = "invalid_split_provenance"
            batch_reports.append(summary)
            continue
        runtime = runtime_fingerprint(inputs, job_name, jobs, cloud)
        summary.update(runtime=runtime, provenance=provenance,
                       status="development_outputs_available")
        try:
            baseline_info, baseline = validate_candidate(inputs, output, "base", provenance, split_path)
        except (ValueError, OSError, KeyError) as exc:
            baseline_info, baseline = {"state": "invalid", "error": str(exc)}, None
        summary["baseline"] = baseline_info
        names = [name for name in provenance.get("config", {}).get("candidates", []) if name in rows]
        for name in names:
            attempt = {"job": job_name, "output": str(output), "state": "pending"}
            try:
                info, run = validate_candidate(inputs, output, name, provenance, split_path)
                attempt.update(info)
                if run is not None and baseline is not None:
                    if name in adapters:
                        for field, filename in (("checkpoint_sha256", "adapter_model.safetensors"),
                                                ("config_sha256", "adapter_config.json")):
                            if not adapters[name].get(field) or adapters[name][field] != info["adapter_files_sha256"].get(filename):
                                raise ValueError("Evaluated " + filename + " differs from the registered checkpoint")
                    if name in refinements:
                        if not rows[name]['training']['complete']:
                            raise ValueError('Refinement lineage is not verified')
                        registered = adapter_refinement.timestamp(refinements[name]['registered_at'])
                        if any(registered >= adapter_refinement.timestamp(s['started_at'])
                               for s in (baseline_info['status'], info['status'])):
                            raise ValueError('Refinement registration must precede baseline and candidate generation')
                    compared_paths = [folder / file for folder in (baseline, run) for file in ("results.json", "records.jsonl")]
                    before = {str(path): digest(path) for path in compared_paths}
                    for folder, declared in ((baseline, baseline_info["status"]), (run, info["status"])):
                        for filename, field in (("results.json", "results_sha256"), ("records.jsonl", "records_sha256")):
                            if before[str(folder / filename)] != declared[field]:
                                raise ValueError("Evaluation changed after its complete status was read; retry after sync")
                    result = analyze(baseline, run, split_path, "tuning", resamples=resamples)
                    if any(digest(path) != before[str(path)] for path in compared_paths):
                        raise ValueError("Evaluation snapshot changed during analysis; retry after sync completes")
                    # Deterministically use the earliest valid attempt, never the most favorable result.
                    if name not in available:
                        rows[name].update(evaluation_state="complete", analysis=result)
                        available[name] = {"run": run, "batch": summary, "runtime": runtime,
                                           "provenance": provenance, "job": job_name,
                                           "record_hashes": {file: before[str(run / file)] for file in ("results.json", "records.jsonl")}}
                        rows[name]["evaluation_provenance"] = {"job": job_name, "runtime": runtime,
                                "adapter_files_sha256": info["adapter_files_sha256"],
                                "evaluator_files_sha256": provenance.get("evaluator_files_sha256")}
                        rows[name]["warnings"] += rows[name]["training"]["warnings"]
                    else:
                        attempt["selection_note"] = "Retained for audit; earliest valid attempt remains the reported estimate"
                elif run is not None:
                    attempt["state"] = "waiting_for_matched_baseline"
            except (ValueError, OSError, KeyError) as exc:
                attempt.update(state="invalid", error=str(exc))
            rows[name]["evaluation_attempts"].append(attempt)
            if rows[name]["evaluation_state"] != "complete":
                rows[name]["evaluation_state"] = attempt["state"]
        batch_reports.append(summary)
    if available:
        first = next(iter(available))
        for name, item in available.items():
            if name != first:
                if item["job"] != available[first]["job"]:
                    rows[name]["warnings"].append("Cross-candidate ranking spans different evaluation batches with separate fresh baselines; run-to-run variation limits interpretation")
                rows[name]["warnings"] += provenance_differences(item["runtime"], available[first]["runtime"],
                                                                  "cross-candidate ranking evaluation vs " + first)
                if item["provenance"].get("evaluator_files_sha256") != available[first]["provenance"].get("evaluator_files_sha256"):
                    rows[name]["warnings"].append("Cross-candidate ranking uses different evaluator code fingerprints")
    contrasts = []
    for candidate, reference in CONTRASTS + [(name, 'dpo-32') for name in refinements]:
        item = {"candidate": candidate, "reference": reference, "state": "pending",
                "analysis": None, "warnings": []}
        if candidate not in rows or reference not in rows:
            continue
        if candidate in available and reference in available:
            a, b = available[candidate], available[reference]
            if a["job"] != b["job"]:
                item["warnings"].append("Direct contrast spans different evaluation batches with separate fresh baselines; it does not isolate the refinement from run-to-run variation")
            item["warnings"] += provenance_differences(a["runtime"], b["runtime"], "evaluation")
            item["warnings"] += provenance_differences(rows[candidate]["training"]["runtime"],
                                                        rows[reference]["training"]["runtime"], "training")
            if a["provenance"].get("evaluator_files_sha256") != b["provenance"].get("evaluator_files_sha256"):
                item["warnings"].append("evaluation: different evaluator code fingerprints")
            item["warnings"].append("Matched questions/seeds do not imply matched compute or isolate hardware/software effects")
            try:
                for item_side in (a, b):
                    if any(digest(item_side["run"] / file) != expected for file, expected in item_side["record_hashes"].items()):
                        raise ValueError("Evaluation changed before controlled contrast; rebuild report after sync")
                item["analysis"] = analyze(b["run"], a["run"], split_path, "tuning", resamples=resamples)
                for item_side in (a, b):
                    if any(digest(item_side["run"] / file) != expected for file, expected in item_side["record_hashes"].items()):
                        raise ValueError("Evaluation changed during controlled contrast; rebuild report after sync")
                item["state"] = "complete_with_comparability_caveats" if len(item["warnings"]) > 1 else "complete"
            except (ValueError, OSError, KeyError) as exc:
                item.update(state="invalid", analysis=None, error=str(exc))
        contrasts.append(item)
    ordered_rows = [rows[name] for name in planned]
    eligible = [row for row in ordered_rows if row["analysis"] and row["training"]["complete"]
                and row["analysis"]["macro"]["passes_exploratory_point_screen"]]
    ranked = sorted(eligible, key=lambda row: (-row["analysis"]["macro"]["token_reduction"],
                                              -row["analysis"]["macro"]["accuracy_delta"], row["name"]))
    ranking = [{"rank": i + 1, "name": row["name"],
                "accuracy_delta_pp": row["analysis"]["macro"]["accuracy_delta"] * 100,
                "total_tokens_saved_percent": row["analysis"]["macro"]["token_reduction"] * 100,
                "thinking_tokens": row["analysis"]["macro"].get("thinking_tokens"),
                "strict_preservation": row["analysis"]["macro"]["strict_preservation"],
                "domain_warnings": row["analysis"].get("decision_flags", {}),
                "comparability_warnings": row["warnings"]}
               for i, row in enumerate(ranked)]
    valid_names = [row["name"] for row in ordered_rows if row["analysis"]]
    return {
        "schema_version": 1, "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "scope": "exploratory development benchmark only; confirmation outcomes not accessed",
        "protocol_registered_at": protocol.get("registered_at"), "planned_candidate_count": len(planned),
        "original_candidate_count": len(original), "additional_refinement_count": len(refinements),
        "completed_development_count": len(valid_names), "all_planned_development_complete": len(valid_names) == len(planned),
        "campaign_states": {"training": training.get("state"), "evaluation": evaluation.get("state")},
        "candidates": ordered_rows, "controlled_contrasts": contrasts, "evaluation_batches": batch_reports,
        "ranking": {"rule": "Training complete, macro point accuracy change >= -1pp and positive token savings; order by token savings, then accuracy. Primary selection is recorded separately in the candidate lock.",
                    "candidates": ranking, "incomplete_study": len(valid_names) != len(planned)},
        "pareto_frontier": {"candidate_names": pareto_names(ordered_rows),
                            "axes": ["macro_accuracy_delta", "macro_token_reduction"],
                            "population": "All valid completed development evaluations, including any diagnostically retained incomplete-training checkpoints",
                            "uncertainty": "Point-estimate frontier; ignores CI overlap, multiple comparisons and hardware differences"},
        "cost": cost_report(ledger, now), "input_sha256": dict(sorted(inputs.hashes.items())),
        "limitations": ["Development comparisons and ranking are exploratory and unadjusted for repeated selection.",
                        "A confidence interval containing zero does not establish accuracy preservation.",
                        "Domain declines remain visible even when the macro screening threshold passes.",
                        "Direct controlled contrasts require matching generation recipe and question coverage; differing GPUs/software are explicit caveats.",
                        "A complete candidate can be reported before its whole batch finishes; archived status/record hashes are verified but the batch backup may remain unverified.",
                        "This development report does not analyze confirmation outputs or create or alter candidate locks."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples < 200:
        raise ValueError("Use at least200 bootstrap resamples; default2000 matches the official recipe")
    output = args.out or args.root / "lab/experiments/study/development-report.json"
    report = build_report(args.root, args.resamples)
    write_json(output, report)
    print(json.dumps({"output": str(output), "completed": report["completed_development_count"],
                      "planned": report["planned_candidate_count"], "confirmation_accessed": False}))


if __name__ == "__main__":
    main()
