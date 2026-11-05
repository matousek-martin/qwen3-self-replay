"""Read-only study completion audit; incomplete evidence never becomes success.

By default only development evidence is inspected. Reserved confirmation outputs
are opened only when --confirmation-run is explicitly supplied, after validating
the candidate lock and completion markers. This tool never trains or provisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

try:
    from experiments import confirmation, study_report
except ModuleNotFoundError:
    import confirmation
    import study_report

ROOT = Path(__file__).resolve().parents[1]
APPROACHES = ("sft-short", "sft-compatible", "dpo", "grpo", "dr-grpo", "rl-fixed",
              "rl-adaptive", "rl-difficulty", "rl-progressive", "rl-replay")
CONTROLS = ("sft-random", "sft-dpo-matched")
REFINEMENTS = ("dpo-16", "dpo-32", "dpo-48")
RL = frozenset(APPROACHES[3:])
STATES = {"proven", "incomplete", "contradicted", "missing"}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def resolved(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include its timezone")
    return result.timestamp()


def item(identifier, state, detail, *, required=True, evidence=None):
    if state not in STATES:
        raise ValueError("Unknown audit state")
    return {"id": identifier, "state": state, "required": required, "detail": detail,
            "evidence": evidence or {}}


def check_hash(path, expected):
    path = Path(path)
    if not expected:
        raise FileNotFoundError("Expected hash not recorded: " + str(path))
    if not path.is_file():
        raise FileNotFoundError("File absent: " + str(path))
    if confirmation.digest(path) != expected:
        raise ValueError("Hash mismatch: " + str(path))


def verify_manifest(directory, mapping):
    """Verify declared hashes without interpreting data contents."""
    directory = Path(directory).resolve()
    if not isinstance(mapping, dict) or not mapping:
        raise FileNotFoundError("Missing or empty hash manifest: " + str(directory))
    for relative, expected in mapping.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or (directory / relative).is_symlink():
            raise ValueError("Unsafe manifest path: " + relative)
        check_hash(path, expected)
    return len(mapping)


def training_budget(name, info):
    """Do not trust a bare complete flag without the registered work budget."""
    if name in RL:
        status = info.get("training_status")
        config = info.get("config")
        if status is None:
            return item("training:" + name, "incomplete", "No final RL training status yet")
        if status.get("status") != "completed":
            return item("training:" + name, "incomplete", "RL run is not complete", evidence=status)
        if status.get("completed_groups") != 128 or type(status.get("optimizer_updates")) is not int or status["optimizer_updates"] < 1:
            return item("training:" + name, "contradicted", "Complete RL label conflicts with 128 groups and at least one update", evidence=status)
        if not config:
            return item("training:" + name, "missing", "Completed RL run lacks its configuration")
        required = {"max_steps": 128, "group_size": 4, "max_new_tokens": 2048, "seed": 17}
        required["variant"] = "correctness" if name in ("grpo", "dr-grpo") else name.removeprefix("rl-")
        required["optimizer"] = "grpo" if name == "grpo" else "dr_grpo"
        if name == "rl-replay":
            required["replay_reward"] = "fixed"
        if name == "rl-progressive":
            required["final_cap"] = 1024
        mismatch = {k: {"expected": v, "actual": config.get(k)} for k, v in required.items() if config.get(k) != v}
        if mismatch:
            return item("training:" + name, "contradicted", "RL matched-budget settings differ", evidence=mismatch)
        return item("training:" + name, "proven", "128 completed groups with genuine optimizer updates",
                    evidence={"groups": 128, "optimizer_updates": status["optimizer_updates"], "config": config})
    summary = info.get("parent_training_result")
    if summary is None:
        return item("training:" + name, "incomplete", "No final offline training result yet")
    if not summary.get("complete"):
        return item("training:" + name, "incomplete", "Offline training stopped before completion", evidence=summary)
    if summary.get("step") != 64 or summary.get("requested_steps") != 64:
        return item("training:" + name, "contradicted", "Completed offline run does not prove 64 requested/completed updates", evidence=summary)
    config = info.get("config")
    if not config:
        return item("training:" + name, "missing", "Offline training configuration absent")
    method = "dpo" if name == "dpo" else "sft"
    if config.get("steps") != 64 or config.get("method") != method:
        return item("training:" + name, "contradicted", "Offline method or update budget differs", evidence=config)
    selected = info.get("selected_checkpoint_state")
    if not selected:
        return item("training:" + name, "missing", "Final selected checkpoint state absent")
    if selected.get("step") != 64:
        return item("training:" + name, "contradicted", "Primary/control registry does not select its final 64-update checkpoint", evidence=selected)
    return item("training:" + name, "proven", "64 completed offline updates and final checkpoint state verified",
                evidence={"steps": 64, "method": method, "training_tokens": summary.get("training_tokens")})


def checkpoint_item(root, name, registration, info):
    if not registration:
        return item("checkpoint:" + name, "incomplete", "Checkpoint not registered yet")
    if not isinstance(registration, dict) or not isinstance(registration.get("source"), str):
        return item("checkpoint:" + name, "contradicted", "Checkpoint registration has no valid source path")
    source = resolved(root, registration["source"])
    staged = Path(root) / "lab/experiments/data/study/adapters" / name
    try:
        for folder in (source, staged):
            check_hash(folder / "adapter_model.safetensors", registration.get("checkpoint_sha256"))
            check_hash(folder / "adapter_config.json", registration.get("config_sha256"))
        config = read_json(source / "adapter_config.json")
        if config.get("base_model_name_or_path") != "Qwen/Qwen3-0.6B" or config.get("revision") != "c1899de289a04d12100db370d81485cdf75e47ca":
            raise ValueError("Checkpoint does not identify the pinned starting model")
        if name in RL:
            state = read_json(source / "training_state.json")
            if state is None:
                raise FileNotFoundError("RL checkpoint state absent")
            if state.get("completed_groups") != 128:
                raise ValueError("Final RL checkpoint is not at group 128")
        else:
            state = read_json(source / "state.json")
            expected = int(name.rsplit("-", 1)[1]) if name in REFINEMENTS else 64
            if state is None:
                raise FileNotFoundError("Offline checkpoint state absent")
            if state.get("step") != expected:
                raise ValueError("Selected checkpoint update count differs from candidate name")
        return item("checkpoint:" + name, "proven", "Archived and staged adapter/config hashes and checkpoint state agree",
                    evidence={"source": str(source), "staged": str(staged), "checkpoint_sha256": registration["checkpoint_sha256"]})
    except FileNotFoundError as exc:
        return item("checkpoint:" + name, "missing", str(exc))
    except (ValueError, KeyError, TypeError) as exc:
        return item("checkpoint:" + name, "contradicted", str(exc))


def development_item(name, row):
    if not row:
        return item("development:" + name, "missing", "Candidate absent from freshly rebuilt study report")
    analysis = row.get("analysis")
    if row.get("evaluation_state") != "complete" or not analysis:
        state = "contradicted" if row.get("evaluation_state") == "invalid" else "incomplete"
        return item("development:" + name, state, "Development evaluation not fully verified",
                    evidence={"state": row.get("evaluation_state"), "attempts": row.get("evaluation_attempts", [])})
    if analysis.get("scope") != "tuning" or analysis.get("samples") != 2 or analysis.get("question_count") != 512 or set(analysis.get("domains", {})) != set(confirmation.DOMAINS):
        return item("development:" + name, "contradicted", "Verified analysis does not match the registered development cohort")
    if not analysis.get("input_sha256") or not row.get("evaluation_provenance"):
        return item("development:" + name, "missing", "Development input/checkpoint provenance absent")
    return item("development:" + name, "proven", "Fresh study_report verified paired 512-question, two-sample development output",
                evidence={"provenance": row["evaluation_provenance"], "input_sha256": analysis["input_sha256"]})


def replay_control_item(rows):
    fixed = rows.get("rl-fixed", {}).get("training", {}).get("config")
    replay = rows.get("rl-replay", {}).get("training", {}).get("config")
    if not fixed or not replay:
        return item("replay_matched_control", "incomplete", "Both fixed and replay configurations are required")
    fields = ("data_sha256", "references_sha256", "length_coefficient", "learning_rate", "rank",
              "group_size", "seed", "max_steps", "max_new_tokens", "optimizer", "scale_rewards_resolved")
    if not fixed.get("references_sha256") or not replay.get("references_sha256"):
        return item("replay_matched_control", "missing", "Shared measured length-reference hashes absent")
    mismatches = {field: {"fixed": fixed.get(field), "replay": replay.get(field)} for field in fields
                  if field not in fixed or field not in replay or fixed[field] != replay[field]}
    if replay.get("replay_reward") != "fixed":
        mismatches["replay_reward"] = replay.get("replay_reward")
    return item("replay_matched_control", "contradicted" if mismatches else "proven",
                "Replay adds its auxiliary loss to the otherwise identical fixed-reward setup",
                evidence={"mismatches": mismatches, "matched_fields": list(fields)})


def rental_items(ledger, now, ceiling=200):
    if ledger is None:
        return [item("rentals_released", "missing", "Cloud ledger absent"),
                item("budget", "missing", "Budget cannot be proved without ledger")]
    jobs = ledger.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        return [item("rentals_released", "missing", "Ledger has no campaign jobs"),
                item("budget", "missing", "No recorded campaign costs")]
    unreleased = {name: {k: j.get(k) for k in ("state", "pod_id", "pod_name", "delete_requested_at", "error")}
                  for name, j in jobs.items() if not j.get("released_at")}
    unproved_release = [name for name, j in jobs.items() if j.get("released_at") and
                       ((not j.get("pod_id") and not j.get("safely_no_pod")) or
                        (j.get("duplicate_pod_ids") and not j.get("duplicate_cleanup_confirmed")))]
    cleanup = item("rentals_released", "incomplete" if unreleased or unproved_release else "proven",
                   "Unreleased/uncertain allocations remain" if unreleased or unproved_release else "Every campaign allocation has a recorded release or reconciled absence",
                   evidence={"unreleased": unreleased, "unproved_release": unproved_release,
                             "scope": "All jobs in the project ledger, including ambiguous creates; no fresh provider query"})
    try:
        for name, j in jobs.items():
            required = ("charged_estimate_usd",) if j.get("released_at") else ("budget_rate", "reserved_usd", "created_at")
            for field in required:
                if field not in j:
                    raise FileNotFoundError(f"{name}: {field} absent")
                value = j[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name}: invalid {field}")
        cost = study_report.cost_report(ledger, now)
        liability = cost["total_conservative_liability_usd"]
        if not cost["cost_accounting_complete"]:
            raise FileNotFoundError("Cost accounting is incomplete")
        state = "proven" if liability <= ceiling else "contradicted"
        budget = item("budget", state, "Conservative ledger liability compared with the $200 ceiling",
                      evidence={"ceiling_usd": ceiling, "liability_usd": liability,
                                "invoice_verified": False, "cost": cost})
    except FileNotFoundError as exc:
        budget = item("budget", "missing", str(exc))
    except (ValueError, TypeError) as exc:
        budget = item("budget", "contradicted", str(exc))
    return [cleanup, budget]


def runtime_path(root, job, value):
    path = Path(value)
    remote = Path("/workspace/project")
    if path.is_relative_to(remote / "data"):
        return Path(root) / "lab/experiments/data" / path.relative_to(remote / "data")
    if path.is_relative_to(remote / "output"):
        return Path(root) / "lab/experiments/cloud" / job / "output" / path.relative_to(remote / "output")
    return resolved(root, path)


def provenance_items(root, protocol, rows, ledger, extra_jobs=()):
    root = Path(root)
    result = []
    data = root / "lab/experiments/data"
    for relative in ("manifest.json", "shared/selected/manifest.json", "shared/compatible/manifest.json"):
        path = data / relative
        try:
            manifest = read_json(path)
            if manifest is None:
                raise FileNotFoundError(str(path))
            count = verify_manifest(path.parent, manifest.get("files"))
            result.append(item("data_manifest:" + relative, "proven", "Declared data artifacts verified without reading example content",
                               evidence={"manifest_sha256": confirmation.digest(path), "files_verified": count}))
        except FileNotFoundError as exc:
            result.append(item("data_manifest:" + relative, "missing", str(exc)))
        except (ValueError, TypeError) as exc:
            result.append(item("data_manifest:" + relative, "contradicted", str(exc)))
    jobs_needed = set(extra_jobs)
    for name, row in rows.items():
        info = row.get("training", {})
        job = info.get("job")
        if job:
            jobs_needed.add(job)
        config = info.get("config")
        if not config or not job:
            result.append(item("training_data:" + name, "incomplete", "Training data provenance not available yet"))
            continue
        try:
            check_hash(runtime_path(root, job, config["data"]), config.get("data_sha256"))
            if config.get("references"):
                check_hash(runtime_path(root, job, config["references"]), config.get("references_sha256"))
            result.append(item("training_data:" + name, "proven", "Training dataset and any length-reference hashes agree with run configuration"))
        except FileNotFoundError as exc:
            result.append(item("training_data:" + name, "missing", str(exc)))
        except (ValueError, KeyError, TypeError) as exc:
            result.append(item("training_data:" + name, "contradicted", str(exc)))
        evaluation_job = row.get("evaluation_provenance", {}).get("job")
        if evaluation_job:
            jobs_needed.add(evaluation_job)
    cloud = root / "lab/experiments/cloud"
    reference = cloud / "bootstrap-20261023T125530Z/source/experiments/offline_training"
    for job in sorted(jobs_needed):
        directory = cloud / job / "source"
        path = directory / "manifest.json"
        try:
            manifest = read_json(path)
            if manifest is None:
                # Historical deployment copies were not archived. The known
                # reference is useful reproducibility context, not proof of upload.
                known_legacy = job in ("sft-20261023T125530Z", "dpo-20261023T125530Z")
                if known_legacy and (reference / "train.py").is_file() and (reference / "common.py").is_file():
                    result.append(item("source_accounting:" + job, "proven", "Historical gap explicitly accounted for; exact deployment code is not provable",
                        evidence={"exact_deployment_source_proven": False, "historical_reference": str(reference),
                                  "reference_hashes": {p.name: confirmation.digest(p) for p in reference.glob("*.py")},
                                  "limitation": "Reference snapshot plus archived configuration/data/model provenance; no claim that this proves exact uploaded bytes"}))
                    result.append(item("exact_source:" + job, "missing", "Historical upload hash was not recorded", required=False))
                    continue
                raise FileNotFoundError("Exact source manifest absent for " + job)
            mandatory = {"ThinkingCap-Test/evaluate.py", "ThinkingCap-Test/utils/eval/recipe.py"}
            if job.startswith("study-eval-") or job in extra_jobs:
                mandatory.add("experiments/study_eval.py")
            elif job.startswith("study-"):
                mandatory.update(("experiments/rl_training/train.py", "experiments/rl_training/objectives.py"))
            if not mandatory.issubset(manifest):
                raise FileNotFoundError("Source manifest omits required execution files: " + str(sorted(mandatory - set(manifest))))
            count = verify_manifest(directory, manifest)
            if job.startswith("study-eval-") or job in extra_jobs:
                execution = read_json(cloud / job / "output/study/study-provenance.json")
                if execution is None:
                    raise FileNotFoundError("Evaluator execution provenance absent for " + job)
                check_hash(directory / "experiments/study_eval.py", execution.get("runner_sha256"))
                verify_manifest(directory / "ThinkingCap-Test", execution.get("evaluator_files_sha256"))
            result.append(item("source_accounting:" + job, "proven", "Archived deployment source manifest verified",
                               evidence={"files_verified": count, "manifest_sha256": confirmation.digest(path), "exact_deployment_source_proven": True}))
        except FileNotFoundError as exc:
            result.append(item("source_accounting:" + job, "missing", str(exc)))
        except (ValueError, TypeError) as exc:
            result.append(item("source_accounting:" + job, "contradicted", str(exc)))
    return result


def locate_confirmation(path):
    path = Path(path)
    for candidate in (path, path / "study", path / "output/study"):
        if (candidate / "study-provenance.json").is_file():
            return candidate
    return path


def confirmation_items(root, run, split_path, protocol, resamples=2000):
    """Only call with an explicit confirmation path; gate outcome reads on lock."""
    if run is None:
        return [item("confirmation_lock", "missing", "No confirmation run supplied; reserved outputs not accessed"),
                item("confirmation_evaluation", "incomplete", "Locked confirmation has not been audited")], None
    output = locate_confirmation(resolved(root, run))
    try:
        provenance = read_json(output / "study-provenance.json")
    except (OSError, ValueError) as exc:
        return [item("confirmation_lock", "contradicted", "Unreadable confirmation provenance: " + str(exc)),
                item("confirmation_evaluation", "incomplete", "Outcome reads blocked by unreadable provenance")], None
    if provenance is None:
        return [item("confirmation_lock", "missing", "Confirmation provenance absent"),
                item("confirmation_evaluation", "incomplete", "No completed confirmation evidence yet")], None
    try:
        if provenance.get("subset") != "confirmation" or provenance.get("samples") != 8 or provenance.get("cohort_total") != 2247:
            raise ValueError("Confirmation provenance must identify 2247 questions and eight samples")
        if provenance.get("split_manifest_sha256") != confirmation.digest(split_path):
            raise ValueError("Confirmation split hash differs")
        lock = provenance.get("candidate_lock")
        if not isinstance(lock, dict):
            raise FileNotFoundError("Embedded candidate lock absent")
        name = lock.get("candidate")
        if not name or lock.get("primary_candidate") != name:
            raise ValueError("Confirmation must name one locked primary candidate")
        extra = study_report.adapter_refinement.registered_specs(protocol)
        if name not in set(APPROACHES + CONTROLS + REFINEMENTS + tuple(extra)):
            raise ValueError("Confirmation candidate is outside the audited registered study candidates")
        if name in extra:
            registry = read_json(Path(root) / 'lab/experiments/data/study/adapters.json')
            if not registry or name not in registry or 'dpo-32' not in registry:
                raise FileNotFoundError('Registered refinement lineage is absent from the checkpoint registry')
            study_report.adapter_refinement.verify_registration(
                root, name, registry[name], registry['dpo-32'], extra[name])
        if provenance.get("config", {}).get("candidates") != ["base", name]:
            raise ValueError("Confirmation must contain exactly a matched base and the primary candidate")
        locked = timestamp(lock["locked_at"])
        if timestamp(protocol["registered_at"]) > locked:
            raise ValueError("Candidate was locked before the protocol registration")
        statuses = {candidate: read_json(output / "status" / (candidate + ".json")) for candidate in ("base", name)}
        if not all(statuses.values()) or not all(s.get("started_at") for s in statuses.values()):
            raise FileNotFoundError("Evaluator start timestamps missing; lock ordering not yet provable")
        if any(locked >= timestamp(status["started_at"]) for status in statuses.values()):
            raise ValueError("Candidate lock does not precede both confirmation evaluator starts")
        data = Path(root) / "lab/experiments/data/study"
        relative = provenance.get("config", {}).get("candidate_lock")
        if not relative:
            raise FileNotFoundError("Candidate lock filename missing from configuration")
        lock_path = (data / relative).resolve()
        if not lock_path.is_relative_to(data.resolve()):
            raise ValueError("Lock path escapes staged study directory")
        actual_lock = read_json(lock_path)
        if actual_lock is None:
            raise FileNotFoundError("Persistent candidate lock file missing: " + str(lock_path))
        if actual_lock != lock:
            raise ValueError("Persistent and evaluator-embedded candidate locks differ")
        adapter = data / "adapters" / name
        hashes = confirmation.adapter_hashes(adapter)
        if lock.get("adapter_files_sha256") != hashes or provenance.get("adapter_files_sha256", {}).get(name) != hashes:
            raise ValueError("Candidate weight/config hashes differ from lock or evaluation")
        if lock.get("split_manifest_sha256") != confirmation.digest(split_path):
            raise ValueError("Lock split hash differs")
        lock_item = item("confirmation_lock", "proven", "Persisted checkpoint hashes locked before both evaluator start timestamps",
                         evidence={"candidate": name, "locked_at": lock["locked_at"], "lock_sha256": confirmation.digest(lock_path),
                                   "evaluator_started_at": {k: v["started_at"] for k, v in statuses.items()},
                                   "limitation": "Recorded timestamp ordering cannot prove absence of prior human inspection or timestamp tampering"})
    except FileNotFoundError as exc:
        return [item("confirmation_lock", "missing", str(exc)), item("confirmation_evaluation", "incomplete", "Outcome reads blocked until lock evidence is complete")], None
    except (ValueError, KeyError, TypeError) as exc:
        return [item("confirmation_lock", "contradicted", str(exc)), item("confirmation_evaluation", "incomplete", "Outcome reads blocked by invalid lock/provenance")], None
    if any(status.get("state") != "complete" for status in statuses.values()):
        return [lock_item, item("confirmation_evaluation", "incomplete", "Both locked confirmation runs must finish before analysis",
                               evidence={k: v.get("state") for k, v in statuses.items()})], None
    try:
        expected_responses = 2247 * 8
        paths = {}
        for candidate, status in statuses.items():
            if status.get("responses") != expected_responses:
                raise ValueError("Confirmation status response count differs from 2247 × 8")
            for filename, field in (("results.json", "results_sha256"), ("records.jsonl", "records_sha256")):
                path = output / "runs" / candidate / filename
                check_hash(path, status.get(field))
                paths[path] = status[field]
            # Inspect structural grading status only, after the lock gate. The
            # numerical outcomes themselves are analyzed by confirmation.analyze.
            with (output / "runs" / candidate / "records.jsonl").open() as stream:
                for line in stream:
                    if line.strip() and json.loads(line).get("grading_error"):
                        raise ValueError("Confirmation contains an unresolved grading error")
        analysis = confirmation.analyze(output / "runs/base", output / "runs" / name, split_path,
                                        "confirmation", resamples=resamples, lock_path=lock_path, adapter=adapter)
        for path, expected in paths.items():
            check_hash(path, expected)
        if analysis.get("scope") != "confirmation" or analysis.get("samples") != 8 or analysis.get("question_count") != 2247 or set(analysis.get("domains", {})) != set(confirmation.DOMAINS):
            raise ValueError("Paired analysis does not cover the complete confirmation cohort")
        if analysis.get("inference_role") != "confirmatory_primary":
            raise ValueError("Analysis is not for the preselected primary candidate")
        for scope, value in {"macro": analysis["macro"], **analysis["domains"]}.items():
            for key, margin in (("strict_preservation", 0), ("one_pp_noninferiority", 1)):
                expected = confirmation.preservation(value["accuracy_delta"], value["intervals"]["accuracy_delta"], margin)
                if value.get(key) != expected:
                    raise ValueError("Incorrect preservation classification for " + scope)
        verdict = item("confirmation_evaluation", "proven", "Paired eight-sample confirmation verified; study completion does not require a positive scientific outcome",
                       evidence={"candidate": name, "questions": 2247, "responses_per_side": expected_responses,
                                 "domains": list(confirmation.DOMAINS), "strict": analysis["macro"]["strict_preservation"],
                                 "one_pp": analysis["macro"]["one_pp_noninferiority"], "input_sha256": analysis["input_sha256"]})
        return [lock_item, verdict], analysis
    except FileNotFoundError as exc:
        return [lock_item, item("confirmation_evaluation", "missing", str(exc))], None
    except (ValueError, KeyError, TypeError) as exc:
        return [lock_item, item("confirmation_evaluation", "contradicted", str(exc))], None


def build_audit(root=ROOT, confirmation_run=None, resamples=2000, now=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    study = root / "lab/experiments/study"
    checks, analysis = [], None
    def object_input(path, identifier, default=None):
        try:
            value = read_json(path, default)
            if value is not None and not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (OSError, ValueError) as exc:
            checks.append(item(identifier, "contradicted", "Unreadable metadata: " + str(exc)))
            return default

    protocol = object_input(study / "protocol.json", "protocol")
    extra = {}
    if protocol:
        try:
            extra = study_report.adapter_refinement.registered_specs(protocol)
        except (ValueError, KeyError, TypeError) as exc:
            checks.append(item('additional_refinements', 'contradicted', str(exc)))
    if protocol is None:
        if not any(check["id"] == "protocol" for check in checks):
            checks.append(item("protocol", "missing", "Registered protocol absent"))
        report = None
    else:
        expected = {"required_approaches": set(APPROACHES), "offline_controls": set(CONTROLS),
                    "initial_refinement_candidates": set(REFINEMENTS)}
        valid = all(set(protocol.get(k, [])) == v and len(protocol[k]) == len(v) for k, v in expected.items())
        valid = valid and protocol.get("budget_total_usd") == 200
        valid = valid and all(protocol.get("rl_matched_budget", {}).get(k) == v for k, v in
                              {"groups": 128, "group_size": 4, "initial_generation_cap": 2048, "seed": 17}.items())
        valid = valid and all(protocol.get("confirmation", {}).get(k) == v for k, v in
                              {"questions": 2247, "samples_per_question": 8, "accuracy_margin_pp": 0}.items())
        checks.append(item("protocol", "proven" if valid else "contradicted", "Required ten approaches, two controls and three refinements checked",
                           evidence={k: protocol.get(k) for k in expected}))
        try:
            report = study_report.build_report(root, resamples=resamples, now=now)
            checks.append(item("development_report", "proven", "Development evidence rebuilt through study_report; stored green flags were not trusted",
                               evidence={"generated_at": report["generated_at"], "input_sha256": report["input_sha256"]}))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report = None
            checks.append(item("development_report", "incomplete", "Could not verify current development evidence: " + str(exc)))
    rows = {r["name"]: r for r in (report or {}).get("candidates", [])}
    registrations = object_input(root / "lab/experiments/data/study/adapters.json", "adapter_registry", {})
    for name in APPROACHES + CONTROLS:
        info = rows.get(name, {}).get("training", {})
        checks.append(training_budget(name, info))
        checks.append(checkpoint_item(root, name, registrations.get(name), info))
    for name in REFINEMENTS:
        info = rows.get(name, {}).get("training", {})
        checkpoint = checkpoint_item(root, name, registrations.get(name), info)
        checks.append(checkpoint)
        parent_registration, selected_registration = registrations.get("dpo", {}), registrations.get(name, {})
        parent = parent_registration.get("source") if isinstance(parent_registration, dict) else None
        source = selected_registration.get("source") if isinstance(selected_registration, dict) else None
        if source and parent and resolved(root, source).parent != resolved(root, parent).parent:
            checks.append(item("refinement_parent:" + name, "contradicted", "Earlier checkpoint does not belong to the registered DPO training run"))
        else:
            checks.append(item("refinement_parent:" + name, "proven" if source and parent else "incomplete", "Refinement lineage to the DPO run checked"))
    for name, spec in extra.items():
        try:
            proof = study_report.adapter_refinement.verify_registration(
                root, name, registrations[name], registrations['dpo-32'], spec)
            if timestamp(spec['registered_at']) < timestamp(protocol['registered_at']):
                raise ValueError('Additional refinement registered before the original study')
            checks.append(item('checkpoint:' + name, 'proven',
                               'Only the registered adapter strength differs from the archived DPO-32 parent',
                               evidence=proof))
        except (FileNotFoundError, KeyError) as exc:
            checks.append(item('checkpoint:' + name, 'missing', str(exc)))
        except (OSError, ValueError, TypeError) as exc:
            checks.append(item('checkpoint:' + name, 'contradicted', str(exc)))
    for name in APPROACHES + CONTROLS + REFINEMENTS + tuple(extra):
        checks.append(development_item(name, rows.get(name)))
    checks.append(replay_control_item(rows))
    ledger = object_input(root / "lab/experiments/cloud/ledger.json", "ledger_integrity")
    checks.extend(rental_items(ledger, now, ceiling=200))
    if protocol:
        extra_jobs = []
        if confirmation_run is not None:
            candidate_output = locate_confirmation(resolved(root, confirmation_run))
            confirmation_job = study_report.job_from_path(candidate_output, root / "lab/experiments/cloud")
            if confirmation_job:
                extra_jobs.append(confirmation_job)
            else:
                checks.append(item("confirmation_source", "missing", "Confirmation output lies outside the job archive; exact deployed source archive cannot be resolved"))
        checks.extend(provenance_items(root, protocol, rows, ledger or {}, extra_jobs=extra_jobs))
        try:
            split_path = resolved(root, protocol["splits_manifest"])
            split = confirmation.load_split(split_path)  # Identity/metadata only; never opens question content.
            if split["subsets"]["confirmation"]["total"] != 2247 or split["policy"]["confirmation_samples"] != 8:
                raise ValueError("Frozen confirmation split/sample count differs from protocol")
            checks.append(item("split", "proven", "Frozen split metadata is disjoint and exhaustive", evidence={"sha256": confirmation.digest(split_path)}))
            confirmation_checks, analysis = confirmation_items(root, confirmation_run, split_path, protocol, resamples)
            checks.extend(confirmation_checks)
        except FileNotFoundError as exc:
            checks.append(item("split", "missing", str(exc)))
            checks.append(item("confirmation_evaluation", "incomplete", "Confirmation cannot be checked without the split"))
        except (ValueError, KeyError, TypeError) as exc:
            checks.append(item("split", "contradicted", str(exc)))
            checks.append(item("confirmation_evaluation", "incomplete", "Confirmation blocked by invalid split metadata"))
    else:
        checks.append(item("confirmation_evaluation", "incomplete", "Confirmation blocked until protocol is available"))
    required = [c for c in checks if c["required"]]
    complete = bool(required) and all(c["state"] == "proven" for c in required)
    historical = [{"job": name, "state": j.get("state"), "error": j.get("error"),
                   "release_recorded": bool(j.get("released_at")), "backup_verified": j.get("backup_verified", False)}
                  for name, j in (ledger or {}).get("jobs", {}).items()
                  if j.get("released_at") and j.get("state") != "complete"]
    return {"schema_version": 1, "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "study_complete": complete, "state": "proven" if complete else "contradicted" if any(c["state"] == "contradicted" for c in required) else "incomplete",
            "required_counts": dict(Counter(c["state"] for c in required)), "items": checks,
            "confirmation_analysis": analysis,
            "historical_failed_or_abandoned_attempts": historical,
            "development_attempts": {name: row.get("evaluation_attempts", []) for name, row in rows.items()},
            "bootstrap_resamples": resamples,
            "scientific_success_required_for_completion": False,
            "scope": "Read-only evidence audit; does not change goal status, run training, or query providers",
            "limitations": ["Historical SFT/DPO exact upload hashes were not recorded; a reference snapshot and model/data/configuration evidence are disclosed separately.",
                            "Ledger cost is a conservative estimate, not a provider invoice; recorded releases are not fresh provider queries.",
                            "Timestamp ordering is evidence of lock order, not proof of prior noninspection.",
                            "A completed study can find regression or inconclusive accuracy; neither is relabeled preservation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--confirmation-run", type=Path, help="Explicit study-output directory (or its parent job/output); omitted means no reserved-output access")
    parser.add_argument("--out", type=Path, help="Optional audit report destination; inputs are never modified")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples < 200:
        parser.error("Use at least 200 paired bootstrap resamples")
    report = build_audit(args.root, args.confirmation_run, args.resamples)
    if args.out:
        confirmation.write_json(args.out, report)
        print(json.dumps({"output": str(args.out), "state": report["state"], "study_complete": report["study_complete"],
                          "required_counts": report["required_counts"]}, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
