"""Run the unchanged contest baseline, back it up, then release its GPU.

Execute with .venv-lab/bin/python ops/run_baseline.py. No time limit is imposed
on evaluation. A failed job is stopped; successful, verified backups allow the
temporary pod to be terminated. W&B records and source data also stay local.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from runpod import ROOT, STATE, api, quote, ssh_args, summary

LAB = ROOT / "lab"
REPO = ROOT / "ThinkingCap-Test"


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def ssh(pod, text, **kwargs):
    return command(ssh_args(pod) + [text], **kwargs)


def sync(pod, source, target, upload=False, exclusions=()):
    conn = ssh_args(pod)
    host, transport = conn[-1], shlex.join(conn[:-1])
    src = str(source) if upload else host + ":" + str(source)
    dst = host + ":" + str(target) if upload else str(target)
    args = ["rsync", "-az", "--timeout=120", "-e", transport]
    for pattern in exclusions:
        args += ["--exclude", pattern]
    command(args + [src, dst])


def backup(pod, archive):
    for name in ("artifacts", "logs"):
        sync(pod, "/workspace/" + name + "/", archive / name)
    sync(pod, "/workspace/ThinkingCap-Test/runs/", archive / "runs")


def verify_backup(archive):
    for line in (archive / "artifacts/baseline.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = archive / relative
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError("Backup checksum mismatch: " + relative)
    results = json.loads((archive / "runs/base/results.json").read_text())
    if results["failed_benchmarks"] or results["smoke"] or results["samples"] != 8:
        raise RuntimeError("Incomplete baseline")
    if sum(r["n"] for r in results["results"].values()) != 22072:
        raise RuntimeError("Unexpected response count")
    with (archive / "runs/base/records.jsonl").open() as f:
        if sum(1 for _ in f) != 22072:
            raise RuntimeError("Incomplete records backup")
    return results


def log_results(run, results, archive, results_dir=None):
    import wandb
    recovered = results_dir is not None
    results_dir = results_dir or archive / "runs/base"
    metrics = {}
    rows = []
    for name, values in results["results"].items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                metrics[f"eval/{name}/{key}"] = value
        for metric in ("accuracy", "mean_output_tokens"):
            lo, hi = values["confidence"][metric]
            metrics[f"eval/{name}/{metric}_ci_low"] = lo
            metrics[f"eval/{name}/{metric}_ci_high"] = hi
        rows.append([name, values["n"], values["accuracy"], values["mean_output_tokens"],
                     values["mean_reasoning_tokens"], values["truncation_rate"]])
    groups = {}
    for name, values in results["results"].items():
        groups.setdefault("bfcl" if name.startswith("bfcl_") else name, []).append(values)
    metrics["eval/macro_accuracy"] = sum(
        sum(v["accuracy"] * v["n"] for v in group) / sum(v["n"] for v in group)
        for group in groups.values()) / len(groups)
    run.log(metrics)
    run.summary.update(metrics)
    run.log({"benchmark_results": wandb.Table(columns=["benchmark", "responses", "accuracy",
             "mean_output_tokens", "mean_reasoning_tokens", "truncation_rate"], data=rows)})
    artifact = wandb.Artifact("baseline-evaluation", type="evaluation")
    for path, name in [(results_dir / "results.json", "results.json"),
                       (REPO / "questions/manifest.json", "questions-manifest.json"),
                       (LAB / "gpu-requirements.lock", "gpu-requirements.lock"),
                       (archive / "provenance.json", "provenance.json"),
                       (results_dir / "SHA256SUMS" if recovered else archive / "artifacts/baseline.sha256",
                        "baseline.sha256")]:
        artifact.add_file(str(path), name=name)
    if recovered:
        for name in ("recovery.json", "grader.patch", "packages.txt"):
            artifact.add_file(str(results_dir / name), name=name)
    return run.log_artifact(artifact)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Controller interrupted; stopping the GPU")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--retry-setup", action="store_true",
                        help="Reuse the stopped pod after a setup failure, before any evaluation")
    args = parser.parse_args()
    # Load only W&B settings here. Runpod credentials stay in runpod.api().
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if "=" in line and not line.startswith("#"):
            name, value = line.split("=", 1)
            if name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_BASE_URL"):
                os.environ[name] = shlex.split(value, comments=True)[0]
    import wandb
    previous = json.loads(STATE.read_text()) if STATE.exists() else None
    if previous and not args.retry_setup:
        raise RuntimeError("A deployment record already exists; inspect it before starting another GPU")
    if args.retry_setup and (not previous or previous["status"] != "EXITED"):
        raise RuntimeError("Setup retry requires an existing stopped pod")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = LAB / ("baseline-" + stamp)
    for name in ("artifacts", "logs", "runs"):
        (archive / name).mkdir(parents=True, exist_ok=True)
    commit = command(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    manifest = json.loads((REPO / "questions/manifest.json").read_text())
    config = {"model": "Qwen/Qwen3-0.6B", "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
              "repo_commit": commit, "samples": 8, "seed": 9001,
              "temperature": 0.6, "top_p": 0.95, "top_k": 20,
              "cohort_sha256": manifest["cohort_sha256"], "cohort_total": 2759,
              "gpu": "RTX 4090", "gpu_count": 1, "time_limit": None}
    provenance = dict(config, source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in REPO.rglob("*.py") if not any(x in p.parts for x in ("questions", ".venv", "__pycache__"))})
    write_json(archive / "provenance.json", provenance)
    run = wandb.init(project="thinkingcap", entity=os.environ.get("WANDB_ENTITY"),
                     name="qwen3-0.6b-baseline-" + stamp, job_type="baseline", config=config,
                     dir=str(archive), mode="online" if os.environ.get("WANDB_API_KEY") else "offline",
                     settings=wandb.Settings(disable_git=True, disable_code=True, save_code=False,
                                             x_disable_stats=True))
    write_json(LAB / "active-run.json", {"archive": str(archive), "wandb_url": run.url, "wandb_id": run.id})
    print("W&B:", run.url, flush=True)
    pod = None
    completed = False
    started = time.time()
    try:
        quotes = {cloud: quote(cloud) for cloud in ("COMMUNITY", "SECURE")}
        write_json(archive / "quotes.json", quotes)
        cloud = previous["cloud"] if args.retry_setup else next(
            (c for c in ("COMMUNITY", "SECURE") if quotes[c]["availability"] != "NONE"), None)
        if cloud is None:
            raise RuntimeError("No RTX 4090 available; nothing rented")
        rate = quotes[cloud]["price"][cloud.lower()]
        if rate > 1.0:
            raise RuntimeError(f"Current price ${rate}/h exceeds the reviewed $0.89/h offer")
        request = {"name": "thinkingcap-baseline-" + stamp,
                   "templateId": "runpod-torch-v280", "cloud": cloud,
                   "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1, "minCudaVersion": "12.8"},
                   "disk": 30, "mounts": {"persistent": {"size": 50, "path": "/workspace"}},
                   "ports": ["22/tcp"], "startSsh": True, "startJupyter": False}
        write_json(archive / "deployment-request.json", request)
        # Never retry an ambiguous create. Reconcile the named pod manually instead.
        if args.retry_setup:
            pod = api("pods/" + previous["id"])
            if "start" not in pod["actions"]:
                raise RuntimeError("Stopped pod cannot be resumed")
            pod = api("pods/" + previous["id"] + "/action", "POST", {"action": "start"})
        else:
            pod = api("pods", "POST", request)
        write_json(STATE, summary(pod))
        write_json(archive / "pod.json", summary(pod))
        run.config.update({"cloud": cloud, "gpu_price_usd_per_hour": rate, "pod_id": pod["id"]})
        print("Created pod", pod["id"], "rate", pod.get("cost"), flush=True)
        while True:
            pod = api("pods/" + pod["id"])
            if pod["status"] in ("ERROR", "EXITED", "TERMINATED"):
                raise RuntimeError("Pod did not start: " + pod["status"])
            if (pod.get("ssh") or {}).get("direct"):
                probe = subprocess.run(ssh_args(pod) + ["true"], capture_output=True, timeout=25)
                if probe.returncode == 0:
                    break
            print("Waiting for SSH:", pod["status"], flush=True)
            time.sleep(20)
        write_json(STATE, summary(pod))
        ssh(pod, "mkdir -p /workspace/ops /workspace/logs /workspace/artifacts /workspace/ThinkingCap-Test/runs; command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)")
        if args.retry_setup:
            ssh(pod, "test ! -e /workspace/ThinkingCap-Test/runs/check/records.jsonl && test ! -e /workspace/ThinkingCap-Test/runs/base/records.jsonl && rm -f /workspace/artifacts/exit.json /workspace/artifacts/phase")
        sync(pod, str(REPO) + "/", "/workspace/ThinkingCap-Test/", upload=True,
             exclusions=(".git", ".venv*", "__pycache__", "runs", ".env*"))
        sync(pod, ROOT / "ops/baseline.sh", "/workspace/ops/baseline.sh", upload=True)
        sync(pod, Path("/root/.local/bin/uv").resolve(), "/workspace/ops/uv", upload=True)
        sync(pod, LAB / "gpu-requirements.lock", "/workspace/ops/gpu-requirements.lock", upload=True)
        sync(pod, "/root/nltk_data/", "/workspace/nltk_data/", upload=True)
        ssh(pod, "nohup bash /workspace/ops/baseline.sh > /workspace/logs/runner.log 2>&1 < /dev/null &")
        while True:
            backup(pod, archive)
            phase_path = archive / "artifacts/phase"
            phase = phase_path.read_text().strip() if phase_path.exists() else "setup"
            count = 0
            records = archive / "runs/base/records.jsonl"
            if records.exists():
                with records.open() as stream:
                    count = sum(1 for line in stream if line.endswith("\n"))
            elapsed = time.time() - started
            print(f"{phase}: {count}/22072 baseline responses; {elapsed/60:.1f} minutes elapsed", flush=True)
            run.log({"progress/responses": count, "runtime/elapsed_seconds": elapsed,
                     "runtime/estimated_gpu_cost_usd": elapsed / 3600 * rate})
            write_json(archive / "progress.json", {"phase": phase, "responses": count, "elapsed_seconds": elapsed})
            exit_path = archive / "artifacts/exit.json"
            if exit_path.exists():
                result = json.loads(exit_path.read_text())
                if result["exit_code"] or result["phase"] != "complete":
                    raise RuntimeError("Remote job failed: " + json.dumps(result))
                results = verify_backup(archive)
                log_results(run, results, archive)
                completed = True
                break
            time.sleep(60)
        run.summary["outcome"] = "completed"
        print("Full baseline backed up and checksums verified:", archive, flush=True)
    except BaseException as error:
        run.summary["outcome"] = "failed"
        print("FAILED:", str(error), flush=True)
        raise
    finally:
        if pod is not None:
            # Free the compute even if W&B or the local controller fails.
            current = api("pods/" + pod["id"])
            if completed:
                api("pods/" + pod["id"], "DELETE")
                write_json(archive / "cleanup.json", {"pod_id": pod["id"], "terminated": True})
                state = summary(current)
                state["status"] = "TERMINATED"
                write_json(STATE, state)
                print("Temporary pod terminated; GPU and its disks released.", flush=True)
            elif "stop" in current.get("actions", []):
                stopped = api("pods/" + pod["id"] + "/action", "POST", {"action": "stop"})
                write_json(STATE, summary(stopped))
                print("Failed job stopped; persistent disk retained for recovery.", flush=True)
        run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
