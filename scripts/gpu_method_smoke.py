#!/usr/bin/env python3
"""Run bounded method checks on one already-leased RTX4090; never allocate compute."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from self_replay.artifacts import sha256, write_json
from self_replay.methods.runtime import environment, initialize
from self_replay.paths import repository


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(command, workspace, env, log, seconds):
    with Path(log).open("w") as stream:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=workspace,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=seconds)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if code:
            raise RuntimeError(f"Command exited {code}; see {log}")


def reload_adapter(adapter, destination):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, revision = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 3 + 4?"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
    ).cuda()
    with torch.inference_mode():
        active = model(ids).logits[:, -1].float()
        with model.disable_adapter():
            inactive = model(ids).logits[:, -1].float()
        difference = float((active - inactive).abs().max())
        generated = model.generate(
            ids, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    if not torch.isfinite(active).all() or difference <= 0 or generated.shape[1] <= ids.shape[1]:
        raise RuntimeError("Reloaded adapter has no finite measurable effect/inference")
    write_json(
        destination,
        dict(
            passed=True,
            max_logit_delta=difference,
            generated_tokens=int(generated.shape[1] - ids.shape[1]),
            adapter_sha256=sha256(Path(adapter) / "adapter_model.safetensors"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--lease", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reload", type=Path)
    a = p.parse_args()
    if a.reload:
        reload_adapter(a.reload, a.out)
        return
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if names != ["NVIDIA GeForce RTX 4090"]:
        raise RuntimeError("Exactly one RTX4090 required")
    lease = json.loads(a.lease.read_text())
    if (
        lease["gpu_count"] != 1
        or lease["gpu"] != names[0]
        or lease["hardware_scope"] != "submission-method-smoke-v1"
    ):
        raise RuntimeError("Lease hardware/scope mismatch")
    if a.out.exists():
        raise ValueError("Fresh output required")
    a.out.mkdir(parents=True)
    suite = json.loads(a.suite.read_text())
    for relative, digest in suite["files"].items():
        if sha256(a.data / relative) != digest:
            raise ValueError("Staged input hash mismatch: " + relative)
    receipt = dict(
        technical_only=True, passed=False, gpu=names[0], started_at=time.time(), cases=[]
    )
    write_json(a.out / "report.json", receipt)
    for case in suite["cases"]:
        item = dict(recipe=case["recipe"], arms=[], passed=False, started_at=time.time())
        receipt["cases"].append(item)
        write_json(a.out / "report.json", receipt)
        print("SMOKE CASE", case["recipe"], flush=True)
        try:
            workspace = initialize(a.out / "workspaces" / case["recipe"], case["recipe"])
            shutil.copy2(a.lease, workspace / "worker-lease.json")
            env = environment(workspace)
            env["SELF_REPLAY_REPO"] = str(repository())
            for arm in case["arms"]:
                remaining = min(lease["deadline"], lease["hard_deadline"]) - time.time() - 180
                if remaining < 180:
                    raise RuntimeError("Insufficient leased time for another test")
                dest = a.out / case["recipe"] / arm["name"]
                dest.mkdir(parents=True)
                values = {
                    "data": str(a.data.resolve()),
                    "out": str(dest / "training"),
                    "workspace": str(workspace),
                    "python": sys.executable,
                }
                cmd = [x.format(**values) for x in arm["command"]]
                result = dict(name=arm["name"], command=cmd, passed=False, started_at=time.time())
                item["arms"].append(result)
                write_json(a.out / "report.json", receipt)
                run(
                    cmd,
                    workspace,
                    env,
                    dest / "training.log",
                    min(arm.get("timeout", 900), remaining),
                )
                output = dest / "training"
                if arm["kind"] == "offline":
                    status = json.loads((output / "result.json").read_text())
                    if not status["complete"] or status["step"] != 2:
                        raise RuntimeError("Offline update budget incomplete")
                    adapter = Path(status["adapter"])
                elif arm["kind"] == "rl":
                    status = json.loads((output / "status.json").read_text())
                    if (
                        status["status"] != "completed"
                        or status["completed_groups"] != arm["groups"]
                        or status["optimizer_updates"] < 1
                    ):
                        raise RuntimeError("RL training incomplete or no optimizer update")
                    metrics = rows(output / "metrics.jsonl")
                    if arm.get("require_replay") and not any(
                        m.get("replay_tokens", 0) > 0 for m in metrics
                    ):
                        raise RuntimeError("Replay loss was not exercised")
                    if arm.get("require_distillation") and not any(
                        m.get("distillation", {}).get("active") for m in metrics
                    ):
                        raise RuntimeError("Distillation loss was not exercised")
                    adapter = output / "adapter"
                else:
                    raise ValueError("Unknown result checker")
                run(
                    [
                        sys.executable,
                        repository() / "scripts/gpu_method_smoke.py",
                        "--reload",
                        adapter,
                        "--out",
                        dest / "reload.json",
                    ],
                    workspace,
                    env,
                    dest / "reload.log",
                    min(180, remaining),
                )
                result.update(
                    passed=True, seconds=time.time() - result["started_at"], status=status
                )
                write_json(a.out / "report.json", receipt)
            item["passed"] = True
        except Exception as error:
            item["error"] = str(error)
            print("SMOKE FAILED", case["recipe"], str(error), flush=True)
        item["seconds"] = time.time() - item["started_at"]
        write_json(a.out / "report.json", receipt)
    receipt.update(
        passed=all(c["passed"] for c in receipt["cases"]),
        seconds=time.time() - receipt["started_at"],
    )
    write_json(a.out / "report.json", receipt)
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
