"""Independent cleanup check; never imposes a runtime limit or deletes a pod.

If the main controller dies, keep backing up a running job. Once the remote
runner reports completion/failure, stop compute and retain its persistent disk
for recovery. The main controller owns normal verified deletion.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from runpod import api, ssh_args
from run_baseline import backup, write_json


def should_stop(controller_active, pod_status, remote_exit):
    return (not controller_active and pod_status == "RUNNING"
            and isinstance(remote_exit, dict)
            and isinstance(remote_exit.get("exit_code"), int)
            and bool(remote_exit.get("finished_at")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "pod_id": args.pod_id}
    try:
        active = subprocess.run(["systemctl", "is-active", "--quiet", args.controller]).returncode == 0
        report["controller_active"] = active
        try:
            pod = api("pods/" + args.pod_id)
        except RuntimeError as error:
            if "Runpod HTTP 404:" in str(error):
                report["pod_status"] = "NOT_FOUND"
                return
            raise
        report["pod_status"] = pod["status"]
        if active or pod["status"] != "RUNNING":
            return
        report["mode"] = "recovering_controller_failure"
        result = subprocess.run(ssh_args(pod) + [
            "if test -f /workspace/artifacts/exit.json; then cat /workspace/artifacts/exit.json; else printf null; fi"
        ], check=True, capture_output=True, text=True, timeout=30)
        remote_exit = json.loads(result.stdout)
        report["remote_exit"] = remote_exit
        try:
            backup(pod, args.archive)
            report["backup_succeeded"] = True
        except Exception as error:
            report["backup_succeeded"] = False
            report["backup_error"] = str(error)
        if should_stop(active, pod["status"], remote_exit):
            report["action"] = "stop_compute_retain_disk" if args.apply else "would_stop_compute_retain_disk"
            if args.apply and "stop" in pod.get("actions", []):
                stopped = api("pods/" + args.pod_id + "/action", "POST", {"action": "stop"})
                report["pod_status_after_stop"] = stopped["status"]
        else:
            report["action"] = "leave_job_running"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(args.archive / "guard-status.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
