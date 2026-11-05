"""Show locally archived run progress without printing credentials."""
import json
from pathlib import Path
from runpod import ROOT

active = json.loads((ROOT / "lab/active-run.json").read_text())
archive = Path(active["archive"])
print("Run:", archive.name)
print("W&B:", active["wandb_url"])
print("Local files:", archive)
if active.get("results_dir"):
    print("Final results:", active["results_dir"])
for name in ("progress", "cleanup"):
    path = archive / (name + ".json")
    if path.exists():
        print(name + ":", json.dumps(json.loads(path.read_text())))
if not (archive / "cleanup.json").exists():
    path = archive / "guard-status.json"
    if path.exists():
        print("guard-status:", json.dumps(json.loads(path.read_text())))
