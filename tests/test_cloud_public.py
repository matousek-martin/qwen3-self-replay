"""Exercise the new cloud boundary without allocating a provider resource."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from self_replay.artifacts import write_json
from self_replay.cloud import controller
from self_replay.cloud.jobs import JobSpec
from self_replay.methods.runtime import environment, initialize
from self_replay.paths import research


@pytest.fixture
def prepared(tmp_path):
    workspace = initialize(tmp_path / "workspace")
    spec = JobSpec(
        "replay-scaled-test",
        "runpod",
        "experiments/run_public_replay.sh",
        10,
        2,
        1,
        gpu="NVIDIA A100-SXM4-80GB",
        gpu_count=2,
        hardware_scope="replay-scaled-a100-v1",
    )
    path = tmp_path / "spec.json"
    write_json(path, asdict(spec))
    auth = spec.authorization(path, approved=True)
    auth["source_user_instruction"] = "Unit test only; no actual rental"
    approval = tmp_path / "approval.json"
    write_json(approval, auth)
    return workspace, path, approval


def test_launch_passes_only_fresh_contract_and_refuses_duplicate(prepared):
    workspace, spec, approval = prepared
    canonical = Mock()
    with (
        patch.object(controller, "canonical", return_value=canonical),
        patch("subprocess.run", return_value=SimpleNamespace(returncode=0)),
    ):
        controller.launch(workspace, spec, approval)
        passed = canonical.run.call_args.args[0]
        assert passed.hours * (2 * 1.25 + 0.15) <= 9 + 1e-9
        assert passed.public_authorization["authorized"] is True
        assert passed.hardware_scope == "replay-scaled-a100-v1"
        with pytest.raises(ValueError, match="already registered"):
            controller.launch(workspace, spec, approval)
        assert canonical.run.call_count == 1


@pytest.mark.parametrize("failure", ["changed-spec", "unapproved", "guard-inactive"])
def test_no_provider_access_before_checks(prepared, failure):
    workspace, spec, approval = prepared
    if failure == "changed-spec":
        value = json.loads(spec.read_text())
        value["max_cost_usd"] = 11
        write_json(spec, value)
    elif failure == "unapproved":
        value = json.loads(approval.read_text())
        value["authorized"] = False
        write_json(approval, value)
    with (
        patch.object(
            controller, "canonical", side_effect=AssertionError("Provider must not be reached")
        ),
        patch("subprocess.run", return_value=SimpleNamespace(returncode=1)),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            controller.launch(workspace, spec, approval)
    assert not (workspace / "lab/experiments/cloud/authorizations").exists()


def test_ticket_detects_post_registration_provider_changes(tmp_path):
    path = research() / "experiments/public_policy.py"
    spec = importlib.util.spec_from_file_location("public_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = JobSpec("test", "vast", "experiments/run.sh", 10, 2, 1, offer_id=123)
    spec_path = tmp_path / "spec.json"
    write_json(spec_path, asdict(job))
    auth = job.authorization(spec_path, approved=True)
    auth["source_user_instruction"] = "Unit test"
    write_json(tmp_path / "authorizations/test.json", auth)
    assert module.public_ticket(tmp_path, "test", asdict(job)) == auth
    for field, changed in [
        ("gpu", "NVIDIA H100"),
        ("provider", "runpod"),
        ("offer_id", 124),
        ("disk_gb", 200),
        ("max_cost_usd", 11),
        ("max_runtime_hours", 3),
    ]:
        with pytest.raises(ValueError):
            module.public_ticket(tmp_path, "test", {**asdict(job), field: changed})


def test_public_replay_entry_is_present_and_source_locked(prepared):
    workspace, _, _ = prepared
    content = (workspace / "experiments/run_public_replay.sh").read_text()
    assert "--stop-after 256" in content
    assert "--resume output/preflight-reference/checkpoints/group-000002" in content
    assert "experiments/setup_gpu.sh" in content
    assert (
        "experiments/run_public_replay.sh"
        in json.loads((workspace / "workspace.json").read_text())["files"]
    )


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_real_controller_reserves_and_voids_without_capacity(tmp_path, provider):
    workspace = initialize(tmp_path / "workspace")
    script = r"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from self_replay.artifacts import write_json
from self_replay.cloud.jobs import JobSpec
from self_replay.cloud.controller import canonical
import json, sys
workspace = Path.cwd()
spec = JobSpec('replay-scaled-contract-test', sys.argv[1], 'experiments/run_public_replay.sh',
    10, 2, 1, gpu='NVIDIA A100-SXM4-80GB', gpu_count=2,
    hardware_scope='replay-scaled-a100-v1', offer_id=123 if sys.argv[1]=='vast' else None)
spec_path = workspace/'spec.json';write_json(spec_path, asdict(spec))
auth = spec.authorization(spec_path, approved=True)
auth['source_user_instruction'] = 'Synthetic test; all provider transport mocked'
cloud = canonical(workspace)
write_json(cloud.STATE/'authorizations'/f'{spec.name}.json', auth)
args = SimpleNamespace(**{k:v for k,v in asdict(spec).items() if k not in ('offer_id','disk_gb')},
    vast_offer_id=spec.offer_id, vast_disk_gb=spec.disk_gb, hours=spec.hours,
    completion_budget_scope=None, public_authorization=auth)
with patch.object(cloud.vastai, 'api', return_value={'credit':100,'can_pay':True}), \
     patch.object(cloud.vastai, 'ensure_ssh_key'), \
     patch.object(cloud, 'api', side_effect=AssertionError('No rental allowed')):
    try:
        cloud.allocate(args, wait_seconds=0)
    except RuntimeError as e:
        assert 'No eligible GPU capacity' in str(e), str(e)
    else:
        raise AssertionError('Unexpected allocation')
    with cloud.ledger() as state:
        job = state['jobs'][spec.name]
        assert job['safely_no_pod']
        assert job['reserved_usd'] == 0
        assert job['max_cost_usd'] == 10
        assert job['gpu_count'] == 2
        assert job['hard_deadline'] <= job['created_at']+7200
"""
    result = subprocess.run(
        [sys.executable, "-c", script, provider],
        cwd=workspace,
        env=environment(workspace),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
