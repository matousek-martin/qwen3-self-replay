import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from self_replay.artifacts import sha256, verify_files, write_json
from self_replay.cloud.jobs import JobSpec, validate_authorization
from self_replay.data.prepare import stage_frozen
from self_replay.evaluation.report import domains, history
from self_replay.methods import runtime
from self_replay.paths import repository


def test_path_escape_and_corruption(tmp_path):
    (tmp_path / "a").write_text("one")
    digest = sha256(tmp_path / "a")
    assert verify_files(tmp_path, {"a": digest}) == 1
    with pytest.raises(ValueError):
        verify_files(tmp_path, {"../a": digest})
    (tmp_path / "a").write_text("two")
    with pytest.raises(ValueError):
        verify_files(tmp_path, {"a": digest})


@pytest.mark.parametrize("name", list(runtime.recipes()))
def test_every_recipe_materializes_and_resolves_entrypoint(tmp_path, name):
    workspace = runtime.initialize(tmp_path / name, name)
    args = runtime.command(name, workspace)
    if "-m" in args:
        module = args[args.index("-m") + 1]
        if module == "torch.distributed.run":
            module = args[args.index("-m", args.index("-m") + 1) + 1]
        assert (workspace / (module.replace(".", "/") + ".py")).is_file()
    else:
        assert (workspace / args[1]).is_file()
    assert runtime.verify(workspace)["recipe"] == name
    with patch("subprocess.run", side_effect=AssertionError("dry-run must not execute")):
        assert runtime.train(name, workspace, dry_run=True) == args


def test_workspace_refuses_overwrite_and_changed_source(tmp_path):
    workspace = runtime.initialize(tmp_path / "run")
    with pytest.raises(ValueError):
        runtime.initialize(workspace)
    (workspace / "experiments/replay_scaled/worker.py").write_text("changed")
    with pytest.raises(ValueError):
        runtime.verify(workspace)


@pytest.mark.parametrize("provider", ["runpod", "vast"])
def test_fresh_authorization_required_and_bound_to_spec(tmp_path, provider):
    spec = JobSpec(
        "replay-scaled-example",
        provider,
        "experiments/replay_scaled/run.sh",
        10,
        2,
        1,
        offer_id=123 if provider == "vast" else None,
    )
    path = tmp_path / "spec.json"
    from dataclasses import asdict

    write_json(path, asdict(spec))
    auth = spec.authorization(path)
    with pytest.raises(ValueError):
        validate_authorization(spec, path, auth)
    auth.update(authorized=True, source_user_instruction="Fixture approval, no provider calls")
    assert validate_authorization(spec, path, auth)
    with pytest.raises(ValueError):
        validate_authorization(replace(spec, max_cost_usd=11), path, auth)
    assert spec.hours * spec.budget_rate <= spec.max_cost_usd - 1 + 1e-9


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_budget_never_reaches_provider(value):
    with pytest.raises(ValueError):
        JobSpec("test", "runpod", "experiments/run.sh", value, 2, 1).validate()


def test_training_hardware_is_scoped():
    base = JobSpec("test", "runpod", "experiments/run.sh", 10, 2, 1)
    with pytest.raises(ValueError):
        replace(base, gpu="NVIDIA H100 80GB HBM3").validate()
    good = replace(
        base,
        name="replay-scaled-example",
        hardware_scope="replay-scaled-a100-v1",
        gpu="NVIDIA A100-SXM4-80GB",
        gpu_count=2,
    )
    assert good.validate()
    with pytest.raises(ValueError):
        replace(good, gpu_count=1).validate()


def test_official_macro_and_subset_visibility():
    rows = history(repository())
    assert [r["checkpoint"] for r in rows] == [128, 256, 576, 768, 850, 1024, 1280]
    for row in rows:
        data = json.loads((repository() / row["comparison"]).read_text())
        d = domains(data)
        macro = sum(d[k]["accuracy_delta"] for k in ("BFCL", "IFEval", "MBPP+", "SVAMP")) / 4
        assert macro == pytest.approx(data["average"]["accuracy_delta"])
    cp256 = next(x for x in rows if x["checkpoint"] == 256)
    assert cp256["domains"]["BFCL no-call"]["accuracy_delta"] < -0.04


def test_frozen_data_cannot_silently_change_splits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for split in ("train", "development"):
        (source / (split + ".jsonl")).write_text(
            json.dumps({"id": "same", "source": "gsm8k", "split": split}) + "\n"
        )
    plan = tmp_path / "plan.json"
    write_json(plan, {"files_sha256": {p.name: sha256(p) for p in source.iterdir()}})
    with pytest.raises(ValueError, match="overlap"):
        stage_frozen(source, tmp_path / "target", plan)
