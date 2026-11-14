import importlib.util

from self_replay.paths import repository


def test_package_only_ships_current_named_verification_receipts(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "package_submission", repository() / "scripts/package_submission.py"
    )
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    root = tmp_path / "source"
    verification = root / "artifacts/verification"
    verification.mkdir(parents=True)
    current = [
        "release-check.json",
        "science-tests.json",
        "science-tests.log",
        "numerics-tests.json",
        "numerics-tests.log",
        "recomputed-results.json",
    ]
    excluded = ["publication.json", "packaged-submission.json", "overview-browser.json"]
    for name in current + excluded:
        (verification / name).write_text("{}")
    destination = tmp_path / "output"
    package.stage_verification(root, destination)
    assert {p.name for p in destination.iterdir()} == set(current)
