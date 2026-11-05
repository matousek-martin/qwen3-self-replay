# Release and submission

## Public release assets

- `Qwen3-0.6B-Self-Replay-256.zip`: primary adapter, model card, original evaluator results and comparison.
- `Qwen3-0.6B-Self-Replay-850.zip`: alternative adapter with the explicit accuracy tradeoff.
- `qwen3-self-replay-responses.zip`: SHA-bound baseline, 256 and 850 records for offline recomputation.
- `self-replay-frozen-data.zip`: prepared train/development questions, plan and dataset provenance; upstream data licenses apply.
- `martin-matousek.zip`: contest upload, under 500 MB, with primary files at root, alternative under `other-checkpoints/checkpoint-850/`, IDEA.md and a self-contained Git bundle.
- `release-manifest.json`: sizes, SHA-256 hashes, source commit and restored-bundle check.

Weights and responses are release assets, not ordinary Git objects. The HF-ready folders are generated under artifacts/huggingface; they keep adapter bytes unchanged and add a model card and evidence. Publishing to a Hub account requires that account's authenticated access. Do not put tokens in a repository, commit message or model card.

## Build

After installing the development/scientific dependencies and staging the released evidence assets:

```bash
pytest -q
python scripts/check_science.py
python scripts/check_numerics.py
python scripts/reproduce_results.py
python scripts/verify_release.py
# Commit source changes before packaging.
python scripts/package_submission.py
```

The exporter `scripts/import_evidence.py --lab PATH` is an optional original-lab migration tool, not a prerequisite for users of the released assets. It reads only its explicit artifact allowlist. Submission packaging requires a clean source tree, verifies evidence, scans reachable text history for known secret patterns, verifies ZIP members, and restores the Git bundle into an empty directory. It never rents compute or sends the form.

## Contest contract

Audited on 2026-11-05 against the [official repository](https://github.com/BottleCapAI/ThinkingCap-Test) and [submission form](https://bottlecapai.com/careers/thinkingcap-test-submission/). Submit the unmodified evaluator's results.json, safetensors adapter, IDEA.md and `<first>-<last>.bundle`; the best checkpoint belongs at the archive root and additional checkpoints under other-checkpoints. The form's ZIP limit is 500 MB. It asks for the applicant's name and the email used in the original application.

Suggested optional note:

> Self-replay for Qwen3-0.6B: checkpoint 256 saves 31.23% of total output tokens, with a +0.21 pp accuracy estimate (95% interval −0.48 to +0.88 pp) on the full official 2,759×8 evaluation. Checkpoint 850 is included as a stronger-compression alternative with a measured accuracy regression. The bundle includes runnable methods, negative experiments, provenance, reusable RunPod/Vast orchestration and an explicit AI-use statement.

The confidence interval does not establish equivalence. No result from a development-only screen is presented as a full official result. Before sending the form, verify the applicant email; a Git commit author address alone does not establish the email used in the application.
