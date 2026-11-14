# Submission package

The [contest requirements](https://github.com/BottleCapAI/ThinkingCap-Test#submission), checked on 2026-11-14, ask for:

- Source code as a Git bundle.
- The evaluator's `results.json`.
- Model or LoRA weights in `.safetensors` format.
- A short `IDEA.md`: motivation, method, results, failures and hardware.
- Coding-agent traces and conversations as a text document, when agents were used.

Put the primary checkpoint at the archive root and alternatives under `other-checkpoints/`. Name the ZIP `<first>-<last>.zip`; the upload limit is 500 MB. Submit using the [contest form](https://bottlecapai.com/careers/thinkingcap-test-submission/).

## Build

From a clean, committed checkout with released model/response assets staged:

```bash
python scripts/reproduce_results.py
python scripts/verify_release.py
python scripts/package_submission.py
```

The packager requires `artifacts/submission-traces/AGENT_TRACES.txt`. Generate it from explicitly selected local logs with `scripts/export_agent_traces.py --session PATH --out PATH`. Our export is selected and redacted, with tool metadata rather than full tool payloads; it is not the complete project history.

Output: `dist/martin-matousek.zip`, plus separate adapter, response and frozen-data archives. The packager verifies hashes, ZIP integrity and restoration of the Git bundle. It neither publishes a release nor submits the form.
