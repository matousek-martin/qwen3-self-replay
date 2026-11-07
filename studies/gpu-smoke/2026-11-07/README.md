# Main submission GPU smoke checks — 2026-11-07

**Passed.** I exercised the public submission at commit `ec6489b` on one RunPod
worker with **2× NVIDIA A100-SXM4-80GB**. The suite took **28.5 minutes**, excluding
provisioning and cleanup. The controller verified the final output archive before
release; subsequent complete provider listings showed no active RunPod or Vast
rentals. The conservative combined charge estimate was **$2.63**, within a $6 cap.

These are execution checks. They do not replace the reported 2,759×8 evaluations,
retrain checkpoint 256/850, or establish the method's effectiveness.

| Check | Observed result |
|---|---|
| Public CLI and frozen-data staging | Source/data verification and trainer launch passed |
| Synthetic four-question training | Completed with three optimizer updates and a nonempty replay buffer |
| Complete-state resume from question 2 | Final adapter, Adam, replay and per-rank RNG state bitwise equal to the uninterrupted run |
| Maximum-length backward pass | 2,048-token backward check passed |
| Normal training path | Four original training rows, one per domain, at the full 2,048-token cap; three optimizer updates, development validation and complete SHA-verified checkpoint |
| Released adapters 256 and 850 | Published file hashes matched; vLLM base→adapter→base routing, measurable adapter effect and base restoration passed for both |
| Official evaluator smoke mode | 30 questions × 8 responses per adapter; 480 responses total; SVAMP, MBPP+, BFCL and IFEval generated and graded with no failed benchmark |
| Cloud lifecycle | RunPod allocation, worker launch, monitoring, output backup verification and confirmed release passed |

## Scope and limitations

The synthetic resume pilot uses its original 256-token generation cap. The
separate normal-path probe uses a new, hashed technical subset with four training
and four development questions, and one calibration question per domain. Its
cohort/calibration differ from the scientific run. The original dataset and
released weights are unchanged; benchmark questions remain evaluation-only.

Evaluation used one A100 at a time and the pinned main inference libraries;
[packages.txt](packages.txt) records the complete installed environment. The
official evaluator selected its `rlimit` sandbox fallback (`auto_downgrade`), so
this run does not validate namespace isolation. No full official baseline was
regenerated; the base routing check uses a tiny synthetic prompt. Smoke scores are
not evidence of model quality. The other 23 method entrypoints were not GPU-tested
in this campaign.

## Failures found and fixed

- Vast returned an unavailable offer. I preserved a sanitized provider error in
  the client and reconciled both failed creates before retrying; this campaign
  does **not** establish successful execution on Vast.
- The first RunPod staging attempt exposed a systemd PATH bug: `uv` was installed
  but not found by the controller. I added executable resolution and a preflight
  before allocation/approval storage, with a regression test. The failed worker
  was confirmed released before the successful replacement.

The scientific update rule did not need a change. All 46 local public tests passed;
the existing 79 release evidence files retained their hashes.

## Evidence and reuse

[report.json](report.json) records every phase, duration and outcome.
[resume-parity.json](resume-parity.json), [maximum-length-check.json](maximum-length-check.json),
[full-cap-completion.json](full-cap-completion.json), routing receipts and evaluator
results are original worker outputs. [provenance.json](provenance.json) records
their hashes, the tested source/archive, finite budget, attempt lifecycle and
verified backup manifest. These fresh checks are separate from historical evidence.

To repeat the bounded suite on an explicitly leased worker, follow
[GPU_SMOKE.md](../../../docs/GPU_SMOKE.md). The script itself never allocates compute.
