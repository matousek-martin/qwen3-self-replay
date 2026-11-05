# One job lifecycle, two providers

The public interface delegates to the original canonical controller; it does not create a second allocator or ledger. RunPod and Vast share finite reservations, unique job identity, startup checks, monitoring, archive verification and release.

## Configure

Credentials stay in controller environment variables `RUNPOD_API_KEY` or `VASTAI_API_KEY`; the SSH key path is `SELF_REPLAY_SSH_KEY`. They are never included in the uploaded source or dataset payload. Use a fresh initialized workspace with a local ledger under lab/experiments/cloud. Historical grants are not copied and the default unfunded allowance is zero.

A job JSON declares name, provider, entry script, exact GPU/count, hardware scope, cost ceiling, runtime ceiling, maximum GPU hourly rate, disk and (for Vast) a current offer ID. Examples are in configs/cloud. Training defaults to RTX 4090; explicitly scoped original replay/curriculum/ShorterBetter A100 recipes preserve their separate worker checks. Evaluation is distinguished from training.

```bash
self-replay cloud plan --spec configs/cloud/runpod-example.json --out artifacts/new-approval.json
self-replay cloud status --workspace workspaces/replay
```

Planning writes an **unapproved** contract and never contacts the provider. After the operator supplies the new approval/source and required inputs, `cloud run --workspace ... --spec ... --approval ...` can execute it. The exact spec hash and every cost/hardware/provider field must match the approval. Zero, NaN, infinity, missing inputs and duplicate job registration are rejected.

The replay examples use `experiments/run_public_replay.sh`: fresh numerical/resume preflight followed by training to question 256. Stage the frozen data under `WORKSPACE/data/replay-scaled` before launch; the public controller uploads that workspace data directory. A 10 USD example cap is illustrative and may stop well before 256. Host requirements include `uv`, `rsync`, OpenSSH and systemd.

## Independent guard

Install the example systemd service/timer from configs/systemd with this checkout's Python executable and the exact workspace path. The timer must be active before launch. Controller and guard must use the same workspace/ledger. The examples are templates, not enabled services; this release preparation did not install or start them.

## Lifecycle and failures

- Before allocation: reserve the entire finite envelope, account for other jobs, verify hardware policy and the independent guard.
- After allocation: verify provider identity, actual rate, SSH availability and actual worker GPU before training.
- Ambiguous create/delete: retain the reservation and reconcile the uniquely named resource; never blindly create a replacement.
- Controller restart: `cloud watch --workspace ... --name ...` resumes observation; it does not allocate, relaunch or reset the budget.
- Completion: synchronize outputs, check archive SHA/completion markers, then release and verify provider absence.
- Hard deadline: retain the existing emergency cleanup path when a final transfer cannot safely finish; report incomplete backup rather than inventing success.
- Baseline evaluation: reuse only a complete compatible cache entry; a missing required baseline is not permission to regenerate it.

Cloud tests use mocked provider calls. No new rental was used to validate this public refactor. Existing campaign-specific operational recovery scripts remain historical implementation details, not new funding permissions or automatic launch instructions.
