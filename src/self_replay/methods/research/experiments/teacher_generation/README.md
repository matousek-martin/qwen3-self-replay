# Teacher generation

Owned by teacher-generation-20261025. Direct OpenAI Responses API for the original 128 training prompts; no training, GPU allocation, trainer edits, or domain expansion. The [replay protocol](../../studies/teacher-pilot/REPLAY-PROTOCOL.md) and [integration request](../../studies/replay-selection/TEACHER-INTEGRATION-REQUEST.md) define the next stage.

## Prepared comparison

GPT-6 Luna and GPT-6.1 Sol, each concise/explicit, 512 requests total. Official Standard input/output prices checked 2026-10-25: $0.10/$0.50 and $2/$10 per million tokens. [Pricing](https://developers.openai.com/api/docs/pricing), [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol). These model IDs are listed in this account's read-only models response. No dated snapshot is published on those model pages; do not claim immutable teacher weights or verified generation permission. Returned model/tier mismatches stop the run.

Each request caps total output, including billed internal reasoning, at 2048 tokens. Low reasoning effort in both models; no tools, previous conversation, API seed, temperature, or retries. Ordinary assistant text contains the teachable explanation; API reasoning output is preserved as provenance but is not a training target. [Reasoning API behavior](https://developers.openai.com/api/docs/guides/reasoning).

Plan reservation uses serialized input UTF-8 bytes plus 2048 framing tokens, all input at the maximum published cache-write rate (12h), plus full output cap. For this plan the conservative reserve is **$9.271562**; rounded proposal **$9.28 within the existing campaign**, not an increase or an approved amount. Cached-input discounts are recorded in a separate nominal usage estimate; conservative accounting remains until invoice reconciliation. No provider invoice or credit balance is implied. Actual usage exceeding a bound stops generation. Reverify pricing after seven days; plan validation checks code/input hashes.

## Commands

Planning and tests use Python's standard library; tokenizer verification uses the isolated local `.venv-teacher`. No model weights or PyTorch are required. Its package versions are in [the environment snapshot](../../studies/teacher-pilot/evidence/teacher-environment.txt).

```bash
python3 -m experiments.teacher_generation.runner plan --out /path/to/new-plan.json
python3 -m experiments.teacher_generation.runner models --out /path/to/models-evidence.json
.venv-teacher/bin/python -m unittest experiments.teacher_generation.test_teacher_generation -v
```

The first command is offline; `models` performs only `GET /v1/models`. The prepared plan is [openai-plan.json](../../studies/teacher-pilot/openai-plan.json). Model metadata and prices live in [openai-comparison.json](openai-comparison.json); the full request bodies, identities, caps, prices, input and source hashes are frozen in the plan. Gold answers are never sent to OpenAI.

Paid execution is deliberately separate and requires an approval record with `approved:true`, scope `teacher-pilot-openai`, canonical `plan_sha256`, `max_usd` covering the entire plan, a future Unix `expires_at`, and `evidence_path`/`evidence_sha256` referencing actual user authorization. The [template](../../studies/teacher-pilot/authorization.example.json) is **not approved**. An agent must not turn it into authorization by itself. A changed plan must be reviewed again; overlap with an existing plan is blocked to avoid duplicate paid requests.

After actual authorization, use a fresh run directory outside the historical input directory:

```bash
python3 -m experiments.teacher_generation.runner run \
  --plan studies/teacher-pilot/openai-plan.json \
  --approval /path/to/actual-approval.json \
  --out lab/experiments/teacher-generation/approved-run --max-requests 8
# After inspecting the technical smoke responses, resume the SAME directory
# with the SAME arguments, omitting --max-requests to finish the plan.
```

No API reservation has been created by preparing the code. Check live campaign/controller/provider jobs and reservations before authorizing a run. The runner atomically reserves in the existing `lab/experiments/cloud/ledger.json` using the same `cloud/lock`; cap $200, usable $180. API entries use `kind:api`, `state:api_reserved`, zero hourly rate and no pod, which the existing GPU guard and slot counter already handle. `cloud.liability` includes the whole reservation, including after a timeout, pause, restart or unknown HTTP result. On full completion it is settled to conservative usage cost. `released_at` then denotes release of this API budget reservation, not a fictional GPU rental. No changes to cloud.py are required, including for an already-running controller. All unrelated jobs are preserved under the shared lock.

## Durable evidence and recovery

A nonblocking run lock prevents simultaneous senders. Each request has a stable prompt/model/variant/payload identity and `X-Client-Request-Id` (a tracing ID, **not** a guarantee of server idempotency). Before POST, `requests/<hash>.json` is atomically written and fsynced as `dispatching`. Response headers' provider request ID, full sanitized response body, provider response ID, statuses, usage, prices and timestamps are durably saved before processing. Source snapshots use an explicit code allowlist; credentials and `.env*` are never archived. The key is read in-process through a strict literal dotenv parser, never sourced/exported to a shell or printed.

On `response_saved`, restart can finish local processing without another POST. On `dispatching` after a crash, timeout, HTTP error, invalid/missing usage, wrong model/tier or malformed response, stop and retain the full reservation. There are no automatic retries, including for 429/5xx. Operator reconciliation must use request/response IDs and provider usage evidence. With `store:false`, do not assume a lost response is retrievable. If it cannot be reconciled, keep its liability and report the missing sample; never delete its request file to cause a resend. A completed/settled run refuses further paid execution. A normal interrupted run resumes completed items without resending them. The runner is synchronous/sequential; provider rate limits are not inferred from model availability.

Verification is offline and writes a new directory:

```bash
.venv-teacher/bin/python -m experiments.teacher_generation.verify \
  --run lab/experiments/teacher-generation/approved-run \
  --out lab/experiments/teacher-generation/verified-v1
```

The independent contest **training** grader checks final answers. Incomplete, refused, malformed, wrong, empty-explanation and >2048 student-token targets are rejected. Exact grader/code/tokenizer provenance is saved. `accepted_answer` does not mean explanation correctness: every target defaults to `replay_eligible:false`. A separate `--reviews /path/reviews.jsonl` can admit independently reviewed explanations, keyed by request and response hash:

```json
{"request_key":"...","response_sha256":"...","reviewer":"human or independent verifier identity","verdict":"pass","evidence":"Specific reasoning check and result"}
```

A correct final answer with a faulty explanation must receive `fail`. No paid LLM judge is silently called. Actual teacher/student selection, pool export, training and evaluation remain separate stages in the protocol; no model-quality or learning result has been measured by CPU fixtures.
