# Method recipes

All entrypoints materialize independently; actual training requires prepared inputs and verified permitted hardware. Historical results do not certify a fresh GPU run of the refactor.

| Recipe | Kind | Default input / prerequisite |
|---|---|---|
| `self-replay` | training | `data/replay-scaled`; `data/replay-scaled/plan.json`, `worker-lease.json` |
| `sft-short` | training | `data/selected/sft-short.jsonl` |
| `sft-random` | training | `data/selected/sft-random.jsonl` |
| `sft-compatible` | training | `data/selected/sft-compatible.jsonl` |
| `sft-dpo-matched` | training | `data/selected/sft-dpo-matched.jsonl` |
| `dpo` | training | `data/selected/dpo.jsonl` |
| `grpo` | training | `data/pilot.jsonl` |
| `dr-grpo` | training | `data/pilot.jsonl` |
| `rl-fixed` | training | `data/pilot.jsonl` |
| `rl-adaptive` | training | `data/pilot.jsonl` |
| `rl-difficulty` | training | `data/pilot.jsonl` |
| `rl-progressive` | training | `data/pilot.jsonl` |
| `rl-replay` | training | `data/pilot.jsonl` |
| `replay-selection` | training | `data/pilot.jsonl` |
| `on-policy-distillation` | training | `data`; `data/isolated/config.json`, `data/isolated/inputs.json`, `data/isolated/sources.json` |
| `adaptive-replay` | training | `data`; `data/isolated/config.json`, `data/isolated/inputs.json`, `data/isolated/sources.json` |
| `teacher-replay` | training | `data`; `data/isolated/config.json`, `data/isolated/inputs.json`, `data/isolated/sources.json`, `data/static/manifest.json` |
| `teacher-generation` | generation | `data` |
| `shorterbetter` | training | `studies/shorterbetter/protocol.json` |
| `opencoder-sft` | training | `data`; `data/opencoder-sft/plan.json`, `data/opencoder-sft/adapter`, `worker-lease.json` |
| `coding-a` | training | `data`; `data/coding-abc/plan.json`, `data/coding-abc/adapter`, `worker-lease.json` |
| `coding-b` | training | `data`; `data/coding-abc/plan.json`, `data/coding-abc/adapter`, `worker-lease.json` |
| `coding-c` | training | `data`; `data/coding-abc/plan.json`, `data/coding-abc/adapter`, `worker-lease.json` |
| `coding-curriculum` | training | `data/coding-curriculum`; `data/coding-curriculum/plan.json`, `worker-lease.json` |

Use `self-replay init workspaces/METHOD --recipe METHOD`, then `self-replay train METHOD --workspace workspaces/METHOD --dry-run` to inspect the exact command. Pass prepared data with `--data`, output with `--out`, and method-specific flags after `--`.

SFT variants consume the verified selector outputs; compatible SFT requires student likelihood scores, and DPO-matched SFT uses the same prompt set as its DPO comparison. RL variants use their source-specific references. Isolated adaptive/OPD/teacher recipes require their registered config, input hashes and caches. Coding A/B/C use the arm in their prepared plan; do not relabel one arm as another.

The self-replay default is a 256-question stop. Its 3,840-question data plan is retained for lineage, not a claim that all questions were trained. A recipe is a runnable implementation, not automatic authorization to generate teacher data or rent hardware.

Teacher generation has explicit `plan`, `models`, and `run` subcommands: for example `self-replay train teacher-generation --workspace WORKSPACE -- plan --out /absolute/teacher-plan.json`. Running API generation requires its separately approved request budget and credentials; inspecting its plan does not generate answers.
