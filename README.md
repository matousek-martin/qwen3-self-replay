# Qwen3 Self-Replay

A [ThinkingCap](https://github.com/BottleCapAI/ThinkingCap-Test) submission exploring whether Qwen3-0.6B can learn to answer more briefly from its own verified solutions.

The model samples four answers, learns from correctness-gated rewards, and replays a short solution when all four attempts are correct. Thinking stays enabled; no external teacher supplies the submitted model's training answers.

| Checkpoint | Total tokens saved | Accuracy change | 95% accuracy interval |
|---|---:|---:|---:|
| **256 — primary** | **31.23%** | **+0.21 pp** | **[−0.48, +0.88] pp** |
| 850 — alternative | 41.60% | −1.17 pp | [−1.96, −0.35] pp |

Both use the full 2,759-question × 8-response evaluation. The primary result does not prove unchanged accuracy: it has one training seed, and BFCL no-call accuracy falls 4.17 pp.

Start with **[IDEA.md](IDEA.md)** for the method, results, failed experiments and limitations. [Reproduction instructions](docs/REPRODUCING.md) cover installation, training and checking the saved results. The [release](https://github.com/matousek-martin/qwen3-self-replay/releases/latest) contains adapters, responses and frozen training data.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
pytest -q
```

For details: [method equations](docs/METHOD.md), [evaluation settings](docs/EVALUATION.md), [other experiments](studies/README.md), [available recipes](docs/METHODS.md) and [submission packaging](docs/RELEASE.md).

This is research code, not a deployed inference service. AI assistance is described in [IDEA.md](IDEA.md#ai-assistance); the submission includes selected redacted conversations and tool-call metadata.

Code and adapters: Apache-2.0. Dataset and dependency attribution: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
