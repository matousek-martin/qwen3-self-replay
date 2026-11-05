#!/usr/bin/env python3
"""Rebuild publication figures from the preserved machine-readable results."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from self_replay.evaluation.report import history
from self_replay.paths import repository

root = repository()
rows = history(root)
out = root / "studies/self-replay/figures"
out.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    for extension in ("svg", "png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=180, metadata={"Creator": "qwen3-self-replay"})
    plt.close(fig)


fig, ax = plt.subplots(figsize=(9.5, 5.7))
ax.axhline(0, color="#64748b", lw=1, ls="--")
for row in rows:
    avg = row["average"]
    x = 100 * avg["token_reduction"]
    y = 100 * avg["accuracy_delta"]
    ci = avg["intervals"]
    color = (
        "#0f766e"
        if row["checkpoint"] == 256
        else "#7c3aed"
        if row["checkpoint"] == 850
        else "#94a3b8"
    )
    ax.errorbar(
        x,
        y,
        xerr=[[x - 100 * ci["token_reduction"][0]], [100 * ci["token_reduction"][1] - x]],
        yerr=[[y - 100 * ci["accuracy_delta"][0]], [100 * ci["accuracy_delta"][1] - y]],
        fmt="o",
        color=color,
        capsize=3,
        ms=8,
        lw=1.4,
    )
    offsets = {
        128: (-16, 10),
        256: (-10, 13),
        576: (-36, -13),
        768: (-40, 12),
        850: (8, 12),
        1024: (8, 8),
        1280: (8, -8),
    }
    ax.annotate(
        str(row["checkpoint"]),
        (x, y),
        xytext=offsets[row["checkpoint"]],
        textcoords="offset points",
        color=color,
        weight="bold",
    )
ax.set(
    xlabel="Total output tokens saved (%) · official geometric aggregation",
    ylabel="Accuracy change vs. base (percentage points)",
    title="Self-replay: shorter answers, with a measurable tradeoff",
)
ax.set_xlim(23, 51)
ax.grid(alpha=0.15)
fig.text(
    0.125,
    -0.025,
    "2,759 questions × 8 responses · 95% paired-prompt intervals · one training seed\nCheckpoints selected on a repeatedly inspected benchmark; intervals do not include selection or training-seed uncertainty.",
    fontsize=9,
    color="#475569",
)
save(fig, "official-tradeoff")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
labels = ["BFCL", "IFEval", "MBPP+", "SVAMP", "BFCL no-call"]
for n, offset, color in [(256, -0.17, "#0f766e"), (850, 0.17, "#7c3aed")]:
    d = next(row for row in rows if row["checkpoint"] == n)["domains"]
    for ax, key in zip(axes, ["accuracy_delta", "token_reduction"]):
        ax.barh(
            [i + offset for i in range(5)],
            [100 * d[k][key] for k in labels],
            height=0.3,
            color=color,
            label=str(n),
        )
for ax, title, label in zip(
    axes,
    ["Accuracy by domain", "Total tokens saved by domain"],
    ["Change vs. base (pp)", "Reduction vs. base (%)"],
):
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#64748b", lw=1)
    ax.set_title(title)
    ax.set_xlabel(label)
    ax.legend(title="Checkpoint")
    ax.grid(axis="x", alpha=0.15)
fig.tight_layout()
fig.text(
    0.05,
    -0.035,
    "BFCL no-call is a diagnostic subset, not a fifth macro-weighted benchmark. Bars are point estimates.",
    fontsize=9,
)
save(fig, "domains")

curve = json.loads((root / "studies/self-replay/training/validation-curve.json").read_text())[
    "checkpoints"
]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x = [p["iteration"] for p in curve]
for domain, color in [("overall", "#0f766e"), ("coding", "#e76f51"), ("tools", "#7c3aed")]:
    axes[0].plot(
        x, [100 * p["metrics"][domain]["accuracy"] for p in curve], "o-", label=domain, color=color
    )
    axes[1].plot(
        x, [p["metrics"][domain]["mean_tokens"] for p in curve], "o-", label=domain, color=color
    )
for ax in axes:
    ax.set_xlabel("Training questions processed")
    ax.grid(alpha=0.15)
    ax.legend()
axes[0].set_ylabel("Development accuracy (%)")
axes[1].set_ylabel("Mean total output tokens")
fig.suptitle("Independent development monitor · 256 questions × 4 responses")
fig.tight_layout()
save(fig, "development")
with (root / "studies/self-replay/summary.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "checkpoint",
            "total_tokens_saved_pct",
            "accuracy_delta_pp",
            "accuracy_ci_low_pp",
            "accuracy_ci_high_pp",
        ]
    )
    for row in rows:
        a = row["average"]
        w.writerow(
            [
                row["checkpoint"],
                100 * a["token_reduction"],
                100 * a["accuracy_delta"],
                *[100 * v for v in a["intervals"]["accuracy_delta"]],
            ]
        )
print("Built three figures in SVG/PNG/PDF and summary.csv")
