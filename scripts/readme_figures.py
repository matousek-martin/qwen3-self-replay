"""Build the three README comparisons from hash-verified full evaluations."""

import hashlib
import json

import matplotlib.pyplot as plt

BLUE, TEAL, ORANGE = "#2563eb", "#0f766e", "#c26924"


def load(root, relative, expected_sha=None):
    path = root / relative
    evidence = json.loads((root / "provenance/evidence.json").read_text())
    expected_sha = expected_sha or evidence[relative]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Changed figure evidence: {relative}")
    result = json.loads(path.read_text())
    if (
        result["samples"] != 8
        or any(result["full_evaluation"].values())
        or result["ungraded_benchmarks"]
        or result["cohort_sha256"]
        != "16df64e67336b8a1cfa49298cd27e0e1f265927d12ad97f064f3968414d55a86"
        or any(
            sum(v[side]["n"] for v in result["results"].values()) != 22072
            for side in ("base", "candidate")
        )
    ):
        raise ValueError(f"Incomplete or incompatible figure evidence: {relative}")
    return result


def save(fig, out, name):
    # PNG keeps the README/PR small; all plotted values remain in source JSON.
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panels(rows, title, footer, out, name, *, grouped=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5), sharey=True)
    positions = [3.4, 2.4, 1, 0] if grouped else [3, 2, 1, 0]
    limits = [(-2, 38), (-0.9, 1.15)] if grouped else [(-2, 41), (-2.8, 1.15)]
    for ax, metric, label, bounds in zip(
        axes,
        ["token_reduction", "accuracy_delta"],
        ["Total output tokens saved (%)", "Accuracy change vs. base (pp)"],
        limits,
    ):
        ax.axvline(0, color="#cbd5e1", lw=1, zorder=0)
        if grouped:
            ax.axhline(1.7, color="#e2e8f0", lw=1)
        for y, (_, avg, color) in zip(positions, rows):
            value = 100 * avg[metric]
            lo, hi = [100 * x for x in avg["intervals"][metric]]
            ax.errorbar(
                value,
                y,
                xerr=[[value - lo], [hi - value]],
                fmt="o",
                color=color,
                capsize=4,
                ms=8,
                lw=2,
            )
            text = f"{value:.2f}%" if metric == "token_reduction" else f"{value:+.2f} pp"
            ax.annotate(
                text,
                (value, y),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                color=color,
                weight="bold",
            )
        ax.set(xlim=bounds, ylim=(-0.45, positions[0] + 0.65), xlabel=label)
        ax.set_yticks(positions, [r[0] for r in rows])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle(title, fontsize=16, y=0.97)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.84, bottom=0.25, wspace=0.25)
    fig.text(0.29, 0.07, footer, fontsize=8.5, color="#475569")
    save(fig, out, name)


def build(root, out):
    checkpoints = {
        n: load(root, f"studies/self-replay/results/checkpoint-{n}/comparison.json")
        for n in [128, 256, 576, 768, 850, 1024, 1280]
    }
    series = [
        ("Self-replay (submission run)", BLUE, [(n, d["average"]) for n, d in checkpoints.items()])
    ]
    for arm, label, color in [("control", "α2", TEAL), ("accuracy", "α5", ORANGE)]:
        series.append(
            (
                f"ShorterBetter {label}",
                color,
                [
                    (
                        n,
                        load(root, f"studies/experiments/shorterbetter/{arm}-{n:03}.json")[
                            "average"
                        ],
                    )
                    for n in range(5, 31, 5)
                ],
            )
        )
    fig, ax = plt.subplots(figsize=(12, 6.8))
    offsets = {
        128: (-10, 12),
        256: (0, 20),
        576: (-40, 5),
        768: (-12, 15),
        850: (18, 15),
        1024: (17, 6),
        1280: (25, 7),
    }
    for index, (name, color, rows) in enumerate(series):
        xs = [100 * a["token_reduction"] for _, a in rows]
        ys = [100 * a["accuracy_delta"] for _, a in rows]
        ax.plot(xs, ys, "o-", color=color, lw=2, ms=6, label=name)
        for (n, _), x, y in zip(rows, xs, ys):
            selected = index == 0 and n == 256
            offset = (
                offsets[n]
                if index == 0
                else (-18 if index == 1 and n == 20 else 0, 13 if index == 1 else -20)
            )
            ax.annotate(
                "256 · submission" if selected else str(n),
                (x, y),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=10,
                weight="bold" if selected else "normal",
            )
    ax.axhline(0, color="#94a3b8", lw=1, zorder=0)
    ax.set(
        xlim=(5, 69),
        ylim=(-5.8, 1),
        xlabel="Total output tokens saved (%)",
        ylabel="Accuracy change vs. unmodified Qwen3-0.6B (pp)",
    )
    ax.grid(alpha=0.14)
    ax.legend(loc="lower left", frameon=False, fontsize=11)
    fig.suptitle(
        "Self-replay and ShorterBetter · every fully evaluated checkpoint", fontsize=16, y=0.97
    )
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.23, top=0.88)
    fig.text(
        0.10,
        0.055,
        "2,759 questions × 8 responses per checkpoint; all four domains. Points show measured estimates.\n"
        "Labels: training questions for self-replay; rollout iterations for ShorterBetter (different training budgets).\n"
        "Lines connect checkpoints in training order. One seed per run; uncertainty intervals omitted for readability.",
        fontsize=9,
        color="#475569",
    )
    save(fig, out, "shorterbetter-checkpoints")

    rows = []
    for arm, n, label, color in [
        ("no-replay", 128, "Without replay · 128", TEAL),
        ("no-length", 256, "Without length reward · 256", ORANGE),
    ]:
        manifest = json.loads((root / f"studies/self-replay/ablations/{arm}.json").read_text())
        rows.append((f"Original method · {n}", checkpoints[n]["average"], BLUE))
        rows.append(
            (
                label,
                load(root, manifest["comparison"], manifest["comparison_sha256"])["average"],
                color,
            )
        )
    panels(
        rows,
        "What happens when replay or the length reward is removed?",
        "Matched checkpoints: 128 training questions above; 256 below.\n"
        "2,759 questions × 8 responses; values relative to unmodified Qwen3-0.6B.\n"
        "95% paired-prompt intervals; one seed. Selection and seed uncertainty are not included.",
        out,
        "replay-ablations",
        grouped=True,
    )

    rows = [("Checkpoint 256 (reference)", checkpoints[256]["average"], BLUE)]
    baseline = {k: v["base"] for k, v in checkpoints[256]["results"].items()}
    for arm, label, color in [
        ("math", "Math only", TEAL),
        ("instructions", "Instructions only", "#7c3aed"),
        ("mixed", "Math + instructions + tools", ORANGE),
    ]:
        result = load(root, f"studies/experiments/domain-training/{arm}-128.json")
        if {k: v["base"] for k, v in result["results"].items()} != baseline:
            raise ValueError("Domain comparison baseline differs from checkpoint 256")
        rows.append((label, result["average"], color))
    panels(
        rows,
        "Checkpoint 256 and domain-specific alternatives",
        "Reference: original four-domain checkpoint 256; alternatives: 128 training questions each.\n"
        "All evaluated on 2,759 questions × 8 responses; values relative to unmodified Qwen3-0.6B.\n"
        "95% paired-prompt intervals; one seed. Selection and seed uncertainty are not included.",
        out,
        "domain-training-comparison",
    )
