"""Read official comparisons without changing weighting or confidence intervals."""

import json
from pathlib import Path


def domains(comparison):
    result = {}
    groups = {
        "BFCL": ["bfcl_irrelevance", "bfcl_multiple", "bfcl_simple_python"],
        "IFEval": ["ifeval"],
        "MBPP+": ["mbpp"],
        "SVAMP": ["svamp"],
        "BFCL no-call": ["bfcl_irrelevance"],
    }
    for domain, keys in groups.items():
        metrics = {}
        for side in ("base", "candidate"):
            rows = [comparison["results"][key][side] for key in keys]
            count = sum(row["n"] for row in rows)
            metrics[side] = {
                "accuracy": sum(row["accuracy"] * row["n"] for row in rows) / count,
                "mean_tokens": sum(row["total_output_tokens"] for row in rows) / count,
            }
        metrics["accuracy_delta"] = metrics["candidate"]["accuracy"] - metrics["base"]["accuracy"]
        metrics["token_reduction"] = (
            1 - metrics["candidate"]["mean_tokens"] / metrics["base"]["mean_tokens"]
        )
        result[domain] = metrics
    return result


def history(root):
    root = Path(root)
    rows = json.loads((root / "studies/self-replay/history.json").read_text())
    for row in rows:
        comparison = json.loads((root / row["comparison"]).read_text())
        if comparison["average"] != row["average"]:
            raise ValueError("History and source comparison disagree")
        row["domains"] = domains(comparison)
    return rows
