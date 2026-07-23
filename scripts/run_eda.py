"""EDA for the APS dataset. Writes figures to reports/figures and a
markdown summary to reports/eda_summary.md."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleet_copilot.data import TRAIN_CSV, load_raw, missingness  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    X, y = load_raw(TRAIN_CSV)
    lines: list[str] = ["# EDA summary — APS Failure training set\n"]

    lines.append(f"Rows: {len(X):,}. Features: {X.shape[1]}. "
                 f"Positives: {int(y.sum()):,} ({y.mean():.2%}) — "
                 "heavy class imbalance.\n")

    # Missingness
    miss = missingness(X)
    lines.append(f"Columns with any missing values: {(miss > 0).sum()} of {len(miss)}. "
                 f"Worst column: `{miss.index[0]}` at {miss.iloc[0]:.1%}. "
                 f"Columns >70% missing: {(miss > 0.7).sum()} "
                 f"({', '.join('`%s`' % c for c in miss[miss > 0.7].index)}).\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(miss.values, bins=40, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Fraction missing")
    ax.set_ylabel("Number of columns")
    ax.set_title("Missingness across 170 features")
    fig.tight_layout()
    fig.savefig(FIG / "missingness_hist.png", dpi=120)

    # Histogram-bin feature groups: columns like ag_000..ag_009 are bins of
    # one physical histogram counter.
    prefixes = sorted({c.split("_")[0] for c in X.columns})
    group_sizes = {p: sum(c.startswith(p + "_") for c in X.columns) for p in prefixes}
    hist_groups = {p: n for p, n in group_sizes.items() if n >= 10}
    lines.append(f"Feature name prefixes: {len(prefixes)}. Histogram-style groups "
                 f"(>=10 bins): {hist_groups} — these are binned distributions of "
                 "single physical quantities (e.g. time spent in load ranges), "
                 "useful for aggregate features later.\n")

    # Class separation on the most complete numeric features
    complete = miss[miss < 0.01].index[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, col in zip(axes.ravel(), complete):
        for label, name in [(0, "neg"), (1, "pos")]:
            vals = np.log1p(X.loc[y == label, col].dropna())
            ax.hist(vals, bins=40, alpha=0.55, label=name, density=True)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("log1p distributions by class — most complete features")
    fig.tight_layout()
    fig.savefig(FIG / "class_separation.png", dpi=120)

    # Cost framing
    lines.append("Baseline costs (train): predict all-negative = "
                 f"{500 * int(y.sum()):,}; predict all-positive = "
                 f"{10 * int((1 - y).sum()):,}. Any useful model must land far "
                 "below both; accuracy is meaningless at 1.7% prevalence.\n")

    (ROOT / "reports" / "eda_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
