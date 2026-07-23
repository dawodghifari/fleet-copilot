"""Official Scania APS challenge cost metric.

Cost = 10 * FP + 500 * FN.

A missed APS failure (false negative) means a potential breakdown in the
field — 50x more expensive than an unnecessary workshop check (false
positive). This asymmetry is the whole point of the problem: accuracy is
useless here (predicting all-negative scores 98.3% accuracy and costs
500 * n_pos).
"""

from __future__ import annotations

import numpy as np

COST_FP = 10
COST_FN = 500


def total_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return COST_FP * fp + COST_FN * fn


def cost_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> int:
    return total_cost(y_true, (np.asarray(proba) >= threshold).astype(int))


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, int]:
    """Scan candidate thresholds, return (threshold, cost) minimizing cost.

    Candidates are the unique predicted probabilities — no finer resolution
    exists. Must only ever be called on validation data, never test.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    order = np.argsort(-proba)  # descending probability
    y_sorted = y_true[order]
    # Predicting positive for the top i items: FP = cumulative negatives,
    # FN = total positives - cumulative positives. Vectorized O(n log n).
    cum_pos = np.cumsum(y_sorted)
    cum_neg = np.cumsum(1 - y_sorted)
    total_pos = int(y_true.sum())
    costs = COST_FP * cum_neg + COST_FN * (total_pos - cum_pos)
    all_neg_cost = COST_FN * total_pos  # threshold above max proba
    i = int(np.argmin(costs))
    if costs[i] >= all_neg_cost:
        return float(np.inf), int(all_neg_cost)
    return float(proba[order][i]), int(costs[i])
