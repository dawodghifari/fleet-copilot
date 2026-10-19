"""The challenge cost metric and the threshold search built on it."""

import numpy as np
import pytest

from fleet_copilot.cost import COST_FN, COST_FP, best_threshold, cost_at_threshold, total_cost


def test_total_cost_weights_misses_fifty_times_false_alarms():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])  # one false alarm, one miss
    assert total_cost(y_true, y_pred) == COST_FP + COST_FN == 510


def test_cost_at_threshold_is_inclusive():
    y = np.array([0, 1])
    proba = np.array([0.2, 0.5])
    assert cost_at_threshold(y, proba, 0.5) == 0  # 0.5 >= 0.5 is flagged
    assert cost_at_threshold(y, proba, 0.51) == COST_FN


def test_best_threshold_on_a_perfect_split():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    thr, cost = best_threshold(y, proba)
    assert cost == 0
    assert thr == pytest.approx(0.8)


def test_best_threshold_prefers_flagging_nothing_when_alarms_cost_more():
    # Catching the one positive means flagging 100 negatives ranked above it:
    # 100 * 10 = 1,000 against 500 for missing it.
    y = np.array([0] * 100 + [1])
    proba = np.array([0.9] * 100 + [0.1])
    thr, cost = best_threshold(y, proba)
    assert thr == np.inf
    assert cost == COST_FN


def test_best_threshold_matches_a_brute_force_scan():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.05).astype(int)
    proba = np.clip(rng.normal(0.2 + 0.5 * y, 0.2), 0, 1)
    thr, cost = best_threshold(y, proba)
    brute = min(cost_at_threshold(y, proba, t) for t in np.unique(proba))
    assert cost == min(brute, COST_FN * int(y.sum()))
    assert cost_at_threshold(y, proba, thr) == cost
