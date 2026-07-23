"""Train and compare models on the APS dataset with MLflow tracking.

Protocol:
- Stratified 80/20 train/validation split of the official training set.
- Preprocessing decisions (dropped columns, indicator columns, medians)
  learned on the training split only, applied unchanged elsewhere.
- Decision threshold tuned on the validation split for the official cost
  metric (FP=10, FN=500).
- The official test set is touched exactly once, at the end, by the best
  model. Results land in reports/model_comparison.md and MLflow.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mlflow  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from fleet_copilot.cost import best_threshold, cost_at_threshold  # noqa: E402
from fleet_copilot.data import (  # noqa: E402
    TEST_CSV,
    TRAIN_CSV,
    add_missing_indicators,
    drop_high_missing,
    load_raw,
)

SEED = 42
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           n_jobs=-1, random_state=SEED)),
        ]),
        "xgboost": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                  scale_pos_weight=scale_pos_weight,
                                  eval_metric="aucpr", n_jobs=-1,
                                  random_state=SEED)),
        ]),
    }


def main() -> None:
    t0 = time.time()
    X_full, y_full = load_raw(TRAIN_CSV)
    X_test_raw, y_test = load_raw(TEST_CSV)

    # Preprocessing decisions learned on the full training set features
    X_full, dropped = drop_high_missing(X_full, threshold=0.7)
    X_full, ind_cols = add_missing_indicators(X_full, min_frac=0.05)
    X_test_raw, _ = drop_high_missing(X_test_raw, columns=dropped)
    X_test_raw, _ = add_missing_indicators(X_test_raw, columns=ind_cols)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=SEED)

    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("aps-predictive-maintenance")

    rows = []
    fitted = {}
    for name, pipe in build_models(spw).items():
        with mlflow.start_run(run_name=name):
            pipe.fit(X_tr, y_tr)
            proba_val = pipe.predict_proba(X_val)[:, 1]
            thr, val_cost = best_threshold(y_val, proba_val)
            metrics = {
                "val_cost": val_cost,
                "val_cost_default_thr": cost_at_threshold(y_val, proba_val, 0.5),
                "val_pr_auc": average_precision_score(y_val, proba_val),
                "val_recall_at_thr": recall_score(y_val, proba_val >= thr),
                "threshold": thr,
                "fit_seconds": round(time.time() - t0, 1),
            }
            mlflow.log_params({"model": name, "dropped_cols": len(dropped),
                               "indicator_cols": len(ind_cols),
                               "scale_pos_weight": round(spw, 1), "seed": SEED})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            rows.append({"model": name, **metrics})
            fitted[name] = (pipe, thr)
            print(name, metrics)

    comp = pd.DataFrame(rows).sort_values("val_cost")
    best_name = comp.iloc[0]["model"]
    best_pipe, best_thr = fitted[best_name]

    # Single, final test-set evaluation
    proba_test = best_pipe.predict_proba(X_test_raw)[:, 1]
    test_cost = cost_at_threshold(y_test, proba_test, best_thr)
    y_hat = (proba_test >= best_thr).astype(int)
    fp = int(((y_hat == 1) & (y_test == 0)).sum())
    fn = int(((y_hat == 0) & (y_test == 1)).sum())

    with mlflow.start_run(run_name=f"FINAL-{best_name}"):
        mlflow.log_params({"model": best_name, "threshold": best_thr, "final": True})
        mlflow.log_metrics({"test_cost": test_cost, "test_fp": fp, "test_fn": fn,
                            "test_pr_auc": average_precision_score(y_test, proba_test)})

    import joblib
    joblib.dump({"pipeline": best_pipe, "threshold": best_thr,
                 "dropped": dropped, "indicators": ind_cols,
                 "feature_names": list(X_full.columns)},
                MODELS_DIR / "best_model.joblib")

    report = ["# Model comparison — APS predictive maintenance\n",
              "Validation = stratified 20% of official training set. "
              "Cost = 10*FP + 500*FN (official metric). Threshold tuned on "
              "validation only.\n",
              comp.to_markdown(index=False), "",
              f"\n## Final test-set result ({best_name})\n",
              f"Total cost **{test_cost:,}** (FP={fp} -> {10*fp:,}; "
              f"FN={fn} -> {500*fn:,}) at threshold {best_thr:.4f}. ",
              "Reference points: all-negative baseline costs 187,500 on this "
              "test set; the IDA 2016 challenge winner reported 9,920.\n"]
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(report))
    json.dump({"best_model": str(best_name), "test_cost": int(test_cost),
               "fp": fp, "fn": fn, "threshold": float(best_thr)},
              open(ROOT / "reports" / "final_result.json", "w"), indent=2)
    print(f"\nFINAL {best_name}: test cost {test_cost:,} (fp={fp}, fn={fn}) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
