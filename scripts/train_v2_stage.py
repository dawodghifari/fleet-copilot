"""v2 training, checkpointed into stages so each fits a short runtime.

Usage:
  python train_v2_stage.py fold <candidate> <fold_idx>   # one CV fold
  python train_v2_stage.py select                        # pick config on OOF
  python train_v2_stage.py final                         # fit full + single test eval

Candidates:
  imputed    — median imputation + missing indicators (v1 config)
  native     — raw features, XGBoost native NaN routing

Calibration was considered and deliberately dropped: the decision
threshold is tuned directly on out-of-fold probabilities for the cost
metric, so monotone recalibration of those probabilities cannot change
the chosen decisions — it would only relabel the threshold value.

v1 flaw being fixed: threshold picked on a single 20% split (200
positives). Here it is picked on 5-fold OOF probabilities (1,000
positives).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CKPT = ROOT / "models" / "v2_checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)

from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
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
N_SPLITS = 5


def make_xgb(spw: float) -> XGBClassifier:
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                         scale_pos_weight=spw, eval_metric="aucpr",
                         n_jobs=-1, random_state=SEED)


def load_features(candidate: str):
    X, y = load_raw(TRAIN_CSV)
    X, dropped = drop_high_missing(X, threshold=0.7)
    ind_cols: list[str] = []
    if candidate == "imputed":
        X, ind_cols = add_missing_indicators(X, min_frac=0.05)
    return X, y, dropped, ind_cols


def build_model(candidate: str, spw: float):
    if candidate == "imputed":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", make_xgb(spw))])
    return make_xgb(spw)


def stage_fold(candidate: str, fold_idx: int) -> None:
    t0 = time.time()
    X, y, _, _ = load_features(candidate)
    spw = float((y == 0).sum() / (y == 1).sum())
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = list(skf.split(X, y))[fold_idx]
    model = build_model(candidate, spw)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    proba = model.predict_proba(X.iloc[va_idx])[:, 1]
    joblib.dump({"va_idx": va_idx, "proba": proba},
                CKPT / f"{candidate}_fold{fold_idx}.joblib")
    print(f"{candidate} fold {fold_idx} done in {time.time() - t0:.0f}s")


def assemble_oof(candidate: str, n: int) -> np.ndarray:
    oof = np.zeros(n)
    for i in range(N_SPLITS):
        d = joblib.load(CKPT / f"{candidate}_fold{i}.joblib")
        oof[d["va_idx"]] = d["proba"]
    return oof


def stage_select() -> None:
    import mlflow
    _, y, _, _ = load_features("native")
    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("aps-predictive-maintenance")
    summary = {}
    for cand in ("imputed", "native"):
        oof = assemble_oof(cand, len(y))
        thr, cost = best_threshold(y.values, oof)
        pr_auc = float(average_precision_score(y, oof))
        summary[cand] = {"oof_cost": int(cost), "oof_pr_auc": round(pr_auc, 4),
                         "threshold": float(thr)}
        with mlflow.start_run(run_name=f"v2-{cand}-oof"):
            mlflow.log_params({"model": f"xgb_{cand}", "cv_folds": N_SPLITS,
                               "seed": SEED})
            mlflow.log_metrics({"oof_cost": cost, "oof_pr_auc": pr_auc,
                                "threshold": thr})
        print(cand, summary[cand])
    winner = min(summary, key=lambda k: summary[k]["oof_cost"])
    json.dump({"winner": winner, **summary},
              open(CKPT / "selection.json", "w"), indent=2)
    print("winner:", winner)


def stage_final() -> None:
    import mlflow
    sel = json.load(open(CKPT / "selection.json"))
    cand = sel["winner"]
    thr = sel[cand]["threshold"]

    X, y, dropped, ind_cols = load_features(cand)
    spw = float((y == 0).sum() / (y == 1).sum())
    model = build_model(cand, spw)
    model.fit(X, y)

    X_test, y_test = load_raw(TEST_CSV)
    X_test, _ = drop_high_missing(X_test, columns=dropped)
    if cand == "imputed":
        X_test, _ = add_missing_indicators(X_test, columns=ind_cols)
    proba_test = model.predict_proba(X_test)[:, 1]
    test_cost = cost_at_threshold(y_test.values, proba_test, thr)
    y_hat = (proba_test >= thr).astype(int)
    fp = int(((y_hat == 1) & (y_test.values == 0)).sum())
    fn = int(((y_hat == 0) & (y_test.values == 1)).sum())
    pr_auc = float(average_precision_score(y_test, proba_test))

    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("aps-predictive-maintenance")
    with mlflow.start_run(run_name=f"FINAL-v2-{cand}"):
        mlflow.log_params({"model": f"xgb_{cand}", "threshold": thr,
                           "final": True, "cv_folds": N_SPLITS})
        mlflow.log_metrics({"test_cost": test_cost, "test_fp": fp,
                            "test_fn": fn, "test_pr_auc": pr_auc})

    joblib.dump({"model": model, "threshold": thr, "dropped": dropped,
                 "indicators": ind_cols, "feature_names": list(X.columns)},
                ROOT / "models" / "best_model_v2.joblib")
    out = {"final_model": f"xgb_{cand}", "test_cost": int(test_cost),
           "fp": fp, "fn": fn, "test_pr_auc": round(pr_auc, 4),
           "threshold": float(thr), "v1_test_cost": 14800,
           "oof_selection": sel}
    json.dump(out, open(ROOT / "reports" / "final_result_v2.json", "w"),
              indent=2)
    print("FINAL v2:", json.dumps(out, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fold":
        stage_fold(sys.argv[2], int(sys.argv[3]))
    elif cmd == "select":
        stage_select()
    elif cmd == "final":
        stage_final()
    else:
        raise SystemExit(f"unknown stage {cmd}")
