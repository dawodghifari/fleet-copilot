"""Data loading and preprocessing for the APS Failure dataset.

The raw files use the string 'na' for missing values. The target column
'class' is 'pos' (APS-related failure) or 'neg' (failure unrelated to APS).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TRAIN_CSV = RAW_DIR / "aps_failure_training_set.csv"
TEST_CSV = RAW_DIR / "aps_failure_test_set.csv"


def load_raw(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a raw APS csv. Returns (features, target) with NaN for 'na'."""
    df = pd.read_csv(path, na_values="na")
    y = (df.pop("class") == "pos").astype(int)
    return df, y


def missingness(df: pd.DataFrame) -> pd.Series:
    """Fraction of missing values per column, descending."""
    return df.isna().mean().sort_values(ascending=False)


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.7,
                      columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with more than `threshold` missing values.

    If `columns` is given (e.g. learned from the training set), drop exactly
    those instead — the test set must use the training set's decision.
    """
    if columns is None:
        frac = df.isna().mean()
        columns = sorted(frac[frac > threshold].index.tolist())
    return df.drop(columns=columns), columns


def add_missing_indicators(df: pd.DataFrame, min_frac: float = 0.05,
                           columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Add binary <col>_missing indicators for columns with notable missingness.

    Missingness in sensor data is often informative (sensor absent/broken),
    so we keep the signal instead of silently imputing over it.
    """
    if columns is None:
        frac = df.isna().mean()
        columns = sorted(frac[frac >= min_frac].index.tolist())
    out = df.copy()
    for c in columns:
        out[f"{c}_missing"] = df[c].isna().astype(np.int8)
    return out, columns
