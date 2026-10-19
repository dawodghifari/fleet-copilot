"""Raw loading and the train-learned preprocessing decisions."""

import numpy as np
import pandas as pd

from fleet_copilot.data import add_missing_indicators, drop_high_missing, load_raw


def test_load_raw_parses_na_and_target(tmp_path):
    csv = tmp_path / "aps.csv"
    csv.write_text("class,aa_000,ab_000\nneg,1,na\npos,2,3\n")
    X, y = load_raw(csv)
    assert list(y) == [0, 1]
    assert "class" not in X.columns
    assert np.isnan(X.loc[0, "ab_000"])


def test_drop_high_missing_learns_on_train_and_replays_on_test():
    train = pd.DataFrame({"a": [1, np.nan, np.nan, np.nan], "b": [1, 2, 3, np.nan]})
    test = pd.DataFrame({"a": [1, 2], "b": [np.nan, np.nan]})
    train_out, dropped = drop_high_missing(train, threshold=0.7)
    assert dropped == ["a"]
    assert list(train_out.columns) == ["b"]
    # The test set uses the training decision, not its own missingness.
    test_out, _ = drop_high_missing(test, columns=dropped)
    assert list(test_out.columns) == ["b"]


def test_missing_indicators_keep_the_signal():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 3.0]})
    out, cols = add_missing_indicators(df, min_frac=0.05)
    assert cols == ["a"]
    assert list(out["a_missing"]) == [0, 1, 0]
    assert "b_missing" not in out.columns
