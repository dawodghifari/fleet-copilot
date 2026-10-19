"""The agent's tools, with the model and the fleet feed replaced by fakes.

No trained artifact, dataset download or API key needed.
"""

import numpy as np
import pandas as pd
import pytest

from fleet_copilot.agent import TOOL_SPECS, FleetTools, VehicleStore


class FakeModel:
    def predict_proba(self, X):
        # Probability rises with feature "c"; also proves the column order.
        assert list(X.columns) == ["b", "c", "b_missing"]
        p = X["c"].to_numpy() / 10
        return np.column_stack([1 - p, p])


def make_tools():
    tools = object.__new__(FleetTools)  # skip loading models/ and the test CSV
    tools.model = FakeModel()
    tools.threshold = 0.5
    tools.dropped = ["a"]
    tools.indicators = ["b"]
    tools.feature_names = ["b", "c", "b_missing"]
    store = object.__new__(VehicleStore)
    store.X = pd.DataFrame({"a": [1.0, 1.0], "b": [np.nan, 2.0], "c": [9.0, 1.0]})
    store.y = pd.Series([1, 0])
    tools.store = store
    return tools


def test_prepare_replays_training_preprocessing():
    tools = make_tools()
    row = tools._prepare(tools.store.get(0))
    assert list(row.columns) == ["b", "c", "b_missing"]
    assert row["b_missing"].iloc[0] == 1


def test_predict_vehicle_flags_against_the_threshold():
    tools = make_tools()
    flagged = tools.predict_vehicle(0)
    healthy = tools.predict_vehicle(1)
    assert flagged == {"vehicle_id": 0, "probability_aps_failure": 0.9,
                       "threshold": 0.5, "flagged": True}
    assert healthy["flagged"] is False


def test_unknown_vehicle_raises_key_error():
    # The API turns this KeyError into a 404.
    with pytest.raises(KeyError):
        make_tools().predict_vehicle(99)


def test_every_tool_spec_has_a_method():
    for spec in TOOL_SPECS:
        assert callable(getattr(FleetTools, spec["name"]))
