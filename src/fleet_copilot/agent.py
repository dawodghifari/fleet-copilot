"""Agent layer: explains model predictions using tools.

The agent answers questions like "why is vehicle 4137 flagged?" by
calling three tools:
  - predict_vehicle: run the trained classifier on one vehicle row
  - explain_features: top feature contributions for that prediction
  - search_knowledge: retrieve maintenance/dataset docs from the RAG index

Design notes (documented for the writeup):
- The model is a real artifact loaded from models/, not a mock — the
  agent's answers are grounded in the actual classifier output.
- The system prompt requires citations of retrieved chunks and forbids
  invented feature meanings (features are anonymized; only histogram
  groups have documented physical interpretations).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fleet_copilot.data import TEST_CSV, load_raw  # noqa: E402
from fleet_copilot.rag import retrieve  # noqa: E402

MODEL_PATH_V2 = ROOT / "models" / "best_model_v2.joblib"
MODEL_PATH_V1 = ROOT / "models" / "best_model.joblib"


class VehicleStore:
    """Test-set rows standing in for a live fleet feed."""

    def __init__(self) -> None:
        X, y = load_raw(TEST_CSV)
        self.X, self.y = X, y

    def get(self, vehicle_id: int) -> pd.DataFrame:
        if not 0 <= vehicle_id < len(self.X):
            raise KeyError(f"vehicle_id must be 0..{len(self.X) - 1}")
        return self.X.iloc[[vehicle_id]]


class FleetTools:
    def __init__(self) -> None:
        path = MODEL_PATH_V2 if MODEL_PATH_V2.exists() else MODEL_PATH_V1
        bundle = joblib.load(path)
        self.model = bundle.get("model") or bundle.get("pipeline")
        self.threshold = bundle["threshold"]
        self.dropped = bundle["dropped"]
        self.indicators = bundle["indicators"]
        self.feature_names = bundle["feature_names"]
        self.store = VehicleStore()

    def _prepare(self, row: pd.DataFrame) -> pd.DataFrame:
        row = row.drop(columns=[c for c in self.dropped if c in row.columns])
        for c in self.indicators:
            row[f"{c}_missing"] = row[c].isna().astype(int)
        return row[self.feature_names]

    def predict_vehicle(self, vehicle_id: int) -> dict:
        row = self._prepare(self.store.get(vehicle_id))
        proba = float(self.model.predict_proba(row)[0, 1])
        return {"vehicle_id": vehicle_id,
                "probability_aps_failure": round(proba, 4),
                "threshold": round(float(self.threshold), 4),
                "flagged": bool(proba >= self.threshold)}

    def explain_features(self, vehicle_id: int, top_n: int = 8) -> dict:
        """Top feature contributions via XGBoost's pred_contribs (SHAP values)."""
        row = self._prepare(self.store.get(vehicle_id))
        booster = getattr(self.model, "get_booster", lambda: None)()
        if booster is None:  # calibrated wrapper — use first inner estimator
            inner = self.model.calibrated_classifiers_[0].estimator
            booster = inner.get_booster()
        import xgboost as xgb
        contribs = booster.predict(
            xgb.DMatrix(row, missing=np.nan), pred_contribs=True)[0]
        pairs = sorted(zip(self.feature_names, contribs[:-1]),
                       key=lambda p: -abs(p[1]))[:top_n]
        return {"vehicle_id": vehicle_id,
                "top_contributions": [
                    {"feature": f, "shap": round(float(s), 4),
                     "value": (None if pd.isna(row.iloc[0][f])
                               else float(row.iloc[0][f]))}
                    for f, s in pairs]}

    def search_knowledge(self, query: str, k: int = 4) -> list[dict]:
        hits = retrieve(query, strategy="sections", k=k)
        return [{"chunk_id": h["chunk_id"], "text": h["text"][:600]}
                for h in hits]


TOOL_SPECS = [
    {"name": "predict_vehicle",
     "description": "Run the APS failure classifier on one vehicle. Returns "
                    "probability, decision threshold, and flagged status.",
     "input_schema": {"type": "object",
                      "properties": {"vehicle_id": {"type": "integer"}},
                      "required": ["vehicle_id"]}},
    {"name": "explain_features",
     "description": "Top SHAP feature contributions for one vehicle's "
                    "prediction. Feature names are anonymized sensor counters.",
     "input_schema": {"type": "object",
                      "properties": {"vehicle_id": {"type": "integer"},
                                     "top_n": {"type": "integer"}},
                      "required": ["vehicle_id"]}},
    {"name": "search_knowledge",
     "description": "Semantic search over APS maintenance docs, dataset "
                    "documentation, model card, and workshop playbook.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"},
                                     "k": {"type": "integer"}},
                      "required": ["query"]}},
]

SYSTEM = """You are Fleet Copilot, a maintenance triage assistant for a
heavy-truck fleet. You have tools to run the APS failure classifier, get
per-feature SHAP contributions, and search maintenance documentation.

Rules:
- Ground every claim in tool output; cite knowledge chunks as [chunk_id].
- Feature names are anonymized (e.g. ag_002). Never invent what a feature
  physically means. Histogram groups (ag, ay, az, ba, cn, cs, ee) are
  documented as binned operating-condition distributions — you may say
  that much and no more.
- If a vehicle is flagged, include: probability vs threshold, the top
  contributing features, and the recommended triage steps from the
  playbook.
- State uncertainty plainly. A flag is a prioritization signal, not a
  diagnosis."""


def run_agent(question: str, model: str = "claude-sonnet-5",
              max_turns: int = 8) -> dict:
    import anthropic

    tools = FleetTools()
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": question}]
    trace = []
    for _ in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=1500,
                                      system=SYSTEM, tools=TOOL_SPECS,
                                      messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {"answer": text, "trace": trace}
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            fn = getattr(tools, block.name)
            try:
                out = fn(**block.input)
            except Exception as e:  # surface tool errors to the model
                out = {"error": str(e)}
            trace.append({"tool": block.name, "input": block.input,
                          "output": out})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    return {"answer": "(agent exceeded max turns)", "trace": trace}
