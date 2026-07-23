"""FastAPI demo app for Fleet Copilot.

Run:  uvicorn fleet_copilot.api:app --reload  (from src/, with .env loaded)

Endpoints:
  GET  /vehicles/{id}/prediction   — classifier output for one vehicle
  GET  /vehicles/{id}/explanation  — top SHAP contributions
  POST /ask                        — RAG answer over the knowledge base
  POST /agent                      — full agent (model-as-tool + RAG)
  GET  /                           — minimal HTML UI
"""

from __future__ import annotations

import os
from pathlib import Path

try:  # load .env if python-dotenv is available
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from fleet_copilot.agent import FleetTools, run_agent
from fleet_copilot.rag import answer as rag_answer

app = FastAPI(title="Fleet Copilot", version="0.1")
_tools: FleetTools | None = None


def tools() -> FleetTools:
    global _tools
    if _tools is None:
        _tools = FleetTools()
    return _tools


class Question(BaseModel):
    question: str


@app.get("/vehicles/{vehicle_id}/prediction")
def prediction(vehicle_id: int):
    try:
        return tools().predict_vehicle(vehicle_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/vehicles/{vehicle_id}/explanation")
def explanation(vehicle_id: int):
    try:
        return tools().explain_features(vehicle_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/ask")
def ask(q: Question):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set")
    return rag_answer(q.question)


@app.post("/agent")
def agent(q: Question):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set")
    return run_agent(q.question)


_UI = Path(__file__).with_name("ui.html")


@app.get("/", response_class=HTMLResponse)
def ui():
    return _UI.read_text() if _UI.exists() else "<h1>Fleet Copilot</h1>"
