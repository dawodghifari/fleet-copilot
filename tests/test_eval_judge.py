"""The retrieval-eval judge: the part of the eval that had a fairness bug."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_rag.py"
_spec = importlib.util.spec_from_file_location("eval_rag", _PATH)
eval_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_rag)
relevant = eval_rag.relevant


def hit(doc, heading="", text=""):
    return {"meta": {"doc": doc, "heading": heading}, "text": text}


def test_wrong_document_is_never_relevant():
    assert not relevant(hit("model_card", "Origin"), {"doc": "dataset_documentation"})


def test_document_match_is_enough_without_a_heading():
    assert relevant(hit("dataset_documentation"), {"doc": "dataset_documentation"})


def test_heading_match_from_metadata():
    expect = {"doc": "air_pressure_system", "heading_contains": "failure modes"}
    assert relevant(hit("air_pressure_system", heading="Common failure modes"), expect)


def test_heading_match_from_body_for_fixed_chunks():
    # Fixed-size chunks carry no heading metadata. Before the fix the judge
    # scored them against metadata alone, so the fixed strategy got 0.04.
    expect = {"doc": "air_pressure_system", "heading_contains": "failure modes"}
    chunk = hit("air_pressure_system", text="## Common failure modes\n\nLeaks at...")
    assert relevant(chunk, expect)


def test_heading_must_appear_somewhere():
    expect = {"doc": "air_pressure_system", "heading_contains": "failure modes"}
    assert not relevant(hit("air_pressure_system", "Main components", "Compressor"), expect)
