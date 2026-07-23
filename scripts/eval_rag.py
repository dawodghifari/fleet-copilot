"""Retrieval evaluation harness.

Compares chunking strategies (fixed vs. sections) on a labeled question
set. Metrics:
- hit@k: fraction of questions where a chunk from the expected document
  (and heading, if specified) appears in the top-k results.
- MRR: mean reciprocal rank of the first relevant chunk.

Runs fully locally (sentence-transformers embeddings) — no API key.
Writes reports/rag_eval.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fleet_copilot.rag import build_index, retrieve  # noqa: E402

QUESTIONS = ROOT / "evals" / "retrieval_questions.json"
K = 4


def relevant(hit: dict, expect: dict) -> bool:
    """A hit is relevant if it comes from the expected document and — when a
    heading is specified — the heading matches OR the heading's text appears
    in the chunk body. The text fallback keeps the judgment fair for the
    fixed-size strategy, whose chunks carry no heading metadata."""
    if hit["meta"]["doc"] != expect["doc"]:
        return False
    needle = expect.get("heading_contains")
    if not needle:
        return True
    needle = needle.lower()
    return (needle in hit["meta"]["heading"].lower()
            or needle in hit["text"].lower())


def evaluate(strategy: str, questions: list[dict]) -> dict:
    hits_at_k, rr_sum, failures = 0, 0.0, []
    for q in questions:
        results = retrieve(q["question"], strategy=strategy, k=K)
        ranks = [i for i, h in enumerate(results, 1) if relevant(h, q["expect"])]
        if ranks:
            hits_at_k += 1
            rr_sum += 1.0 / ranks[0]
        else:
            failures.append(q["question"])
    n = len(questions)
    return {"strategy": strategy, "n": n, f"hit@{K}": round(hits_at_k / n, 3),
            "mrr": round(rr_sum / n, 3), "failures": failures}


def main() -> None:
    questions = json.loads(QUESTIONS.read_text())
    lines = ["# Retrieval evaluation\n",
             f"{len(questions)} labeled questions; k={K}; embeddings: "
             "all-MiniLM-L6-v2 (local).\n"]
    for strategy in ("fixed", "sections"):
        name, n_chunks = build_index(strategy)
        res = evaluate(strategy, questions)
        lines.append(f"## {strategy} ({n_chunks} chunks)\n")
        lines.append(f"hit@{K}: **{res[f'hit@{K}']}** — MRR: **{res['mrr']}**\n")
        if res["failures"]:
            lines.append("Missed questions:\n")
            lines.extend(f"- {q}" for q in res["failures"])
        lines.append("")
        print(res["strategy"], {k: v for k, v in res.items() if k != "failures"})
    (ROOT / "reports" / "rag_eval.md").write_text("\n".join(lines))
    print("wrote reports/rag_eval.md")


if __name__ == "__main__":
    main()
