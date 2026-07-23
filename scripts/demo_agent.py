"""One-command demo: runs the full agent on a flagged vehicle.

Usage (from the repo root, after `pip install -r requirements.txt`):
    python scripts/demo_agent.py
    python scripts/demo_agent.py "Is vehicle 100 at risk? What should we check?"

Requires ANTHROPIC_API_KEY in .env (repo root). The first run downloads
the MiniLM embedding model (~90 MB) and builds the Chroma index.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

# macOS: torch and xgboost each bundle their own OpenMP runtime; loading
# both in one process can segfault. These settings must be in place before
# either library is imported.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fleet_copilot.rag import CHROMA_DIR, build_index  # noqa: E402


def ensure_index() -> None:
    """Build the index unless the collection actually exists and is populated.

    Checking the directory alone is not enough — an aborted earlier run can
    leave an empty chroma_db/ behind.
    """
    try:
        import chromadb

        col = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
            "kb_sections")
        if col.count() > 0:
            return
    except Exception:
        pass
    print("building knowledge index...")
    name, n = build_index("sections")
    print(f"  indexed {n} chunks into '{name}'")


def main() -> None:
    question = (sys.argv[1] if len(sys.argv) > 1 else
                "Why is vehicle 42 flagged, and what should the workshop "
                "check first?")
    ensure_index()
    from fleet_copilot.agent import run_agent

    print(f"\nQ: {question}\n")
    result = run_agent(question)
    print("tool calls:")
    for step in result["trace"]:
        print(f"  -> {step['tool']}({json.dumps(step['input'])})")
    print("\n" + result["answer"])


if __name__ == "__main__":
    main()
