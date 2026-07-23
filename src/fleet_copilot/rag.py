"""RAG pipeline: chunking, embedding, retrieval, grounded answering.

Design choices (documented for the eval writeup):
- Embeddings: sentence-transformers all-MiniLM-L6-v2, local and free, so
  the retrieval layer and its evals run without any API key.
- Vector store: Chroma (persistent, single directory, no server).
- Answering: Anthropic API. The answer step is the only part that needs
  a key; retrieval evals are key-free.
- Two chunking strategies are implemented so the eval harness can compare
  them: fixed-size (with overlap) and heading-aware markdown sections.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", ROOT / "chroma_db"))

# Override with a local model directory (e.g. an offline mirror) via env.
EMBED_MODEL = os.environ.get("EMBED_MODEL_PATH", "all-MiniLM-L6-v2")


# ---------------------------------------------------------------- chunking

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    heading: str | None = None


def chunk_fixed(text: str, doc_id: str, size: int = 800,
                overlap: int = 150) -> list[Chunk]:
    """Fixed-size character chunks with overlap, split on whitespace."""
    chunks, start, i = [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # don't cut words
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(doc_id, f"{doc_id}::fixed{i}", piece))
            i += 1
        start = max(end - overlap, start + 1)
    return chunks


def chunk_sections(text: str, doc_id: str, max_len: int = 1600) -> list[Chunk]:
    """Markdown-heading-aware chunks; long sections fall back to fixed."""
    parts = re.split(r"(?m)^(#{1,4} .*)$", text)
    chunks: list[Chunk] = []
    heading = None
    i = 0
    for part in parts:
        if re.match(r"^#{1,4} ", part or ""):
            heading = part.lstrip("# ").strip()
            continue
        body = (part or "").strip()
        if not body:
            continue
        section = f"{heading}\n\n{body}" if heading else body
        if len(section) <= max_len:
            chunks.append(Chunk(doc_id, f"{doc_id}::sec{i}", section, heading))
            i += 1
        else:
            for sub in chunk_fixed(section, doc_id):
                sub.chunk_id = f"{doc_id}::sec{i}"
                sub.heading = heading
                chunks.append(sub)
                i += 1
    return chunks


def load_kb(strategy: str = "sections") -> list[Chunk]:
    chunker = chunk_sections if strategy == "sections" else chunk_fixed
    chunks: list[Chunk] = []
    for path in sorted(KB_DIR.glob("*.md")):
        chunks.extend(chunker(path.read_text(), path.stem))
    return chunks


# ---------------------------------------------------------------- indexing

def build_index(strategy: str = "sections", collection_name: str | None = None):
    """Embed all KB chunks into a persistent Chroma collection."""
    import chromadb
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    name = collection_name or f"kb_{strategy}"
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(
        name, embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL))
    chunks = load_kb(strategy)
    col.add(ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"doc": c.doc_id, "heading": c.heading or ""}
                       for c in chunks])
    return name, len(chunks)


def retrieve(query: str, strategy: str = "sections", k: int = 4) -> list[dict]:
    import chromadb
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(
        f"kb_{strategy}", embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL))
    res = col.query(query_texts=[query], n_results=k)
    return [{"chunk_id": i, "text": d, "meta": m, "distance": dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])]


# ---------------------------------------------------------------- answering

ANSWER_SYSTEM = """You are a fleet maintenance assistant. Answer using ONLY
the provided context passages. Cite passages inline as [chunk_id]. If the
context does not contain the answer, say so plainly — do not guess."""


def answer(query: str, strategy: str = "sections", k: int = 4,
           model: str = "claude-haiku-4-5-20251001") -> dict:
    import anthropic

    hits = retrieve(query, strategy, k)
    context = "\n\n".join(f"[{h['chunk_id']}]\n{h['text']}" for h in hits)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=model, max_tokens=600, system=ANSWER_SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nQuestion: {query}"}])
    return {"answer": msg.content[0].text, "retrieved": hits}
