"""Chunker regression tests. Pure Python: no embedding model, no Chroma."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fleet_copilot.rag import chunk_fixed, chunk_sections, load_kb  # noqa: E402


def test_fixed_chunk_count_matches_text_length():
    # 2,000 chars at size 800 / overlap 150 needs three chunks. The tail-loop bug
    # produced about 150 extra near-duplicate fragments after the last one.
    text = "word " * 400
    chunks = chunk_fixed(text, "doc")
    assert 2 <= len(chunks) <= 4


def test_fixed_chunks_cover_the_whole_text():
    text = " ".join(f"w{i}" for i in range(600))
    chunks = chunk_fixed(text, "doc")
    assert chunks[0].text.startswith("w0 ")
    assert chunks[-1].text.endswith("w599")


def test_fixed_chunks_have_no_tiny_tail_fragments():
    text = "word " * 400
    assert all(len(c.text) > 150 for c in chunk_fixed(text, "doc"))


def test_short_text_is_one_chunk():
    assert len(chunk_fixed("a short document", "doc")) == 1


def test_sections_split_on_headings():
    text = "# A\n\nalpha body\n\n## B\n\nbeta body\n"
    chunks = chunk_sections(text, "doc")
    assert [c.heading for c in chunks] == ["A", "B"]
    assert chunks[1].text.startswith("B\n\nbeta")


def test_knowledge_base_chunk_counts_are_sane():
    kb_chars = sum(len(c.text) for c in load_kb("sections"))
    fixed = load_kb("fixed")
    # Fixed chunks of up to 800 chars with 150 overlap: the count is bounded by
    # the corpus size, not by the number of characters in each document's tail.
    assert len(fixed) <= kb_chars // 500 + 8
