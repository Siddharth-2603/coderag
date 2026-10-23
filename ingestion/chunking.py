"""Chunking strategies.

M1 ships a simple line-window chunker with overlap. At M4 we make the strategy
configurable and let the eval harness compare strategies on real metrics.
"""

from __future__ import annotations

from typing import List

from store import Chunk


def chunk_text(
    doc_id: str,
    text: str,
    window_lines: int = 20,
    overlap_lines: int = 5,
) -> List[Chunk]:
    """Split text into overlapping line windows.

    Overlap keeps context from spilling across chunk boundaries (important for
    code and docs, where an answer may straddle a split).
    """
    if window_lines <= 0:
        raise ValueError("window_lines must be positive")
    if overlap_lines >= window_lines:
        raise ValueError("overlap_lines must be smaller than window_lines")

    lines = text.splitlines()
    if not lines:
        return []

    step = window_lines - overlap_lines
    chunks: List[Chunk] = []
    start = 0
    n = 0
    while start < len(lines):
        window = lines[start : start + window_lines]
        chunk_text_str = "\n".join(window).strip()
        if chunk_text_str:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}:{n}",
                    text=chunk_text_str,
                    metadata={
                        "start_line": start + 1,
                        "end_line": min(start + window_lines, len(lines)),
                    },
                )
            )
            n += 1
        start += step
    return chunks
