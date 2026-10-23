"""Vector store abstraction.

M1 ships an in-memory store so the pipeline runs with no external services.
At M1.5 we add a QdrantStore behind the same interface and swap it in without
touching the ingestion or query logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

from embeddings import cosine_similarity


@dataclass
class Chunk:
    """A retrievable piece of a document plus metadata for retrieval + eval."""

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class InMemoryStore:
    """Tiny cosine-similarity index. Good enough to prove the flow."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query_vec: List[float], top_k: int) -> List[Tuple[Chunk, float]]:
        scored = [
            (chunk, cosine_similarity(query_vec, vec))
            for chunk, vec in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: Path) -> None:
        """Persist so the Go service (or a later query run) can read it."""
        payload = {
            "chunks": [asdict(c) for c in self._chunks],
            "vectors": self._vectors,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "InMemoryStore":
        payload = json.loads(path.read_text())
        store = cls()
        store._chunks = [Chunk(**c) for c in payload["chunks"]]
        store._vectors = payload["vectors"]
        return store
