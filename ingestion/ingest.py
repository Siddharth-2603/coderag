"""Ingestion pipeline (milestone 0 skeleton).

Flow we are building toward:
    load documents -> chunk -> embed -> upsert into the vector store

Milestone 0 just wires the skeleton with clear TODOs so each step can be
filled in and tested independently. Nothing here calls a paid API yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """A piece of a document, with metadata for retrieval + eval."""

    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents(root: Path) -> list[tuple[str, str]]:
    """Return (doc_id, text) pairs.

    TODO(M1): walk the directory, read code + docs files. For code, keep
    enough context (file path, surrounding lines) that chunks stay useful.
    """
    raise NotImplementedError("load_documents: implement in milestone 1")


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    """Split one document into retrievable chunks.

    TODO(M1): start with a simple fixed-size + overlap strategy. Make the
    strategy configurable (M4) so the eval harness can compare strategies.
    """
    raise NotImplementedError("chunk_document: implement in milestone 1")


def embed(chunks: list[Chunk]) -> list[list[float]]:
    """Turn chunk text into embedding vectors.

    TODO(M1): call the embedding source (hosted API or local model). Pick the
    cheapest option that works first — see docs/COST-GUARDRAILS.md.
    """
    raise NotImplementedError("embed: implement in milestone 1")


def upsert(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Write chunks + vectors into the vector store.

    TODO(M1): decide Qdrant vs pgvector, then implement the client call.
    """
    raise NotImplementedError("upsert: implement in milestone 1")


def run(root: Path) -> None:
    docs = load_documents(root)
    all_chunks: list[Chunk] = []
    for doc_id, text in docs:
        all_chunks.extend(chunk_document(doc_id, text))
    vectors = embed(all_chunks)
    upsert(all_chunks, vectors)
    print(f"ingested {len(all_chunks)} chunks from {len(docs)} documents")


if __name__ == "__main__":
    # Placeholder entry point. Milestone 1 wires argument parsing + a real path.
    print("ingestion skeleton — implement load/chunk/embed/upsert in milestone 1")
