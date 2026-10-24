"""Ingestion pipeline (M1 — working thin slice).

Flow:  load documents -> chunk -> embed -> upsert into the vector store

M1 uses a dependency-free stack (HashEmbedder + InMemoryStore) so the whole
thing runs with zero setup and zero cost. The store is saved to disk so the
query step (and later the Go service) can read it.

Usage:
    python3 ingest.py <docs_dir> [--out index.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from chunking import chunk_text
from embeddings import load_env, make_embedder
from store import Chunk, InMemoryStore

# Files we treat as ingestible text. Expand as needed.
TEXT_SUFFIXES = {".md", ".txt", ".py", ".go", ".rst", ".yaml", ".yml", ".json", ".toml"}


def load_documents(root: Path) -> List[Tuple[str, str]]:
    """Return (doc_id, text) pairs for every text-like file under root."""
    docs: List[Tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            doc_id = str(path.relative_to(root))
            docs.append((doc_id, text))
    return docs


def run(root: Path, out: Path, embedder_name: str) -> None:
    docs = load_documents(root)
    if not docs:
        print(f"no ingestible files found under {root}")
        return

    embedder = make_embedder(embedder_name)
    store = InMemoryStore()

    all_chunks: List[Chunk] = []
    for doc_id, text in docs:
        all_chunks.extend(chunk_text(doc_id, text))

    vectors = [embedder.embed(c.text) for c in all_chunks]
    store.upsert(all_chunks, vectors)
    store.save(out)

    print(f"ingested {len(all_chunks)} chunks from {len(docs)} documents "
          f"using '{embedder_name}' -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG index.")
    parser.add_argument("docs_dir", type=Path, help="directory of documents to ingest")
    parser.add_argument("--out", type=Path, default=Path("index.json"), help="index output path")
    parser.add_argument("--embedder", default="gemini", choices=["gemini", "hash"],
                        help="which embedder to use (must match at query time)")
    args = parser.parse_args()
    load_env(Path(__file__).resolve().parent.parent / ".env")
    run(args.docs_dir, args.out, args.embedder)


if __name__ == "__main__":
    main()
