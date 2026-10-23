"""Query the ingested index (M1 — working thin slice).

Loads the saved index, embeds the question with the same embedder used for
ingestion, and prints the top-k most similar chunks.

Usage:
    python3 query.py "your question here" [--index index.json] [--top-k 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from embeddings import HashEmbedder
from store import InMemoryStore


def run(question: str, index_path: Path, top_k: int) -> None:
    if not index_path.exists():
        print(f"index not found: {index_path} (run ingest.py first)")
        return

    store = InMemoryStore.load(index_path)
    embedder = HashEmbedder(dim=256)
    query_vec = embedder.embed(question)

    results = store.search(query_vec, top_k)
    print(f"query: {question!r}  (index has {len(store)} chunks)\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        loc = f"{chunk.doc_id} (lines {chunk.metadata.get('start_line')}-{chunk.metadata.get('end_line')})"
        preview = chunk.text.replace("\n", " ")[:120]
        print(f"#{rank}  score={score:.3f}  {loc}")
        print(f"     {preview}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--index", type=Path, default=Path("index.json"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(args.question, args.index, args.top_k)


if __name__ == "__main__":
    main()
