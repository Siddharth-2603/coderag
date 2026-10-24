"""Evaluation harness — turns "it feels better" into measured numbers.

For each labelled question we retrieve the top-k chunks and check whether any
of them come from a document marked as relevant (ground truth). We report
recall@k: the fraction of questions where retrieval surfaced a correct doc.

Why doc-level (not chunk-level) ground truth: chunk IDs shift whenever we
re-chunk (M4 changes chunk size/overlap). Documents are stable, so labelling
by source doc keeps the test set valid across chunking experiments.

Usage:
    python3 run_eval.py --embedder gemini --top-k 3
    python3 run_eval.py --embedder hash   --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the ingestion code (same embedder, chunker, store) so eval measures the
# real pipeline, not a reimplementation.
ING = Path(__file__).resolve().parent.parent / "ingestion"
sys.path.insert(0, str(ING))

from chunking import chunk_text          # noqa: E402
from embeddings import load_env, make_embedder  # noqa: E402
from store import Chunk, InMemoryStore   # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "sample-docs"
TESTSET = Path(__file__).resolve().parent / "testset.json"
TEXT_SUFFIXES = {".md", ".txt", ".py", ".go", ".rst"}


def build_index(embedder) -> InMemoryStore:
    """Ingest the sample docs fresh with the given embedder."""
    store = InMemoryStore()
    chunks: list[Chunk] = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            doc_id = path.name
            chunks.extend(chunk_text(doc_id, path.read_text(encoding="utf-8")))
    vectors = [embedder.embed(c.text) for c in chunks]
    store.upsert(chunks, vectors)
    return store


def recall_at_k(embedder_name: str, top_k: int) -> float:
    embedder = make_embedder(embedder_name)
    store = build_index(embedder)
    testset = json.loads(TESTSET.read_text())
    questions = testset["questions"]

    hits = 0
    print(f"\n=== embedder={embedder_name}  top_k={top_k} ===")
    for q in questions:
        query_vec = embedder.embed(q["question"])
        results = store.search(query_vec, top_k)
        retrieved_docs = {chunk.doc_id for chunk, _ in results}
        relevant = set(q["relevant_docs"])
        # "Hit" = at least one retrieved chunk came from a relevant document.
        hit = bool(retrieved_docs & relevant)
        hits += 1 if hit else 0
        mark = "OK " if hit else "MISS"
        print(f"  [{mark}] {q['question'][:55]:55s} -> got {sorted(retrieved_docs)}")

    recall = hits / len(questions)
    print(f"  recall@{top_k} = {hits}/{len(questions)} = {recall:.2f}")
    return recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval eval (recall@k).")
    parser.add_argument("--embedder", default="gemini", choices=["gemini", "hash"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    load_env(Path(__file__).resolve().parent.parent / ".env")
    recall_at_k(args.embedder, args.top_k)


if __name__ == "__main__":
    main()
