"""Evaluation harness — turns "it feels better" into measured numbers.

For each labelled question we retrieve the top-k chunks and check whether any
of them come from a document marked as relevant (ground truth). We report
recall@k: the fraction of questions where retrieval surfaced a correct doc.

Modes:
  gemini  - semantic retrieval only
  hash    - lexical/keyword retrieval only (baseline)
  hybrid  - fuse semantic + keyword with Reciprocal Rank Fusion (RRF)

Why doc-level (not chunk-level) ground truth: chunk IDs shift whenever we
re-chunk. Documents are stable, so labelling by source doc keeps the test set
valid across chunking experiments.

Usage:
    python3 run_eval.py --mode gemini --top-k 1
    python3 run_eval.py --mode hash   --top-k 1
    python3 run_eval.py --mode hybrid --top-k 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Reuse the ingestion code (same embedder, chunker, store) so eval measures the
# real pipeline, not a reimplementation.
ING = Path(__file__).resolve().parent.parent / "ingestion"
sys.path.insert(0, str(ING))

from chunking import chunk_text          # noqa: E402
from embeddings import load_env, make_embedder  # noqa: E402
from store import Chunk, InMemoryStore   # noqa: E402
from hybrid import hybrid_search         # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "sample-docs"
TESTSET = Path(__file__).resolve().parent / "testset.json"
TEXT_SUFFIXES = {".md", ".txt", ".py", ".go", ".rst"}


def load_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            chunks.extend(chunk_text(path.name, path.read_text(encoding="utf-8")))
    return chunks


def build_index(embedder, chunks: List[Chunk]) -> InMemoryStore:
    store = InMemoryStore()
    vectors = [embedder.embed(c.text) for c in chunks]
    store.upsert(chunks, vectors)
    return store


def evaluate(mode: str, top_k: int) -> float:
    chunks = load_chunks()
    chunk_to_doc: Dict[str, str] = {c.chunk_id: c.doc_id for c in chunks}
    testset = json.loads(TESTSET.read_text())
    questions = testset["questions"]

    # Build whatever indexes this mode needs.
    gemini = make_embedder("gemini")
    hash_ = make_embedder("hash")
    if mode == "gemini":
        sem_store = build_index(gemini, chunks)
    elif mode == "hash":
        kw_store = build_index(hash_, chunks)
    else:  # hybrid needs both
        sem_store = build_index(gemini, chunks)
        kw_store = build_index(hash_, chunks)

    hits = 0
    print(f"\n=== mode={mode}  top_k={top_k} ===")
    for q in questions:
        if mode == "gemini":
            results = sem_store.search(gemini.embed(q["question"]), top_k)
            retrieved_docs = {c.doc_id for c, _ in results}
        elif mode == "hash":
            results = kw_store.search(hash_.embed(q["question"]), top_k)
            retrieved_docs = {c.doc_id for c, _ in results}
        else:  # hybrid
            chunk_ids = hybrid_search(
                sem_store, kw_store,
                gemini.embed(q["question"]), hash_.embed(q["question"]),
                top_k,
            )
            retrieved_docs = {chunk_to_doc[cid] for cid in chunk_ids}

        relevant = set(q["relevant_docs"])
        hit = bool(retrieved_docs & relevant)
        hits += 1 if hit else 0
        mark = "OK " if hit else "MISS"
        print(f"  [{mark}] {q['question'][:52]:52s} -> {sorted(retrieved_docs)}")

    recall = hits / len(questions)
    print(f"  recall@{top_k} = {hits}/{len(questions)} = {recall:.2f}")
    return recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval eval (recall@k).")
    parser.add_argument("--mode", default="gemini", choices=["gemini", "hash", "hybrid"])
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    load_env(Path(__file__).resolve().parent.parent / ".env")
    evaluate(args.mode, args.top_k)


if __name__ == "__main__":
    main()
