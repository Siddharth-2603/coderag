"""Ingest Confluence pages into the RAG index.

Flow:  fetch pages -> normalize -> chunk (existing pipeline) -> embed -> save index

Reuses the EXISTING ingestion modules (chunking, embeddings, store) — the
connector only adds the "get pages from Confluence" + "clean them up" parts.

Usage:
    # offline demo, no real Confluence needed:
    python3 ingest_confluence.py --mock --out confluence_index.json
    # real:
    python3 ingest_confluence.py --out confluence_index.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingestion"))

from embeddings import load_env, make_embedder  # noqa: E402
from chunking import chunk_text                 # noqa: E402
from store import InMemoryStore                 # noqa: E402

from confluence import (                          # noqa: E402
    ConfluenceClient, MockConfluenceClient, normalize_page, PageSource,
)


def ingest(source: PageSource, out: Path, embedder_name: str = "gemini") -> None:
    pages = source.fetch_pages()
    if not pages:
        print("no pages fetched")
        return

    embedder = make_embedder(embedder_name)
    store = InMemoryStore()
    all_chunks = []
    for page in pages:
        doc = normalize_page(page)
        # Reuse the existing chunker; carry the page metadata onto each chunk.
        chunks = chunk_text(doc.doc_id, doc.text)
        for c in chunks:
            c.metadata.update(doc.metadata)  # keep source/title/url for citations
        all_chunks.extend(chunks)

    vectors = [embedder.embed(c.text) for c in all_chunks]
    store.upsert(all_chunks, vectors)
    store.save(out)
    print(f"ingested {len(all_chunks)} chunks from {len(pages)} Confluence pages -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Confluence pages into the RAG index.")
    parser.add_argument("--mock", action="store_true", help="use canned pages (no real Confluence)")
    parser.add_argument("--out", type=Path, default=Path("confluence_index.json"))
    parser.add_argument("--embedder", default="gemini", choices=["gemini", "hash"])
    args = parser.parse_args()
    load_env(ROOT / ".env")

    source = MockConfluenceClient() if args.mock else ConfluenceClient()
    ingest(source, args.out, args.embedder)


if __name__ == "__main__":
    main()
