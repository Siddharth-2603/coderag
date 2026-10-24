"""Hybrid retrieval via Reciprocal Rank Fusion (RRF).

We run two independent rankers over the same chunks:
  - a SEMANTIC ranker (Gemini embeddings; good at meaning)
  - a KEYWORD ranker (hash/bag-of-words; good at exact terms)

Each produces an ordered list. RRF fuses them using ONLY rank position, never
the raw scores (cosine vs term-frequency are on different scales and are not
comparable). A chunk ranked high in BOTH lists accumulates the most fused score.

    rrf_score(item) = sum over rankers of  1 / (k + rank)

where rank is 1-based position in that ranker's list and k is a damping
constant (60 is the common default from the original RRF paper).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

ING = Path(__file__).resolve().parent.parent / "ingestion"
sys.path.insert(0, str(ING))

from store import Chunk, InMemoryStore  # noqa: E402

RRF_K = 60


def _ranked_chunk_ids(store: InMemoryStore, query_vec: List[float], depth: int) -> List[str]:
    """Return chunk_ids ordered best-first from one ranker (top `depth`)."""
    results = store.search(query_vec, depth)
    return [chunk.chunk_id for chunk, _ in results]


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """Fuse several ranked lists of chunk_ids into one, by rank position only."""
    scores: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, chunk_id in enumerate(ranking, start=1):  # rank is 1-based
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused


def hybrid_search(
    semantic_store: InMemoryStore,
    keyword_store: InMemoryStore,
    semantic_qvec: List[float],
    keyword_qvec: List[float],
    top_k: int,
    fuse_depth: int = 10,
) -> List[str]:
    """Return the top_k chunk_ids after fusing semantic + keyword rankings.

    fuse_depth = how many results to pull from each ranker before fusing.
    """
    sem_ranking = _ranked_chunk_ids(semantic_store, semantic_qvec, fuse_depth)
    kw_ranking = _ranked_chunk_ids(keyword_store, keyword_qvec, fuse_depth)
    fused = reciprocal_rank_fusion([sem_ranking, kw_ranking])
    return [chunk_id for chunk_id, _ in fused[:top_k]]
