"""Pluggable embedding backends.

M1 ships a dependency-free, deterministic HashEmbedder so the whole pipeline
runs end to end with zero cost and zero setup. Later we add a real model
(local via Ollama, or a hosted API) behind the same interface — nothing else
in the pipeline has to change.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Protocol


class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    dim: int

    def embed(self, text: str) -> List[float]:
        ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Not semantically strong — it is a stand-in so we can build and test the
    full retrieval flow before wiring a real model. Same word -> same bucket,
    so exact-term overlap between query and chunk produces real similarity.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 7) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    # Vectors are L2-normalized, so dot product == cosine similarity.
    return sum(x * y for x, y in zip(a, b))
