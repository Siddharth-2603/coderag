"""Pluggable embedding backends.

M1 ships a dependency-free, deterministic HashEmbedder so the whole pipeline
runs end to end with zero cost and zero setup. Later we add a real model
(local via Ollama, or a hosted API) behind the same interface — nothing else
in the pipeline has to change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
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


def load_env(path: Path) -> None:
    """Minimal .env loader (no external dependency).

    Reads KEY=VALUE lines into os.environ if not already set. We keep this tiny
    on purpose — one less package to install on a locked-down machine.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class GeminiEmbedder:
    """Real embeddings via Google's Gemini embedding API.

    Same Embedder interface as HashEmbedder, so ingest/query only change which
    embedder they construct. We call the REST endpoint directly with urllib —
    no SDK to install, and each call is a small HTTPS request (allowed even
    where large model downloads are blocked).
    """

    # gemini-embedding-001 is the current stable embedding model.
    MODEL = "models/gemini-embedding-001"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"

    def __init__(self, dim: int = 3072) -> None:
        # dim is the expected size; actual length comes from the API response.
        self.dim = dim
        self._api_key = os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Put it in .env and call load_env() first."
            )

    def embed(self, text: str) -> List[float]:
        url = self.ENDPOINT.format(model=self.MODEL) + f"?key={self._api_key}"
        payload = {
            "model": self.MODEL,
            "content": {"parts": [{"text": text}]},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Surface the API's error message without leaking the key.
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Gemini API error {e.code}: {detail}") from None

        values = body.get("embedding", {}).get("values")
        if not values:
            raise RuntimeError(f"unexpected Gemini response shape: {body}")
        # Gemini returns normalized-ish vectors; we L2-normalize to be safe so
        # cosine similarity stays a clean dot product (same rule as HashEmbedder).
        return _l2_normalize(values)


def make_embedder(name: str) -> Embedder:
    """Factory so ingest and query build the SAME embedder from one place.

    Using one factory is how we enforce the same-embedder rule in code:
    both scripts call make_embedder(args.embedder), so they cannot drift apart.
    """
    if name == "gemini":
        return GeminiEmbedder()
    if name == "hash":
        return HashEmbedder(dim=256)
    raise ValueError(f"unknown embedder: {name!r} (use 'gemini' or 'hash')")
