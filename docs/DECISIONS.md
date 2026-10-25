# Design Decisions & Alternatives (interview prep)

For each major choice: what we picked, the alternatives, the tradeoff, and the
one-line justification. Aim: answer "why this and not that?" with confidence.

---

## 1. Cosine similarity (vs Euclidean / dot product)
- **Chose:** cosine similarity to compare query and chunk vectors.
- **Alternatives:** Euclidean (L2) distance; raw dot product.
- **Tradeoff:** Euclidean is sensitive to vector *magnitude*; two texts about the
  same topic but different lengths can be far apart in L2 even when their
  direction (meaning) matches. Cosine measures only direction, so it is
  magnitude-invariant — the right fit for text embeddings.
- **Why:** meaning lives in direction, not length, so cosine is the standard for
  semantic text retrieval.

## 2. L2 normalization of vectors
- **Chose:** L2-normalize vectors to unit length.
- **Alternative:** leave vectors un-normalized and compute full cosine each time.
- **Tradeoff:** once vectors are unit length, cosine similarity == dot product,
  so similarity is a single cheap multiply-and-sum. Without it you must divide by
  both norms every comparison.
- **Why:** normalize once up front, then every similarity is a fast dot product.

## 3. Gemini embeddings (vs OpenAI / Cohere / local model)
- **Chose:** Google `gemini-embedding-001`.
- **Alternatives:** OpenAI `text-embedding-3`, Cohere embed, local model (nomic
  via Ollama).
- **Tradeoff:** OpenAI/Cohere are comparable quality but paid and off my chosen
  stack. A local model is free + offline but lower quality and needs a model
  download (which my locked-down environment blocked). Gemini has a usable free
  tier and fits a GCP-leaning direction.
- **Why:** free tier + good quality + aligned with my cloud direction; the code is
  behind an interface so switching providers is a one-class change.

## 4. Pluggable Embedder interface (vs hardcoding one embedder)
- **Chose:** an `Embedder` interface with `HashEmbedder` and `GeminiEmbedder`.
- **Alternative:** call one embedding provider directly throughout.
- **Tradeoff:** the interface is a little more upfront structure, but it let me
  start with a zero-cost hash embedder to validate the pipeline, then swap in a
  real model with no pipeline changes — and A/B them in the eval.
- **Why:** abstraction made the swap trivial and enabled measured comparison.

## 5. Hash embedder as the M1 placeholder (vs real model from day one)
- **Chose:** build a deterministic bag-of-words hash embedder first.
- **Alternative:** wire the real embedding API immediately.
- **Tradeoff:** the hash embedder is semantically weak (exact-word match only),
  but it is free, offline, and instant — perfect for proving chunk/store/query
  plumbing before spending on an API or fighting a model download.
- **Why:** validate the plumbing at zero cost/risk, then upgrade the one swappable piece.

## 6. In-memory brute-force store (vs Qdrant / pgvector)
- **Chose:** an in-memory list with brute-force cosine search.
- **Alternatives:** Qdrant or pgvector with an ANN (e.g. HNSW) index.
- **Tradeoff:** brute-force is O(N) per query — fine for a small corpus and it is
  exact and dependency-free. A real vector DB uses approximate nearest-neighbor
  to search millions in milliseconds, at the cost of running a service and a
  little accuracy.
- **Why:** simplest exact approach for a small corpus; the store is behind an
  interface so a Qdrant backend swaps in unchanged at scale.

## 7. Doc-level ground truth in eval (vs chunk-level)
- **Chose:** label each test question with the source *document* that answers it.
- **Alternative:** label with specific chunk IDs.
- **Tradeoff:** chunk IDs shift whenever chunking changes (size/overlap), which
  would silently invalidate the test set. Documents are stable across re-chunking.
- **Why:** stable ground truth lets me run chunking experiments without re-labelling.

## 8. recall@k as the metric (and evaluating at top_k=1)
- **Chose:** recall@k; reported at a strict top_k=1 to discriminate.
- **Alternatives:** precision, MRR, nDCG; or only reporting a loose recall@3.
- **Tradeoff:** recall@k is simple and answers "did we surface a right doc?"
  recall@3 over a tiny corpus was too easy (even random scores high), so top_k=1
  exposed the real quality gap. MRR/nDCG add rank-position nuance but are more to
  explain.
- **Why:** recall@k is clear and sufficient; choosing a strict k made the metric
  actually discriminate (a senior point: the metric must be hard enough to matter).

## 9. Hybrid search via RRF — and the decision NOT to keep it
- **Chose:** implemented hybrid (semantic + keyword) with Reciprocal Rank Fusion,
  measured it, and found it regressed (0.83 vs pure semantic 1.00); kept semantic.
- **Alternatives:** weighted RRF; a real lexical ranker (BM25) instead of the hash
  embedder; score-based fusion.
- **Tradeoff:** RRF fuses by rank position so it needs no score-scale
  reconciliation, but equal-weight fusion of a strong ranker with a weak one let
  the weak signal drag results down. Score-based fusion would need the two
  incomparable scales reconciled.
- **Why:** the eval showed naive hybrid hurt because the keyword ranker was too
  weak; I let data override the assumption. Fixes (weighted RRF / BM25) are known
  next steps.

## 10. Polyglot: Python ingestion + Go serving (vs one language)
- **Chose:** Python builds the index; Go serves queries (embed + search + generate).
- **Alternative:** do everything in Python, or everything in Go.
- **Tradeoff:** two languages means two codebases, but it matches real production
  (ML/ingestion in Python, high-performance serving in Go) and plays to my Go
  backend strength.
- **Why:** mirrors how companies actually split this, and showcases Go serving depth.

## 11. Grounded prompt that refuses when unsure (vs free generation)
- **Chose:** a prompt instructing the model to answer only from context and admit
  when the answer isn't present; return sources.
- **Alternative:** pass the question straight to the LLM with no grounding.
- **Tradeoff:** grounding can make the model decline to answer when context is
  thin, but it sharply reduces hallucination and makes answers verifiable.
- **Why:** correctness + verifiability matter more than always producing an answer;
  verified it refuses out-of-corpus questions instead of inventing facts.

## 12. Retry with backoff on the generation call (vs single attempt)
- **Chose:** retry transient failures (429, 5xx, client timeouts) with backoff.
- **Alternative:** single request, fail on any error.
- **Tradeoff:** retries add latency on a bad path, but LLM APIs routinely return
  short-lived 503/timeout under load; a single attempt makes the service flaky.
- **Why:** a serving layer must absorb transient upstream failures to be reliable.

---

### Decisions still open / honest "next steps" to mention
- Swap in a real vector DB (Qdrant/HNSW) for scale.
- Weighted RRF or BM25 to make hybrid actually help.
- Streaming (SSE) the generated answer; semantic caching; metrics; deployment.
- An agent layer (tool-calling, multi-step) on top of generation.
