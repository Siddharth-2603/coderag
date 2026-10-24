# Concept Log

Running one-line notes on every concept as we build. Newest at the bottom.
Goal: be able to explain each of these in an interview without looking.

## Embeddings & similarity
- Embedding = turn text into a vector (list of numbers) so we can compare meaning, not exact words.
- A trained model is what *puts* the meaning into the vector (similar meaning -> nearby vectors).
- Cosine similarity = measures the angle between two vectors; same direction = 1.0, unrelated ≈ 0.
- L2 normalize = scale a vector to length 1; then cosine similarity simplifies to a plain dot product.
- Similarity ≠ correctness: a chunk can be topically similar but not contain the answer (e.g. "define a car" vs "how to start a car").
- Because similarity ≠ correctness, we MUST measure retrieval quality with an eval harness (recall@k) — not vibes.

## Hash embedder (our M1 placeholder)
- HashEmbedder buckets each word into a slot by hashing; only matches EXACT shared words.
- Weakness: cannot match synonyms/paraphrases ("login" vs "authenticate") — no real meaning.
- It exists to prove the pipeline plumbing at zero cost; a real model replaces it at M1.5.

## Chunking
- Chunk = split a document into smaller pieces, embed each separately, so retrieval pulls the specific relevant section.
- One vector for a whole doc blurs everything into a muddy average — chunking keeps sections distinct.
- Overlap = repeat the last few lines of one chunk at the start of the next.
- Why overlap: stops an answer from being split across a boundary so NO single chunk holds it whole.
- Zero overlap risk: miss answers that straddle a chunk boundary.
- Tradeoff: more overlap = fewer boundary losses BUT more duplicate text, more chunks, higher cost/storage.
- Chunk size & overlap are knobs — tune them later with eval numbers (M4), don't guess.

## Retrieval metrics (preview, built in M3)
- recall@k = of all test questions, how often the correct chunk appears in the top-k results.
- answer relevance = once generation is added, does the final answer actually address the question.

## Vector store
- Vector store = where chunk vectors live + how we search them.
- upsert = "update or insert": save chunks+vectors (update if exists, else insert).
- search = compare a query vector to stored vectors by cosine, sort, return top-k. This IS retrieval.
- Our InMemoryStore = two lists + brute-force: it scores EVERY stored vector each search.
- Brute-force is exact (guaranteed true top-k) but O(N) per query — fine for small N, too slow at millions.
- Real vector DBs use ANN (Approximate Nearest Neighbor): prebuilt index skips most vectors, near-exact but far faster.
- ANN tradeoff: trade a tiny bit of accuracy for huge speed (millions in ms).
- Common ANN technique = HNSW (Hierarchical Navigable Small World graph). Qdrant uses this.
- Library analogy: brute-force = read every spine; ANN = catalog routes you to the right shelf.
- Same upsert/search interface means InMemoryStore -> QdrantStore swaps with no pipeline changes.

## RAG flow (ingestion vs query)
- RAG has two phases at different times: ingestion (offline, batch) and query (online, per-request).
- Ingestion (ingest.py): load -> chunk -> embed -> upsert -> save index. Expensive; run when docs change.
- Query (query.py): load index -> embed the question -> cosine top-k search -> return chunks. Cheap; per request.
- Query does NOT re-chunk/re-embed docs; it only embeds the single incoming question.
- RULE: ingestion and query MUST use the same embedder, else vectors live in different spaces and cosine is meaningless.
- When swapping to a real model (M1.5), change the embedder in BOTH ingest.py and query.py together.

## Real embeddings (M1.5 — Gemini)
- Real embedding model (Gemini gemini-embedding-001) captures MEANING; matches synonyms/paraphrases, not just exact words.
- Proof: query "how does a client prove its identity?" -> Gemini ranks auth.md #1; hash embedder ranks it WRONG (caching.md #1).
- Same Embedder interface means swapping HashEmbedder -> GeminiEmbedder changed only which class we build, not the pipeline.
- make_embedder() factory enforces the same-embedder rule: ingest and query both build from one place, cannot drift.
- Gemini vectors are 3072-dim (vs hash 256); different dim = different space = must re-ingest, can't mix indexes.
- Called the REST API with urllib (no SDK): one small HTTPS POST per text; allowed even where big downloads are blocked.
- Gemini scores sit in a compressed range (~0.5-0.7); what matters is the RELATIVE gap (related > unrelated), not absolute value.
- API key kept in gitignored .env, loaded via a tiny load_env(); never committed, never printed.
