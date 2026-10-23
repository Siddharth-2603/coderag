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
