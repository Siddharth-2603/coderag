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

## Hash embedder internals (how the numbers appear)
- A vector = a fixed-length list of numbers; dim buckets all starting at 0.
- For each word: hash it (md5 -> big number), bucket = hash % dim, pick +1/-1 from a hash bit, add to that bucket.
- So the vector is basically a word-count tally scattered across dim buckets.
- Same word -> same hash -> same bucket every time (deterministic); repeating a word increases that bucket's value.
- Shared words -> same buckets hit -> high cosine; different words (synonyms) -> different buckets -> ~0 similarity (the weakness).

## Dimension (what it's for)
- Dimension = number of buckets/numbers in the vector (hash: 256, Gemini: 3072).
- More dimensions = more room to keep distinct things apart; fewer forced overlaps.
- Hash collision = different inputs forced into the same bucket (happens a lot when dim is tiny).
- Too few dims -> constant collisions -> unrelated texts look similar -> noisy retrieval.
- For real models, each dimension encodes some learned feature of meaning, so more dims = more nuance.
- Tradeoff: more dims = richer representation BUT more storage + slower comparison (compare all dims each search).

## Eval harness (M3 — the differentiator)
- Eval harness = replace "it feels better" with measured numbers over a labelled test set.
- Needs 3 things: ground-truth test set, a metric, a runner.
- Ground truth = questions each labelled with which doc(s) count as a correct retrieval.
- Label by DOC not chunk id: chunk ids shift when we re-chunk (M4); docs are stable.
- recall@k = fraction of questions where a relevant doc appears in the top-k retrieved.
- "Hit" = at least one retrieved chunk came from a relevant doc (set intersection).
- Measured result: hash recall@1 = 0.50, gemini recall@1 = 1.00 (semantic wins big).
- Metric design matters: recall@3 over 5 docs was too easy (random scores high); top_k=1 exposed the real gap.
- Write questions worded DIFFERENTLY from the docs, so eval tests meaning not word overlap.
- Senior signal: critique your own metric (is it hard enough to discriminate?), not just report it.

## Hybrid search + RRF (M4)
- Semantic embeddings capture meaning but can miss exact terms (error codes, headers, identifiers).
- Keyword/lexical search nails exact tokens but misses meaning. Hybrid = run both, fuse the rankings.
- RRF (Reciprocal Rank Fusion) fuses ranked lists using ONLY rank position: score = sum of 1/(k+rank), k=60.
- RRF ignores raw scores (cosine vs term-frequency are different scales); it trusts POSITION, not value.
- A doc ranked high in BOTH lists accumulates the most fused score and rises to the top.
- MEASURED RESULT (surprise): hybrid recall@1 = 0.83 was WORSE than pure gemini 1.00.
- Cause: our keyword ranker (hash embedder) is weak (0.58 alone); equal-weight fusion let noise drag good results down.
- LESSON: hybrid is only as good as BOTH rankers; fusing a strong + a weak signal can hurt. Don't assume fancy = better.
- Fixes: weight the fusion toward the strong ranker, or use a real lexical ranker (BM25). Or document the finding.
- Interview gold: "I measured, got a counterintuitive regression, and explained the cause" > "I added hybrid search".

## RRF code + the k constant
- scores.get(chunk_id, 0.0) + contribution = ACCUMULATE: a chunk in both lists adds both contributions (don't overwrite).
- enumerate(ranking, start=1) = 1-based rank; the formula 1/(k+rank) expects positions starting at 1.
- sorted(..., key=lambda kv: kv[1], reverse=True) = order chunks by fused score, highest first.
- k is a damping constant: small k -> rank 1 dominates hugely (1/2 vs 1/3); large k -> positions nearly equal weight.
- Small k = trust the very top of each list; large k = reward AGREEMENT across lists over raw top position.
- Why 60: it's the canonical default from the original 2009 RRF paper; most libraries ship 60.
- How to choose k for your own system: it's a hyperparameter — tune it by MEASURING recall@k on your data.
- Recurring theme: chunk size, overlap, top_k, embedder, k — all knobs resolved by the eval harness, never by guessing.

## Go serving layer (connected to the index)
- Polyglot split: Python ingests + builds index.json; Go loads it and serves retrievals. Mirrors real production.
- Go struct JSON tags must EXACTLY match the Python-written JSON keys (doc_id, chunk_id, text, metadata, vectors).
- Query embedding must happen at request time (you can't precompute an unknown question) -> Go calls Gemini REST too.
- Same-embedder rule across languages: Go must use the SAME model (gemini-embedding-001) as Python ingestion.
- Go loads chunks+vectors into memory once at startup; each /query embeds the question then cosine-searches.
- Observed: ~0.6-1.6s per query, almost all of it the Gemini API call; cosine search itself is sub-millisecond.
- That latency is the motivation for a future semantic cache (avoid re-embedding/re-answering repeat queries).
- Ops gotcha: a stale server can hold the port ("address already in use"); find+kill with lsof -ti:PORT.

## Generation (M2 — the G in RAG)
- Generation = feed retrieved chunks to an LLM as CONTEXT and have it write a grounded answer.
- Grounding = instruct the model to answer ONLY from the provided context and admit when it can't.
- Why grounding: reduces hallucination + lets the model answer about YOUR data (not its training) + enables citing sources.
- Return answer AND source chunks so the answer is verifiable (you can see which docs it came from).
- Verified: in-corpus question -> cited answer; out-of-corpus ("refund policy") -> "I don't know based on the provided documents" (no hallucination).
- Embedding model and generation model are DIFFERENT endpoints/models (gemini-embedding-001 vs a flash generation model).
- "flash-lite" generation models were fast/stable (~2s); full "flash" models were timing out / 503 under load.
- Resilience: retry with backoff on transient failures — 429, 5xx, AND client timeouts (status==0). Fail fast on other 4xx.
- Per-attempt timeout should be short enough to fail fast and retry, not block for a minute.

## Polyglot architecture (what it means)
- Polyglot = a system built with more than one programming language (many "tongues").
- This system: Python does ingestion (chunk, embed, write index.json); Go does serving (load, embed query, search, generate).
- The two don't call each other — they share one file, index.json (Python writes it, Go reads it). JSON is the contract.
- That's why Go structs need json tags matching Python's keys exactly (doc_id, chunk_id, ...).
- Why split: Python has the ML ecosystem (ingestion); Go is fast + concurrent for serving. Mirrors real production.
- Precise wording: the SYSTEM is polyglot; the Go service itself is just Go (one half of the polyglot system).
- Tradeoff: two codebases + a defined contract; worth it for right-tool-per-job and showcasing Go serving depth.

## Agents (step 1 — concept + tool abstraction)
- Basic RAG = fixed flow (always retrieve -> generate). Agent = the LLM decides which steps/tools to use.
- Agent core loop: think -> act (call a tool) -> observe (tool result) -> repeat, until the LLM gives a final answer.
- The LLM never runs code; it only ASKS to call a tool by name with args. Our code runs the tool and feeds the result back.
- Why beyond basic RAG: agent can skip retrieval when pointless (e.g. "2+2"), retrieve multiple times, or use other tools.
- Why Go suits agents: goroutines + context give easy concurrency (parallel tools) and timeouts (don't let a slow tool hang).
- Tool = {Name (what LLM calls it), Description (tells LLM WHEN to use it — wording matters), Params, Run (the Go func)}.
- Run is a function stored as a value: func(args map[string]any) (string, error) — takes LLM-supplied args, returns text result.
- Toolbox = registry of tools keyed by name, for lookup when the LLM requests one.
- Go rule noticed: unused EXPORTED types/funcs are fine; only unused local vars/imports are compile errors.

## Gateway (and why Go for it)
- Gateway = the single entry point in FRONT of the system; handles everything AROUND the core work.
- It does: receive requests, auth, rate-limit, stream responses, dispatch to the agent brain. NOT the AI reasoning.
- Analogy: front desk of a building — checks IDs, directs traffic; doesn't do the work inside the offices.
- Why Go for the gateway: lightweight goroutines (tens of thousands of concurrent connections cheaply), great streaming,
  low per-request overhead (compiled), first-class net/HTTP/WebSocket + context for timeouts/cancellation.
- Gateway is CONNECTION-bound (lots of concurrent I/O) -> Go's strength. Brain is LLM-LATENCY-bound -> Python's ecosystem wins.
- Honest caveat: at a learning-project scale you won't feel Go's advantage; use it because it's the production-correct
  pattern (defensible in interviews) and it showcases backend depth. Either language would perform fine at small scale.

## venv + Python version
- venv = an isolated private copy of Python+packages for ONE project, so deps don't collide or pollute system Python.
- Create: python3.11 -m venv .venv ; activate: source .venv/bin/activate ; install; deactivate when done.
- The .venv folder IS the env (large, machine-specific) -> gitignore it; recreate via pip install -r requirements.txt.
- Upgraded agent venv from 3.9 -> 3.11: clears Google-library "3.9 end-of-life" FutureWarnings. 3.11 is fully supported.

## Agent tools in LangChain (step 2)
- A LangChain tool = a normal Python function + the @tool decorator.
- The decorator auto-generates the function-calling declaration from: function name, docstring, typed params.
  name -> tool name ; docstring -> description (LLM reads this to choose) ; params -> args ; body -> what runs.
- Confirms the Go Tool struct fields were language-agnostic concepts; Python just expresses them natively.
- Built two tools so the agent has a real CHOICE: search_docs (retrieval) and calculator (math -> lets it skip retrieval).
- search_docs reuses the existing ingestion modules (same embedder/store as the eval) — reuse proven code, don't rewrite.
- Lazy singleton: build the index once on first tool use and cache it (embedding every chunk costs API calls).
- SECURITY: calculator does NOT use eval() (eval runs arbitrary code = injection risk). It parses an AST and whitelists
  only + - * / and numbers; anything else is rejected. Verified __import__("os") was blocked. (Strong interview point.)
- Tools tested in ISOLATION first (call directly, no LLM) to separate "do tools work?" from "does the agent choose right?".
- tool.invoke({...}) runs a tool; tool.name / tool.description / tool.args show what the LLM will see.
