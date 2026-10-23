# Plan & Milestones

Cadence: ~2 hrs/day, dynamic. Report progress daily; targets adjust up or down.
Protected path: **RAG core + eval harness** always ships before serving-layer polish.

## Daily split (flexible)
- ~70 min  flagship project
- ~35 min  DSA
- ~15 min  system design (a few days/week, not daily)

## Milestones (vertical slices — each one works end to end)

### M0 — Scaffolding ✅ (in progress)
- Repo structure, README, plan, cost guardrails, .gitignore
- Decide: vector store (Qdrant vs pgvector), embedding source

### M1 — Thin end-to-end slice
- Python: load a few docs → chunk → embed → upsert into vector store
- Go: a single `/query` endpoint → embed query → retrieve top-k → return chunks
- Goal: ask a question, get relevant chunks back. No generation yet.

### M2 — Generation + streaming
- Go: wire retrieval → LLM → stream answer over SSE
- Basic prompt assembly (context + question)

### M3 — The differentiator: eval harness
- A small labelled test set (questions → expected source chunks/answers)
- Metrics: recall@k, answer relevance
- Baseline numbers recorded so later changes show measurable improvement

### M4 — Retrieval quality
- Configurable chunking strategies; compare via eval
- Hybrid search (semantic + keyword), reranking; measure tradeoffs
- Target a concrete win, e.g. "recall@5 68% → 84%"

### M5 — Serving-layer engineering (Go strength)
- Semantic caching, rate limiting, retries/fallback
- Metrics: p50/p95/p99 latency, tokens/sec, cost per request (Prometheus + Grafana)

### M6 — Deployment
- Dockerize → Kubernetes (kind/minikube) → Terraform
- Load test + latency/throughput writeup (case-study README)

## Parallel tracks
- DSA: pattern-based (sliding window, two pointers, graphs, heaps, DP basics)
- System design: rate limiter, LLM-serving API, caching, RAG system
- Targeted open source (from ~M3): 3–5 real PRs to tools used here (Ollama/Qdrant/etc.)

## Interview launch
- Resume rewrite around this project + Go backend depth
- Start applying once M3 is solid with real eval numbers
