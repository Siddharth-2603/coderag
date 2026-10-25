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


## Later milestones (tracked, sequenced after the agent layer)

These are valuable but deliberately deferred so the agent layer (higher value
for the AI-engineer target) comes first. Do them if time allows before January.

### MCP server wrapper (provider side of tool-calling)
- Expose the RAG /query as an MCP (Model Context Protocol) server so any
  MCP-capable assistant can call it as a tool ("search my project's docs").
- Complements the agent layer: agent = LLM CALLS tools; MCP = we EXPOSE tools to other LLMs.
- Value: current/in-demand; strong resume line; deepens tool-calling understanding.

### GCP deployment (the infra / "I operated it" story)
- NOTE: GCP account exists but is UNUSED so far — everything currently runs locally.
  Can't claim GCP on the resume until this is done.
- Containerize the Go service (Docker) -> deploy to GKE (managed Kubernetes) ->
  provision with Terraform. Add basic metrics/observability.
- Value: turns "I built a service" into "I deployed & operated it on GCP" (infra story).
- Lower priority than agents for the AI-engineer target, but a strong capstone.


## Real use case: Jira & Confluence ingestion (interview narrative)

Honest framing: modeled on a real company context (exploring AI; a sister team
building RAG for Jira). Built as a learning project that designs this use case
end to end — NOT claimed as a shipped production integration.

### Which approach per source (ingest vs live tool)
- **Confluence pages** (design docs, runbooks, specs) = relatively static reference
  content -> INGEST into the vector index. Searched semantically via search_docs.
- **Jira issues** = split by need:
  - Historical/closed issues used as knowledge -> INGEST (semantic search over past tickets).
  - "What's open now / status of PROJ-123" = dynamic -> LIVE TOOL (query Jira API at request time).
- Real system does BOTH: ingest the stable stuff, live-tool the dynamic stuff.
- Live tool can be hand-written OR via the Atlassian MCP server (standardized, pre-built tools).

### Ingestion connector design (Confluence/Jira -> index)
- Pull content via the Atlassian REST API (Confluence pages, Jira issues + comments).
- Normalize: strip markup, keep metadata (space/project, page/issue id, author, updated_at, URL).
- Chunk (same pipeline) -> embed -> upsert into the vector store with a STABLE chunk id
  derived from source id (e.g. "CONF:<pageId>:<n>", "JIRA:<issueKey>:<n>").
- Keep source metadata on each chunk so answers can cite the exact page/issue + link.

### Incremental refresh (the key scaling point)
- Current project does FULL re-index every run (re-embeds everything, overwrites index.json).
  Fine for a tiny corpus; unacceptable for all of Confluence/Jira.
- Production approach = INCREMENTAL / delta sync:
  - Track each source doc by updated_at / version / content hash.
  - On sync: only re-embed NEW or CHANGED docs; leave unchanged chunks as-is.
  - upsert changed chunks by stable id; DELETE chunks for removed/archived pages/issues.
  - Run on a schedule (e.g. periodic poll) or event-driven (Atlassian webhooks on page/issue update).
- Tradeoff: full rebuild = simple + always consistent but expensive/slow; incremental = cheap +
  supports live updates but must track changes and handle deletions (risk of stale/orphaned chunks).

### Hard parts to mention (shows depth)
- **Permissions/ACLs**: Confluence/Jira content is access-controlled. A real system must respect
  per-user permissions at query time (don't return chunks a user can't see) — filter by the
  requesting user's access, or index ACL metadata and enforce on retrieval.
- **Chunking structured content**: Jira issues have fields + threaded comments; chunking must
  preserve that structure, not blindly window lines.
- **Freshness vs cost**: sync frequency is a tradeoff (stale answers vs embedding cost/API rate limits).
- **Citations**: every answer should link back to the exact Confluence page / Jira issue.

### Known limitation of the current build (honest talking point)
- "My ingestion does a full re-index for simplicity. For Confluence/Jira at scale I'd move to
  incremental upsert keyed by stable source ids, driven by webhooks, with ACL-aware retrieval,
  backed by a real vector DB (Qdrant). Full rebuild trades efficiency for simplicity."
