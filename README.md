# CodeRAG — A Retrieval-Augmented Generation Platform over Codebases & Technical Docs

A production-shaped RAG system that answers questions over a large codebase and its
technical documentation. Built to go deep on the engineering behind real LLM serving:
retrieval quality, serving-layer performance, and measurable evaluation.

## Why this project

Most RAG demos stop at "it answers three questions." This one is built around the parts
that actually signal engineering depth:

- A **Go serving layer** (retrieval + generation API, streaming, caching, rate limiting, metrics)
- A **Python ingestion pipeline** (chunking, embeddings, vector-store writes)
- An **evaluation harness** that *measures* retrieval quality (recall@k, answer relevance),
  so improvements are provable, not vibes.

The domain — code + technical docs — is deliberately chosen: quality is hard, measurable,
and the hard problems (chunking code, keeping context across files, hybrid search) are the
ones that make the system interesting.

## Architecture

```
            ┌────────────────────┐
            │  Go Serving Layer   │   ← query API, streaming (SSE),
            │  (gateway + RAG API)│     semantic cache, rate limit, metrics
            └─────────┬───────────┘
                      │ retrieve
                      ▼
            ┌────────────────────┐
            │   Vector Database   │   ← Qdrant / pgvector
            └─────────┬───────────┘
                      ▲ write
                      │
            ┌────────────────────┐
            │ Python Ingestion    │   ← load → chunk → embed → upsert
            │ (workers)           │
            └────────────────────┘
                      │
                      ▼
            ┌────────────────────┐
            │  Eval Harness       │   ← recall@k, relevance, regression checks
            └────────────────────┘
```

## Tech stack

| Layer            | Choice                          | Notes                                  |
| ---------------- | ------------------------------- | -------------------------------------- |
| Serving          | Go                              | Core strength; API, streaming, caching |
| Ingestion        | Python                          | Chunking + embeddings                  |
| Vector store     | Qdrant or pgvector (TBD)        | Decide in milestone 1                  |
| Embeddings       | Hosted API or local model (TBD) | Cheapest path first                    |
| Generation       | Free-tier API or small local    | Keep cost near zero                    |
| Deployment       | Docker → Kubernetes → Terraform | Later milestones                       |
| Cloud            | GCP                             | $300 free credit; AWS concepts port    |

## Status

Early scaffolding. See `docs/PLAN.md` for milestones and `docs/COST-GUARDRAILS.md`
for the budget discipline this project runs under (target: well under the budget cap).

## Note on origin

This is a personal learning project built to develop AI-platform / LLM-infrastructure
skills. It is not affiliated with or owned by any employer.
