# CodeRAG — an agentic RAG platform over technical docs

A retrieval-augmented generation system with an **agent brain** and a
**production-shaped, polyglot architecture**: a Go ingress gateway in front of a
Python/LangGraph agent that decides when to retrieve, calls tools, and answers
questions grounded in a document corpus — with citations and a measured
evaluation harness.

Built as a personal learning project to go deep on how real LLM systems are
engineered (modeled on a Jira/Confluence knowledge-assistant use case). It is
not a shipped product; the value is the depth and the design decisions.

---

## Architecture

```
          [ client ]
              │  HTTP  POST /ask
              ▼
 ┌───────────────────────────────┐
 │  GO INGRESS GATEWAY (gateway/) │   logging · rate limiting · forwarding
 │  - protects the backend        │   (edge concerns; no AI reasoning)
 └───────────────┬───────────────┘
                 │  HTTP  POST /agent
                 ▼
 ┌───────────────────────────────┐
 │  PYTHON AGENT (agent/)         │   LangGraph ReAct loop:
 │  - decides which tool to use   │   think → act → observe → repeat
 └───────┬───────────────┬───────┘
         │               │
   search_docs        calculator
         │               │
         ▼               ▼
 ┌────────────────┐  (demonstrates an agent
 │ RAG RETRIEVAL  │   choosing NOT to retrieve)
 │ (ingestion/)   │
 │ chunk·embed·   │
 │ cosine search  │
 └────────────────┘
```

Cross-language communication is over **HTTP** (Go calls Python by POSTing to it),
mirroring how production LLM systems split a high-throughput serving gateway
(Go) from the agent/ML brain (Python).

---

## Components

| Folder | Language | Role |
| --- | --- | --- |
| `gateway/` | Go | Ingress gateway: logging, per-client rate limiting, forwards requests to the agent, relays responses. The serving-layer concerns Go is strong at. |
| `agent/` | Python | LangGraph agent — the reasoning brain. Hand-built graph (state, nodes, conditional edge, loop) plus a prebuilt version for comparison. Tools: `search_docs`, `calculator`. Runs as an HTTP server. |
| `ingestion/` | Python | The RAG building blocks: chunking, embeddings (pluggable: Gemini + a hash baseline), an in-memory vector store, ingest/query scripts. |
| `eval/` | Python | Evaluation harness: recall@k over a labelled test set; a hybrid-search (RRF) experiment. The measurement that turns "it feels better" into numbers. |
| `go-service/` | Go | A standalone full-RAG service (loads a prebuilt index, embeds the query, retrieves, generates). Demonstrates the production pattern of ingest-offline / load-and-serve. |
| `docs/` | — | `CONCEPTS.md` (study notes), `DECISIONS.md` (design tradeoffs), `PLAN.md`, `COST-GUARDRAILS.md`. |

---

## Measured results (the differentiator)

Most RAG projects ship no evaluation. This one measures retrieval quality with
**recall@k** on a labelled test set whose questions are deliberately worded
differently from the source docs (to test meaning, not word overlap):

| Retrieval approach | recall@1 |
| --- | --- |
| Lexical baseline (hash embedder) | 0.50 |
| Semantic (Gemini embeddings) | **1.00** |

A **hybrid** search (semantic + keyword fused with Reciprocal Rank Fusion) was
implemented and measured — and it **regressed** (0.83 vs 1.00). The cause: the
keyword ranker was too weak and the corpus too small for semantic retrieval to
fail, so equal-weight fusion dragged good results down. The eval caught the
regression before it shipped; semantic-only was kept. (Details in
`docs/DECISIONS.md` #9.) The point: measure, don't assume.

---

## Agentic behavior

The agent *decides* per question, instead of always retrieving:

| Question | Tool chosen |
| --- | --- |
| "How does authentication work?" | `search_docs` (retrieve) |
| "What is 47 * 13?" | `calculator` (skips retrieval) |
| "Hi, what can you help with?" | no tool (answers directly) |

Grounded generation refuses to hallucinate: asked something outside the corpus
("what is the refund policy?"), it answers *"I don't know based on the provided
documents"* rather than inventing an answer.

---

## Key engineering decisions (see `docs/DECISIONS.md`)

- **Cosine + L2 normalization** so similarity is a fast dot product.
- **Pluggable embedder interface** — started with a zero-cost hash embedder to
  validate the pipeline, then swapped in Gemini with no pipeline change.
- **Python/LangGraph for the agent brain, Go for the gateway** — agent latency
  is dominated by waiting on the LLM, so Go gives no speed win for the brain;
  the ecosystem favors Python. Go's strength is the gateway (connections,
  concurrency, streaming).
- **Bounded timeouts + retries + model fallback** on LLM calls, after hitting
  real transient 503s.
- **Rate-limit keyed on client IP** (not IP:port — the source port varies per
  connection, which would defeat the limiter).

---

## Running it

Prereqs: Go 1.22+, Python 3.11+, a `GEMINI_API_KEY` in a `.env` file at the repo root.

```bash
# 1. set up the Python env
python3.11 -m venv agent/.venv
agent/.venv/bin/pip install -r agent/requirements.txt

# 2. start the Python agent (listens on :9000)
agent/.venv/bin/python agent/server.py

# 3. in another terminal, start the Go gateway (listens on :8080)
cd gateway && go run .

# 4. ask a question through the gateway
curl -s -X POST localhost:8080/ask -d '{"question":"how does rate limiting work?"}'
```

Run the evaluation harness:
```bash
agent/.venv/bin/python eval/run_eval.py --mode gemini --top-k 1
agent/.venv/bin/python eval/run_eval.py --mode hash   --top-k 1
```

---

## Scope & honest limitations

Built (working, end-to-end): the full gateway → agent → tools → RAG vertical
slice, the eval harness, grounded generation with citations.

Deliberately **not** built (would be the next production steps):
- A real vector DB (Qdrant/HNSW) instead of in-memory brute-force search — needed at scale.
- Incremental ingestion (delta sync, upsert by stable id, deletions) instead of full re-index — essential for a live Jira/Confluence corpus.
- ACL-aware retrieval (don't return chunks a user can't see), response streaming, auth, circuit breaking, and a FastAPI/uvicorn agent server.

These are understood and documented (`docs/PLAN.md`, `docs/DECISIONS.md`) rather
than hand-waved — the limitations and their fixes are part of the design.
