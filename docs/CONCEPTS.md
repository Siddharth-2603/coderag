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

## Agent loop (step 3 — LangGraph ReAct)
- LangGraph models an agent with: STATE (the growing message list), NODES (units of work), EDGES (flow between nodes).
- Two nodes: "agent" node (LLM decides next move) and "tools" node (runs the requested tool, appends result to state).
- Conditional edge: after the agent node, if a tool was requested -> tools node -> BACK to agent node; if final answer -> END.
- The back-to-agent edge is what makes it a LOOP (think -> act -> observe -> repeat).
- State ACCUMULATES (question + tool calls + tool results + answer); each node appends. That accumulation is the agent's memory.
- ReAct = Reasoning + Acting; create_agent() (current API) builds the whole graph (binds tools + wires nodes/edges) for us.
- temperature=0 -> deterministic tool choices (same question -> same decision), which is what you want for an agent.
- Observed the agent REPHRASE the question into its own tool query ("authentication token expiration") — real reasoning, not forwarding.
- Version lesson: create_react_agent (langgraph.prebuilt) is deprecated in v1.0; current is create_agent from langchain.agents (needs the `langchain` package).
- Gemini may return structured content ([{'type':'text','text':...}]) not a plain string; normalize before printing.

## Hand-built LangGraph agent (step 4 — the real skill)
- Prebuilt create_agent() hides the graph; building it by hand is how you actually learn LangGraph.
- STATE = TypedDict with messages: Annotated[list, add_messages]. add_messages is a REDUCER = APPENDS (not replaces).
  The append behavior is what makes state accumulate = the agent's memory.
- agent_node: prepend system prompt, call tool-bound LLM, return {"messages":[response]} so it's appended.
- tools_node: read last message's tool_calls, look up each tool by name, run with LLM args, wrap result in ToolMessage
  (tool_call_id links the result to the specific call). Append results.
- conditional edge should_continue: if last_message.tool_calls present -> "tools"; else -> END. Pure routing, no LLM.
- Assemble: StateGraph(State) -> add_node x2 -> add_edge(START,"agent") -> add_conditional_edges("agent", fn, map)
  -> add_edge("tools","agent") [the loop-back] -> .compile().
- The single add_edge("tools","agent") is what creates the LOOP (after a tool runs, go back so the LLM sees the result).
- Two decisions, two places: LLM (agent node) chooses WHICH tool via docstrings; edge only checks WHETHER a tool call exists.

## Demonstrating agentic choice (step 5)
- Proving it's an agent = show it ROUTES differently by question type (a fixed pipeline would treat all the same).
- Verified routing: docs Q -> search_docs; math Q ("47*13") -> calculator (SKIPS retrieval); greeting -> NO tool (direct answer).
- The math + greeting cases are the proof: a fixed RAG would pointlessly retrieve; the agent decides retrieval is irrelevant.
- No new agent code needed — step 5 just RUNS the existing graph on varied inputs and observes which tool each triggered.
- Interview demo: this contrast (retrieve vs calculate vs answer-directly) is the clearest evidence of agentic behaviour.

## Production agent architecture (target shape)
- 3 layers: Go INGRESS GATEWAY (auth, rate-limit, streaming, routing, 100k conns)
  -> Python AGENT (LangGraph ReAct loop, state/checkpointing) -> BACKING SERVICES (tools).
- Request flows DOWN through layers and back UP once (not ping-pong). Bottom Go services != the top Go gateway.
- What we build (vertical slice): Go gateway -> Python agent -> existing tools. Working end-to-end.
- What we only MENTION (productionizing): Redis/Postgres checkpointing, gRPC, circuit breakers, token billing, 100k conns.

## Tools: retrieval vs action
- Retrieval tool (search_docs) = fetch INFORMATION. Action tool (process_order) = DO something in another system.
- Action tools are what make agents powerful (create orders, update records) — not just Q&A.
- Mechanism is the same: the tool's Run sends an HTTP call to a backing Go/Java microservice and returns the result.

## Adding a new source (e.g. Jira / Confluence) — two ways
- Approach 1 INGEST: pull pages via their REST API -> chunk -> embed -> vector store; search_docs finds them.
  Use for STATIC reference content you want to search semantically (design docs, runbooks). Just a new ingestion connector.
- Approach 2 LIVE TOOL: a tool (e.g. search_jira) that queries the API in REAL TIME when the agent calls it.
  Use for DYNAMIC content/actions ("open tickets now", "status of PROJ-123").
- MCP = the STANDARDIZED way to do Approach 2: vendors (e.g. Atlassian) publish MCP servers exposing Jira/Confluence
  as ready-made tools, so you connect to them instead of hand-coding the API integration.
- Rule of thumb: static+searchable -> ingest; dynamic/live/actions -> tool (MCP if a server exists). Real systems do BOTH.

## Go<->Python integration: Python agent as HTTP server (half 1)
- To let the Go gateway "call" the agent, the agent runs as an HTTP SERVER (POST /agent, GET /healthz).
- Communication is HTTP between two running programs — NOT importing code across languages.
- Used Python's built-in http.server (no dependency); production would use FastAPI+uvicorn (async, validation, concurrency).
- Build the graph ONCE at startup, reuse per request (don't rebuild index/recompile graph each call).
- Error handling: a failed request returns 502 and is logged server-side, but does NOT crash the server (try/except).

## Debugging lesson: silent LLM hangs
- Symptom: /agent requests hung for minutes, then GoogleAPIError. healthz fine (so server was alive).
- Root cause: transient 503 "model experiencing high demand" on gemini-3.1-flash-lite + the client's built-in
  retries with long backoff = long silent stalls.
- Fixes applied: (1) set timeout=30 + max_retries=2 on the LLM so a flaky call FAILS FAST instead of hanging;
  (2) switch model to gemini-flash-lite-latest (the other was overloaded). Same 503 lesson as M2 generation.
- General lesson: always set explicit timeout + bounded retries on external API calls, and log full errors server-side
  (my first handler hid the real error, making it look like a code bug when it was a 503).

## Registration vs execution (when the index is built)
- Registering a tool (giving the LLM its name+description) does NOT run the tool's body.
- Build-graph at startup = tools registered; NO tool code runs yet, so NO index/embeddings built.
- The index builds ONLY when search_docs is actually CALLED (LLM chose it) -> _ensure_index() runs on first such call.
- A pure math/greeting question never calls search_docs -> index is never built. Lazy = pay embedding cost only when needed.
- Two embedding types / two times: DOCUMENT embeddings = whole corpus, once (lazy in agent / offline in Go via ingest.py);
  QUERY embedding = one per request (can't precompute an unknown question).

## Go gateway (integration half 2 — architecture complete)
- Gateway is a SEPARATE Go program (gateway/) from the RAG service (go-service/): different role = different deployable.
- Flow: client --POST /ask--> gateway :8080 --POST /agent--> python agent :9000 --> tools --> answer --> back.
- Gateway does edge concerns only (log, rate-limit, forward, relay) — NO RAG/agent reasoning.
- "Go calls Python" = Go sends an HTTP POST to the running Python server. Cross-language = HTTP, not imports.
- Gateway client timeout (90s) must EXCEED the agent's internal LLM timeout so the agent can finish before we give up;
  server WriteTimeout must exceed the client timeout too.
- /healthz also pings the agent's /healthz -> reports agent_reachable (a gateway should know if its backend is alive).
- Rate limiter: fixed-window per client, guarded by a Mutex (safe across concurrent goroutines). Rejects with 429
  BEFORE calling the agent (protect the backend). Fixed-window is simplest; prod uses token-bucket/sliding-window.
- BUG found + fixed: keyed the limiter on r.RemoteAddr (ip:PORT) -> port changes per request -> every request looked
  like a new client -> limiter never fired. Fix: net.SplitHostPort to key on IP only. (Verified 429 after the fix.)
- Productionizing (mention, not built): auth, response streaming, circuit breaking, X-Forwarded-For parsing.

## defer + mutex, and IP-vs-port in rate limiting
- defer X runs X when the surrounding FUNCTION returns (any return path, even a panic) — not end-of-block.
- "lock then defer unlock" guarantees the mutex is released on every exit path; avoids forgetting an unlock on a new return = deadlock. Idiomatic Go.
- Two different ports: SERVER port (gateway :8080, fixed, every client connects to it) vs CLIENT SOURCE port (random
  ephemeral port the client OS picks PER CONNECTION).
- r.RemoteAddr = clientIP:clientSourcePort (the client's address, not the server's :8080).
- Source port VARIES per connection: curl = new port each call; browsers open several connections = several ports.
  That's why keying the limiter on ip:port made every request look like a new client (the bug).
- Fix: key on IP only (stable per machine). Verified it then limits correctly.
- Caveat (prod): IP isn't ideal — users behind one NAT/proxy share an IP (shared bucket); mobile IPs change.
  Production rate-limits on API key / authenticated user id instead.

## MCP server (provider side of tool-calling)
- Agent = tool CONSUMER (its LLM calls tools). MCP server = tool PROVIDER (other AIs call OUR tool).
- MCP = a protocol (JSON-RPC) letting an AI client discover + call tools/resources/prompts on a server.
- Transport: stdio is the common local pattern — client launches the server as a subprocess, talks over stdin/stdout.
  (HTTP/SSE transport also exists.) Different from our gateway's plain HTTP.
- A server exposes tools (callable fns), resources (readable data), prompts (templates). We exposed one tool: search_docs.
- Our MCP server is a THIN wrapper reusing the agent's existing retrieval — one retrieval impl, exposed two ways.
- @server.tool() registers a fn; the SDK builds the tool declaration from name+docstring+typed params (like LangChain @tool).
- Wiring into a client = a JSON config entry (command=venv python, args=[mcp_server.py]) in e.g. .kiro/settings/mcp.json.
- Verified: list_tools() -> ['search_docs']; call_tool('search_docs',{...}) -> CallToolResult with source-cited chunks.
- Version lesson (again): MCP SDK v2 renamed FastMCP -> MCPServer. APIs drift; read the current one. (pin mcp<2 for v1.)

## Confluence connector (ingesting an external source)
- A connector only adds "fetch + clean"; it REUSES the existing pipeline (chunk -> embed -> store).
- Split for testability: ConfluenceClient (network, needs token) vs normalize_page (pure logic) vs a Mock client (canned pages).
- PageSource interface lets mock and real clients swap freely -> verify the whole path offline, add creds later, no code change.
- Confluence API: GET /wiki/rest/api/content?expand=body.storage ; auth = HTTP Basic email:api_token over HTTPS.
- Body is storage-format XHTML -> strip tags to text (block tags -> newlines) before chunking.
- Stable doc_id CONF:<page_id> so re-ingest UPDATES not duplicates (basis of incremental sync).
- Carry metadata (source/title/url/updated_at) onto each chunk so answers can CITE the exact page.
- CRITICAL for real use: ACL/permissions — enforce per-user access at query time or RAG leaks restricted pages.
- Jira (dynamic) -> better as a LIVE TOOL / Atlassian MCP, not pre-ingested. Confluence (static docs) -> ingest.
