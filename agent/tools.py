"""Agent tools (step 2).

A LangChain "tool" is a normal Python function + the @tool decorator. The
decorator reads the function's name, docstring, and typed parameters and turns
them into a declaration the LLM can call via function calling. So:
    - function name   -> tool name the LLM uses
    - docstring       -> description the LLM reads to decide WHEN to use it
    - parameters      -> arguments the LLM must supply
    - function body   -> what actually runs when the tool is called

We define two tools so the agent later has a real CHOICE:
    - search_docs : retrieve from the indexed documentation (knowledge tool)
    - calculator  : evaluate a simple math expression (so the agent can choose
                    NOT to retrieve for questions like "2+2")

This step only DEFINES and TESTS the tools directly. Wiring them to the LLM and
the decision loop comes in later steps.
"""

from __future__ import annotations

import ast
import operator
import sys
from pathlib import Path

from langchain_core.tools import tool

# Reuse the existing ingestion code so retrieval behaves exactly like the eval
# harness (same embedder, same store). We add the ingestion folder to the path.
ROOT = Path(__file__).resolve().parent.parent
ING = ROOT / "ingestion"
sys.path.insert(0, str(ING))

from embeddings import load_env, make_embedder  # noqa: E402
from store import InMemoryStore                 # noqa: E402
from chunking import chunk_text                 # noqa: E402

DOCS_DIR = ROOT / "sample-docs"
TEXT_SUFFIXES = {".md", ".txt", ".py", ".go", ".rst"}

# --- lazy singletons: build the index + embedder once, reuse across calls ---
_store: InMemoryStore | None = None
_embedder = None


def _ensure_index() -> None:
    """Build the in-memory index on first use (embeds every chunk once)."""
    global _store, _embedder
    if _store is not None:
        return
    load_env(ROOT / ".env")
    _embedder = make_embedder("gemini")
    chunks = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            chunks.extend(chunk_text(path.name, path.read_text(encoding="utf-8")))
    vectors = [_embedder.embed(c.text) for c in chunks]
    store = InMemoryStore()
    store.upsert(chunks, vectors)
    _store = store


@tool
def search_docs(query: str) -> str:
    """Search the project's technical documentation for passages relevant to a
    question about the system (authentication, caching, deployment, logging,
    rate limiting, etc.). Use this when the answer depends on the project's docs.
    Returns the most relevant document excerpts."""
    _ensure_index()
    query_vec = _embedder.embed(query)
    results = _store.search(query_vec, top_k=3)
    if not results:
        return "No relevant documents found."
    parts = []
    for chunk, score in results:
        parts.append(f"(source: {chunk.doc_id}, score: {score:.2f})\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


# Safe arithmetic: only allow +, -, *, /, parentheses, numbers. No eval() of
# arbitrary code — we parse the expression and permit a tiny operator set only.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (addition, subtraction,
    multiplication, division, parentheses). Use this for direct math questions
    like '2 + 2' or '(10 * 3) / 2'. Do NOT use it for anything about the
    project's documentation."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception:
        return f"Could not evaluate expression: {expression!r}"
