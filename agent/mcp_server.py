"""MCP server — exposes our RAG retrieval as a tool OTHER AI assistants can call.

This is the PROVIDER side of tool-calling. Our agent is a tool *consumer* (its
LLM calls tools); here we flip it: any MCP-compatible client (Claude Desktop,
Kiro, Cursor, ...) can connect to this server and call `search_docs` to query
our document corpus as a native tool.

Transport: stdio (the client launches this script as a subprocess and exchanges
JSON-RPC over stdin/stdout — the standard local-MCP pattern).

We reuse the EXISTING retrieval code (same as the agent), so this is a thin
protocol wrapper over capability we already built — no duplicated logic.

Run (for a client):  python3 mcp_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingestion"))

from embeddings import load_env  # noqa: E402

load_env(ROOT / ".env")

from mcp.server.mcpserver import MCPServer  # noqa: E402

# Reuse the agent's retrieval helpers (index build + the underlying tool logic).
# tools.py already has _ensure_index + the retrieval against the in-memory store.
import tools as rag_tools  # noqa: E402

server = MCPServer("coderag")


@server.tool()
def search_docs(query: str) -> str:
    """Search the project's technical documentation (authentication, caching,
    deployment, logging, rate limiting) and return the most relevant excerpts
    with their source document names. Use for questions about how this system
    works."""
    # Reuse the exact retrieval the agent uses. rag_tools.search_docs is a
    # LangChain tool object, so call its underlying function via .invoke.
    return rag_tools.search_docs.invoke({"query": query})


if __name__ == "__main__":
    # run() speaks the MCP protocol over stdio until the client disconnects.
    server.run()
