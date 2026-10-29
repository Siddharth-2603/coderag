# CodeRAG MCP Server

Exposes the RAG retrieval as an **MCP tool** (`search_docs`) that any
MCP-compatible assistant (Kiro, Claude Desktop, Cursor, ...) can call. This is
the *provider* side of tool-calling: instead of our agent calling tools, other
AIs call ours.

## What it exposes
- Tool: `search_docs(query: str)` — searches the document corpus and returns the
  most relevant excerpts with source citations. Reuses the same retrieval the
  agent uses (no duplicated logic).

## Transport
stdio — the MCP client launches `mcp_server.py` as a subprocess and exchanges
JSON-RPC over stdin/stdout. No port, no HTTP; the client manages the process.

## Wiring it into an MCP client
Add an entry like this to the client's MCP config (e.g. Kiro's
`.kiro/settings/mcp.json`, or Claude Desktop's config). Adjust the absolute
paths to this machine:

```json
{
  "mcpServers": {
    "coderag": {
      "command": "/Users/E.Siddharth/Desktop/projects/RAG/agent/.venv/bin/python",
      "args": ["/Users/E.Siddharth/Desktop/projects/RAG/agent/mcp_server.py"],
      "env": {},
      "disabled": false
    }
  }
}
```

Using the venv's Python directly (the `command`) ensures the MCP SDK and the
ingestion deps are available without activating the venv.

## Verified locally
- Server initializes and registers the tool: `list_tools()` -> `['search_docs']`.
- Tool invocation through the MCP machinery returns real retrieved content
  (`call_tool('search_docs', {...})` -> CallToolResult with source-cited chunks).
- Full client round-trip (clicking it inside an assistant) is done in the client
  via the config above.

## Note on the SDK version
Built against the MCP Python SDK **v2** (`from mcp.server.mcpserver import
MCPServer`). In v1 this class was `FastMCP` — the SDK renamed it. If you install
`mcp<2`, the import changes accordingly.
