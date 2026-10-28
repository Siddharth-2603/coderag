"""HTTP server wrapping the LangGraph agent (Go-integration, Python half).

The Go gateway will send questions here over HTTP. We expose:
    POST /agent   body: {"question": "..."}   -> {"question":..., "answer":...}
    GET  /healthz -> {"status":"ok"}

We use Python's built-in http.server so there is NO extra dependency. For
production you'd use FastAPI + uvicorn (async, concurrency, validation); this
is the minimal version for the learning slice.

The agent graph is built ONCE at startup and reused per request (so we don't
rebuild the index or recompile the graph on every call).

Run:
    python3 server.py            # listens on :9000
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_graph import build_graph, _as_text

# Build the agent graph once at import/startup — reused by every request.
_APP = build_graph()


def _run_agent(question: str) -> str:
    """Invoke the agent and return the final text answer."""
    result = _APP.invoke({"messages": [("user", question)]})
    # The final answer is the last AIMessage with text content.
    for msg in reversed(result["messages"]):
        if msg.__class__.__name__ == "AIMessage":
            text = _as_text(msg.content)
            if text.strip():
                return text
    return "(no answer produced)"


class AgentHandler(BaseHTTPRequestHandler):
    """Handles the HTTP requests the Go gateway will send."""

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/agent":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return
        question = (data.get("question") or "").strip()
        if not question:
            self._send(400, {"error": "question is required"})
            return
        try:
            answer = _run_agent(question)
        except Exception as e:  # keep the server up even if one request fails
            # Log the full detail server-side for debugging; return a clean error.
            import traceback
            print(f"[agent-server] ERROR: {e}")
            traceback.print_exc()
            self._send(502, {"error": f"agent failed: {type(e).__name__}"})
            return
        self._send(200, {"question": question, "answer": answer})

    # Quieter logging: one concise line per request.
    def log_message(self, fmt: str, *args) -> None:
        print(f"[agent-server] {self.address_string()} {fmt % args}")


def main() -> None:
    port = int(os.getenv("AGENT_PORT", "9000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), AgentHandler)
    print(f"agent server listening on http://127.0.0.1:{port}  (POST /agent, GET /healthz)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
