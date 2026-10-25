"""The agent, built BY HAND in raw LangGraph (step 4).

Same think -> act -> observe loop as agent.py, but instead of the prebuilt
create_agent() we construct the graph ourselves: the state, the two nodes, the
conditional edge, and the wiring. This is how you actually learn LangGraph.

The graph we are building:

        ┌──────────────────────────────┐
        ▼                              │
   (START)->[agent node] --has tool_call--> [tools node]
                  │                              (loops back up)
                  └── no tool_call ──> (END)

Usage:
    python3 agent_graph.py "your question"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingestion"))

from embeddings import load_env  # noqa: E402

load_env(ROOT / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI       # noqa: E402
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage  # noqa: E402
from langgraph.graph import StateGraph, START, END              # noqa: E402
from langgraph.graph.message import add_messages                # noqa: E402

from tools import search_docs, calculator                       # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful assistant for a software project. "
    "Use search_docs for questions about the project's documentation "
    "(auth, caching, deployment, logging, rate limiting). "
    "Use calculator for arithmetic. "
    "If a question needs neither tool, answer directly. "
    "Ground documentation answers in the retrieved text and cite the source doc."
)

# ---------------------------------------------------------------------------
# 1. THE STATE
# The state is what flows through the graph. For an agent it is just the
# message list. The `add_messages` reducer is the important part: when a node
# returns messages, they are APPENDED to the existing list (not replaced).
# That append-behavior is what makes the state accumulate = the agent's memory.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Model with tools bound (so it can emit tool_calls), and a tool registry so
# the tools node can look up a tool by the name the LLM asked for.
_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
_tools = [search_docs, calculator]
_llm_with_tools = _llm.bind_tools(_tools)
_tools_by_name = {t.name: t for t in _tools}


# ---------------------------------------------------------------------------
# 2. THE AGENT NODE
# Calls the LLM with the full conversation so far (prepending the system
# prompt). The LLM returns one AIMessage — which may contain tool_calls or a
# final answer. We return it wrapped in {"messages": [...]} so add_messages
# appends it to the state.
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    conversation = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(conversation)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# 3. THE TOOLS NODE
# Runs whatever tool(s) the LLM requested in the last message. For each
# tool_call: look up the tool by name, run it with the LLM-supplied args, and
# wrap the result in a ToolMessage (tagged with tool_call_id so the LLM knows
# which call it answers). These get appended to the state.
# ---------------------------------------------------------------------------
def tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        tool = _tools_by_name[call["name"]]
        output = tool.invoke(call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


# ---------------------------------------------------------------------------
# 4. THE CONDITIONAL EDGE
# Pure routing logic (no LLM): look at the last message. If it has tool_calls,
# route to the tools node; otherwise the LLM gave a final answer, so END.
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"   # there is a pending tool call -> run it
    return END           # no tool call -> we're done


# ---------------------------------------------------------------------------
# 5. ASSEMBLE THE GRAPH
# Add the two nodes, set the entry point, wire the edges:
#   START -> agent
#   agent -> (conditional) -> tools OR END
#   tools -> agent   (the loop-back that lets the LLM see tool results)
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after running tools, go back to the LLM

    return graph.compile()


def _as_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text"]
        return "\n".join(parts)
    return str(content)


def run(question: str) -> None:
    app = build_graph()
    result = app.invoke({"messages": [("user", question)]})
    print(f"\n=== Question: {question} ===\n")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"[AGENT -> tool] {tc['name']}({tc['args']})")
        elif role == "ToolMessage":
            print(f"[TOOL result] {_as_text(msg.content).replace(chr(10), ' ')[:110]}...")
        elif role == "AIMessage":
            text = _as_text(msg.content)
            if text.strip():
                print(f"\n[FINAL ANSWER]\n{text}")


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python3 agent_graph.py "your question"')
        return
    run(sys.argv[1])


if __name__ == "__main__":
    main()
