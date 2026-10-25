"""The agent (step 3) — the think -> act -> observe loop.

We assemble a ReAct-style agent (Reasoning + Acting) using LangGraph's prebuilt
helper. Under the hood it builds exactly the graph we described:

    [agent node] --tool requested--> [tools node] --result--> back to [agent node]
         |
         +--final answer--> END

The "state" is the growing list of messages (question, tool calls, tool
results, answer); every node appends to it — that accumulation is the agent's
memory across steps.

Usage:
    python3 agent.py "your question"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingestion"))

from embeddings import load_env  # noqa: E402

load_env(ROOT / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from langchain.agents import create_agent                  # noqa: E402

from tools import search_docs, calculator                  # noqa: E402

# A short system prompt steers overall behaviour. The per-tool docstrings still
# drive WHICH tool is chosen; this just sets the agent's general instructions.
SYSTEM_PROMPT = (
    "You are a helpful assistant for a software project. "
    "Use search_docs for questions about the project's documentation "
    "(auth, caching, deployment, logging, rate limiting). "
    "Use calculator for arithmetic. "
    "If a question needs neither tool, answer directly. "
    "Ground documentation answers in the retrieved text and cite the source doc."
)


def build_agent():
    """Create the ReAct agent: a Gemini model + our two tools, wired into the
    think/act/observe loop by LangGraph's prebuilt helper."""
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    # create_react_agent binds the tools to the model AND builds the graph
    # (agent node, tools node, the conditional loop edge) for us.
    agent = create_agent(
        model=llm,
        tools=[search_docs, calculator],
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def run(question: str) -> None:
    agent = build_agent()
    # The input is the initial state: one human message. The agent appends to
    # this message list as it thinks, calls tools, and answers.
    result = agent.invoke({"messages": [("user", question)]})
    messages = result["messages"]

    # Print the full trace so we can SEE the loop: which tools were called and
    # the final answer.
    print(f"\n=== Question: {question} ===\n")
    for msg in messages:
        role = msg.__class__.__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"[AGENT decided to call tool] {tc['name']}({tc['args']})")
        elif role == "ToolMessage":
            preview = _as_text(msg.content).replace("\n", " ")[:120]
            print(f"[TOOL result] {preview}...")
        elif role == "AIMessage":
            text = _as_text(msg.content)
            if text.strip():
                print(f"\n[FINAL ANSWER]\n{text}")


def _as_text(content) -> str:
    """Normalize message content to plain text. Newer Gemini models return a
    list of parts like [{'type':'text','text':...}] instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python3 agent.py "your question"')
        return
    run(sys.argv[1])


if __name__ == "__main__":
    main()
