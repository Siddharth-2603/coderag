"""Step 5 — demonstrate that the agent actually CHOOSES.

Runs several question types through the hand-built graph and reports which tool
(if any) the agent decided to use for each. This proves agentic behaviour:
  - docs question   -> search_docs (retrieve)
  - math question   -> calculator  (skip retrieval)
  - greeting/general -> no tool    (answer directly)

A fixed RAG pipeline would retrieve for ALL of these; an agent routes them.
"""

from __future__ import annotations

from agent_graph import build_graph, _as_text

QUESTIONS = [
    "How does authentication work in this system?",   # expect: search_docs
    "What is 47 * 13?",                                # expect: calculator
    "Hi! What kinds of things can you help me with?",  # expect: no tool
    "What does a 429 response mean here?",             # expect: search_docs
    "What is (100 - 16) / 4?",                         # expect: calculator
]


def tools_used(messages) -> list[str]:
    """Collect the names of every tool the agent called during the run."""
    used = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            used.append(tc["name"])
    return used


def final_answer(messages) -> str:
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage":
            text = _as_text(msg.content)
            if text.strip():
                return text
    return "(no text answer)"


def main() -> None:
    app = build_graph()
    print("\n================ AGENT TOOL-CHOICE DEMO ================\n")
    for q in QUESTIONS:
        result = app.invoke({"messages": [("user", q)]})
        used = tools_used(result["messages"])
        label = ", ".join(used) if used else "NO TOOL (answered directly)"
        answer = final_answer(result["messages"]).replace("\n", " ")
        print(f"Q: {q}")
        print(f"   -> tool chosen: {label}")
        print(f"   -> answer: {answer[:100]}")
        print()


if __name__ == "__main__":
    main()
