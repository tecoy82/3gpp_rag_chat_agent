"""
Layer 4: Agent
--------------
A LangGraph agent that wraps the RAG retriever as a tool.

Why an agent vs a simple chain?
- A chain always retrieves then answers. Fixed pipeline.
- An agent DECIDES when to retrieve, how many times, whether to rephrase
  the query for a better retrieval, and when it has enough context to answer.

Mental model for interviews: agent = LLM + tools + a loop.
  Loop: LLM thinks -> calls a tool -> observes result -> thinks again -> ...
  Until: LLM decides it has enough to give a final answer.

LangGraph models this loop as a graph of nodes:
  [user input] -> [agent node] -> [tool node] -> [agent node] -> ... -> [END]
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CHROMA_PATH, EMBEDDING_MODEL, RETRIEVER_TOP_K

def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


def build_agent():
    # --- Vector store (must be pre-built by embed.py) ---
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_PATH}. Run scripts/embed.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": _best_device()},
    )
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    # --- Tool: RAG retrieval ---
    # The agent will call this tool when it needs to look up 3GPP spec content
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_3gpp_specs",
        description=(
            "Search the 3GPP specification documents for information about LTE (4G) "
            "and 5G NR standards. Use this tool whenever the user asks about network "
            "architecture, protocols, interfaces, procedures, or technical parameters "
            "defined in 3GPP standards."
        ),
    )

    # --- LLM ---
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,  # deterministic answers for technical Q&A
    )

    # --- Memory: persists conversation across turns ---
    # MemorySaver keeps history in RAM; swap for SqliteSaver for disk persistence
    memory = MemorySaver()

    # --- System prompt ---
    system_prompt = (
        "You are a 3GPP standards expert assistant specialising in LTE (4G) and 5G NR. "
        "Use the search_3gpp_specs tool to find relevant information. "
        "Use at most two searches. "
        "IMPORTANT: Never narrate your search process. Never start a response with "
        "'Let me search', 'I will search', 'Based on my search', or similar phrases. "
        "Start every response directly with the answer content. "
        "If the documents don't contain the answer, say so clearly. "
        "Cite the specification (e.g. TS 38.300) your answer comes from."
    )

    # create_react_agent implements the ReAct loop (Reason + Act)
    agent = create_react_agent(
        llm,
        tools=[retriever_tool],
        checkpointer=memory,
        prompt=system_prompt,
    )

    return agent


def chat(agent, question: str, session_id: str = "default") -> str:
    """Send one message to the agent and return the text response."""
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )
    return result["messages"][-1].content


def stream_chat(agent, question: str, session_id: str = "default"):
    """Generator that yields (event_type, data) tuples for SSE streaming.

    State machine logic:
      - Before any tool result: buffer AI tokens (could be reasoning before a tool call)
      - When a tool call is detected: discard buffer, emit "status"
      - After first tool result received: stream AI tokens immediately (this is the final answer)
      - If agent answers with no tool calls at all: flush the buffer as the answer

    This hides the agent's internal monologue ("Let me search...") and only
    streams the final answer to the user.

    Event types:
      "status" — agent is searching (show to user as progress indicator)
      "token"  — a text token from the final answer
      "done"   — complete answer string (used for caching)
    """
    config = {"configurable": {"thread_id": session_id}}
    full_response = []
    prev_text = None  # one-chunk lookahead: hold the previous chunk until we know it's safe

    for chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            raw = chunk.content
            if isinstance(raw, list):
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in raw
                )
            else:
                text = raw or ""

            if chunk.tool_calls:
                # Current chunk is a tool call → prev_text was reasoning, discard it
                prev_text = None
                yield "status", "Searching 3GPP specifications..."
            else:
                # Current chunk is more text → prev_text is confirmed safe to yield
                if prev_text:
                    full_response.append(prev_text)
                    yield "token", prev_text
                prev_text = text if text else None
        else:
            # ToolMessage → prev_text was pre-tool reasoning, discard it
            prev_text = None

    # End of stream → prev_text is the last token of the final answer
    if prev_text:
        full_response.append(prev_text)
        yield "token", prev_text

    yield "done", "".join(full_response)


if __name__ == "__main__":
    agent = build_agent()
    print("3GPP RAG Agent ready. Type 'quit' to exit.\n")
    session = "cli-session"
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        answer = chat(agent, q, session)
        print(f"\nAgent: {answer}\n")
