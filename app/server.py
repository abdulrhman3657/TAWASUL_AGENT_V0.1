# app/server.py
# This gives you a RESTful chat endpoint.

# to run app with uvicorn:
# uvicorn app.server:app --reload

from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import build_agent
from .tools import save_text_tool
from .fallback_detector import is_semantic_fallback


app = FastAPI(title="Agentic AI — Round 1")


class ChatRequest(BaseModel):
    message: str
    # Optional session_id. If not provided, we fall back to a single 'default' session.
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


# In-memory registry of agents, keyed by session_id.
# For a simple demo, this is fine. For large scale, you'd want a more robust store.
_agents: Dict[str, object] = {}

# Optional: cap the number of agents to avoid unbounded growth in a long-running server.
_MAX_AGENTS = 1000


def _get_agent_for_session(session_id: str):
    """Return an agent for this session_id, creating it if needed."""
    # Reuse existing agent if present
    if session_id in _agents:
        return _agents[session_id]

    # Very simple cap: if we have too many, clear everything (demo only).
    # You could implement LRU or smarter eviction if needed.
    if len(_agents) >= _MAX_AGENTS:
        _agents.clear()

    agent = build_agent()
    _agents[session_id] = agent
    return agent


# Accepts JSON: {"message": "hello", "session_id": "abc123"}.
# Runs the agent for that session.
# Returns JSON: {"reply": "...", "session_id": "abc123"}.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Use supplied session_id, or fall back to a default one.
    session_id = req.session_id or "default"

    agent = _get_agent_for_session(session_id)

    # Invoke the agent with structured input (OPENAI_FUNCTIONS-friendly)
    result = agent.invoke({"input": req.message})
    reply = result.get("output", str(result))

    # Semantic fallback detection → log FAQ candidate automatically
    try:
        is_fb, score = is_semantic_fallback(reply)
        if is_fb:
            # Store the raw user message as a potential FAQ candidate.
            # You can add stronger PII guards here if you want (email/order-id stripping).
            save_text_tool(
                text=req.message,
                tag="faq_candidate",
                meta={
                    "source": "fastapi",
                    "similarity": score,
                    "session_id": session_id,
                },
            )
    except Exception:
        # Never break the API just because logging failed
        pass

    return ChatResponse(reply=reply, session_id=session_id)
