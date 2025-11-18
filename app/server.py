# app/server.py
# This gives you a RESTful chat endpoint.

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import build_agent
from .tools import save_text_tool
from .fallback_detector import is_semantic_fallback


# Defines a simple ChatRequest model with one field: message.
# Builds one global agent instance.

app = FastAPI(title="Agentic AI — Round 1")


class ChatRequest(BaseModel):
    message: str


agent = build_agent()


# Accepts JSON: {"message": "hello"}.
# Runs the agent.
# Returns JSON: {"reply": "..."}.
@app.post("/chat")
def chat(req: ChatRequest):
    # Invoke the agent with structured input (OPENAI_FUNCTIONS-friendly)
    result = agent.invoke({"input": req.message})
    reply = result.get("output", str(result))

    # Semantic fallback detection → log FAQ candidate automatically
    try:
        is_fb, score = is_semantic_fallback(reply)
        if is_fb:
            # Store the raw user message as a potential FAQ candidate.
            # You can add PII guards here if you want (email/order-id stripping).
            save_text_tool(
                text=req.message,
                tag="faq_candidate",
                meta={
                    "source": "fastapi",
                    "similarity": score,
                },
            )
    except Exception:
        # Never break the API just because logging failed
        pass

    return {"reply": reply}


# for example, with uvicorn:
# uvicorn app.server:app --reload
# Then POST to /chat.
