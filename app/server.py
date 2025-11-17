# This gives you a RESTful chat endpoint.

from fastapi import FastAPI
from pydantic import BaseModel
from .agent import build_agent

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
    result = agent.invoke({"input": req.message})
    reply = result.get("output", str(result))
    return {"reply": reply}


# for example, with uvicorn:
# uvicorn app.server:app --reload
# Then POST to /chat.
