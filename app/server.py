
from fastapi import FastAPI
from pydantic import BaseModel
from .agent import build_agent

app = FastAPI(title="Agentic AI — Round 1")

class ChatRequest(BaseModel):
    message: str

agent = build_agent()

@app.post("/chat")
def chat(req: ChatRequest):
    reply = agent.run(req.message)
    return {"reply": reply}
