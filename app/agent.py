# app/agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder

from dotenv import load_dotenv
load_dotenv()


# NEW: StructuredTool + Pydantic schemas
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from .tools import save_text_tool, call_api_tool, send_email_tool
from .rag import similarity_search

SYSTEM_PROMPT = (
    "You are an autonomous agent. Choose tools wisely. "
    "Use 'search_docs' for knowledge in the vector DB. "
    "If confidence is low or the issue is critical, escalate via send_email. "
    "Always explain what you did. Keep answers concise."
)

# ---------- Pydantic arg schemas ----------

class SaveTextInput(BaseModel):
    text: str = Field(..., description="The content to save")
    tag: str = Field("note", description="Optional tag for classification")

class CallApiInput(BaseModel):
    endpoint: str = Field(..., description="e.g., /orders/12345")
    method: str = Field("GET", description="HTTP method, default GET")

class SendEmailInput(BaseModel):
    to: str = Field(..., description="Email recipient")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body text")

class SearchDocsInput(BaseModel):
    query: str = Field(..., description="Semantic search query")
    k: int = Field(4, ge=1, le=20, description="Top-k passages to retrieve")

# ---------- Build agent ----------

def build_agent(model: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model=model, temperature=0.2)

    tools = [
        StructuredTool.from_function(
            name="save_text",
            func=lambda text, tag="note": save_text_tool(text=text, tag=tag),
            args_schema=SaveTextInput,
            description="Store snippets or FAQ candidates."
        ),
        StructuredTool.from_function(
            name="call_api",
            func=lambda endpoint, method="GET": call_api_tool(endpoint=endpoint, method=method),
            args_schema=CallApiInput,
            description="Mock external API. Supports GET /orders/{id}."
        ),
        StructuredTool.from_function(
            name="send_email",
            func=lambda to, subject, body: send_email_tool(to=to, subject=subject, body=body),
            args_schema=SendEmailInput,
            description="Escalate complex or low-confidence cases."
        ),
        StructuredTool.from_function(
            name="search_docs",
            func=lambda query, k=4: "\n\n".join(similarity_search(query=query, k=int(k))),
            args_schema=SearchDocsInput,
            description="Retrieve relevant passages from the Chroma knowledge base (RAG)."
        ),
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        system_message=SystemMessage(content=SYSTEM_PROMPT),
        memory=memory,
        agent_kwargs={
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")]
        },
    )
    return agent
