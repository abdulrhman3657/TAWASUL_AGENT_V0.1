# app/agent.py
from langchain_openai import ChatOpenAI # OpenAI chat model wrapper.
from langchain.agents import initialize_agent, AgentType # builds an agent that uses OpenAI-style function calling.
from langchain_core.messages import SystemMessage # lets you inject a “system prompt” (instructions) into the conversation.
from langchain.memory import ConversationBufferMemory # keeps a history of messages.
from langchain_core.prompts import MessagesPlaceholder # placeholder so memory can be injected into the system prompt.

from dotenv import load_dotenv
load_dotenv(override=True)

# a tool wrapper that uses Pydantic schemas for arguments.
# This lets the LLM pass structured JSON arguments (like {"query": "...", "k": 5}) instead of just a single string.
# lets the agent call tools with typed JSON arguments (via Pydantic models).
# Pydantic models define the input schema for each tool.
from langchain.tools import StructuredTool

from pydantic import BaseModel, Field

from .tools import save_text_tool, call_api_tool, send_email_tool
from .rag import similarity_search

# This prompt instructs the model on when to use which tool and how to answer.
SYSTEM_PROMPT = (
    "You are an autonomous agent. Choose tools wisely. "
    "Use 'search_docs' for knowledge from the vector database. "
    "Use 'call_api' when the user asks about order data. "
    "Use 'save_text' to store useful notes or FAQ candidates. "

    "Escalation rules: "
    "Always call the 'send_email' tool and send the email to support@tawasul31.com "
    "when any of the following conditions are true: "
    "1) The user explicitly requests escalation, OR "
    "2) You cannot solve the user's issue, OR "
    "3) The issue is too complex, unclear, or beyond your capabilities. "

    "When escalating, summarize the issue clearly in the email body. "
    "After you call the escalation tool, briefly explain to the user that the issue "
    "has been escalated. "

    "Keep answers concise and always explain what you did."
)

# ---------- Pydantic arg schemas ----------

# Each tool gets a structured input specification:

# text: str          # required text to save
# tag: str = "note"  # optional classification tag
class SaveTextInput(BaseModel):
    text: str = Field(..., description="The content to save")
    tag: str = Field("note", description="Optional tag for classification")

# endpoint: str      # e.g. "/orders/12345"
# method: str = "GET"
class CallApiInput(BaseModel):
    endpoint: str = Field(..., description="e.g., /orders/12345")
    method: str = Field("GET", description="HTTP method, default GET")

# to: str
# subject: str
# body: str
class SendEmailInput(BaseModel):
    to: str = Field(..., description="Email recipient")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body text")

# query: str
# k: int = 4  # top-k docs
class SearchDocsInput(BaseModel):
    query: str = Field(..., description="Semantic search query")
    k: int = Field(4, ge=1, le=20, description="Top-k passages to retrieve")

# These schemas ensure the LLM passes valid structured arguments to tools instead of free-form strings.

# ---------- Build agent ----------

# This function creates the agent
def build_agent(model: str = "gpt-4o-mini"):
    #  Uses the model name passed in (e.g. gpt-4o or gpt-4o-mini).
    # Low temperature → deterministic, concise answers.
    llm = ChatOpenAI(model=model, temperature=0.2)

    # Each tool is wrapped as a StructuredTool
    # tool choice is influenced by:
    # System prompt (high-level policy).
    # Tool names (e.g., search_docs, send_email).
    # Tool descriptions (what the tool is for).
    # Args schema (what inputs are expected).
    tools = [
        StructuredTool.from_function(
            name="save_text",
            func=lambda text, tag="note": save_text_tool(text=text, tag=tag),
            args_schema=SaveTextInput,
            description="Store snippets or FAQ candidates."
        ),
        # A mock external API, currently supports GET /orders/{id}.
        StructuredTool.from_function(
            name="call_api",
            func=lambda endpoint, method="GET": call_api_tool(endpoint=endpoint, method=method),
            args_schema=CallApiInput,
            description="Mock external API. Supports GET /orders/{id}."
        ),
        # Queues emails in a JSONL outbox.
        StructuredTool.from_function(
            name="send_email",
            func=lambda to, subject, body: send_email_tool(to=to, subject=subject, body=body),
            args_schema=SendEmailInput,
            description="Escalate complex or low-confidence cases."
        ),
        # Uses the RAG vector store to retrieve relevant passages and concatenate them.
        StructuredTool.from_function(
            name="search_docs",
            func=lambda query, k=4: "\n\n".join(similarity_search(query=query, k=int(k))),
            args_schema=SearchDocsInput,
            description="Retrieve relevant passages from the Chroma knowledge base (RAG)."
        ),
    ]

    # Stores previous messages under chat_history, which will be injected into the prompts.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Uses OpenAI function calling style.
    # system_message sets the behavior (tool choice, escalation, etc.).
    # extra_prompt_messages = the chat history placeholder.
    # verbose=True logs the tool calls and thought process at the console.
    # Returns an AgentExecutor you can .run() or .invoke().
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
