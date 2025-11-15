# app/agent.py
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain.tools import StructuredTool

from .tools import (
    save_text_tool,
    call_api_tool,
    send_email_tool,
    upsert_ticket_tool,
    get_user_profile_tool,
    record_analytics_tool,
    close_last_open_ticket_tool,
    SUPPORT_EMAIL,
)
from .rag import similarity_search

load_dotenv()


SYSTEM_PROMPT = (
    "You are AgentX, an autonomous customer support coordinator.\n"
    "\n"
    "LANGUAGE:\n"
    "- You understand Arabic and English.\n"
    "- Always reply in the same language the user is using.\n"
    "\n"
    "ALLOWED TOPICS (IN-DOMAIN):\n"
    "- Orders, shipping, delivery, and tracking.\n"
    "- Refunds, returns, cancellations, and exchanges.\n"
    "- Billing, payments, subscriptions, and invoices.\n"
    "- Product usage, technical issues, troubleshooting, and account access.\n"
    "- Questions directly related to the official policies and FAQs in the knowledge base.\n"
    "\n"
    "OFF-TOPIC HANDLING (OUT-OF-DOMAIN):\n"
    "- If the user asks about anything outside the topics above (for example: office pet policy, HR questions, "
    "company internal rules not in the knowledge base, general knowledge, personal questions, politics, etc.), "
    "you MUST answer with exactly this sentence and NOTHING else:\n"
    "  \"I'm a customer service agent and can only help with questions related to our products and services.\"\n"
    "- Do NOT try to be helpful outside your domain. Do NOT explain, guess, or add extra text.\n"
    "\n"
    "UNKNOWN / INSUFFICIENT INFORMATION (IN-DOMAIN BUT UNSURE):\n"
    "- If the question IS about an allowed topic, but the tools and documents do not give you enough reliable "
    "information to answer confidently, you MUST respond with exactly this sentence and NOTHING else:\n"
    "  \"I don't have enough information to answer that.\"\n"
    "- You must NOT invent policies, numbers, or details. Never guess.\n"
    "\n"
    "TOOLS USAGE (ONLY FOR ALLOWED TOPICS):\n"
    "- Use 'search_docs' to answer questions from the knowledge base (RAG over FAISS).\n"
    "- Use 'manage_ticket' to create or update support tickets with topic, urgency, department, and status.\n"
    "- Use 'get_user_profile' to see if the user is new, returning, or has open tickets.\n"
    "- Use 'send_email' to escalate urgent or unclear in-domain cases to a human.\n"
    "- Use 'record_analytics' and 'save_text' to log outcomes and potential FAQ candidates.\n"
    "- Use 'call_api' only for relevant customer-service operations (e.g., order status lookups).\n"
    "- Never call tools for clearly off-topic questions; in that case, just reply with the off-topic sentence above.\n"
    "\n"
    "STRICT BEHAVIOR RULES:\n"
    "1) For every new serious issue, you MUST create a ticket using 'manage_ticket'.\n"
    "2) If the issue is critical, about money (billing/charges/refunds), or you are unsure "
    "   about the correct action, you MUST escalate using 'escalate_to_support'.\n"
    "3) When you escalate, your next action must be to call 'escalate_to_support' with a clear "
    "   subject and body that includes the ticket id and a short summary.\n"
    "4) After escalation, clearly tell the user that their issue was escalated to human support.\n"
    "5) Whenever the user says 'close the ticket' or clearly indicates the issue is solved, you MUST "
    "   call the 'close_last_ticket' tool to mark the most recent open ticket as resolved.\n"
    "\n"
    "GENERAL STYLE (ONLY WHEN YOU ARE ALLOWED TO ANSWER):\n"
    "- Be concise and focused on solving the user's issue.\n"
    "- Briefly explain what you did (classification, ticket id, actions taken) when answering in-domain questions.\n"
    "- Maintain a polite, professional customer-support tone.\n"
)



# ---------- Pydantic arg schemas ----------


class SaveTextInput(BaseModel):
    text: str = Field(..., description="The content or insight to save.")
    tag: str = Field(
        "note",
        description="Optional tag for classification (e.g., 'faq_candidate', 'bug', 'trend').",
    )


class CallApiInput(BaseModel):
    endpoint: str = Field(..., description="e.g., /orders/12345")
    method: str = Field(
        "GET",
        description="HTTP method, default GET.",
    )


class EscalateToSupportInput(BaseModel):
    subject: str = Field(..., description="Short subject line for the support team.")
    body: str = Field(
        ...,
        description="Email body text, include ticket id, user_id, and brief context."
    )



class SearchDocsInput(BaseModel):
    query: str = Field(..., description="Semantic search query in Arabic or English.")
    k: int = Field(
        4,
        ge=1,
        le=20,
        description="Top-k passages to retrieve from the knowledge base.",
    )


class TicketUpsertInput(BaseModel):
    user_id: str = Field(
        ...,
        description="Stable identifier for the user (email, phone, or customer id).",
    )
    message: str = Field(
        ...,
        description="The latest user message or summary of the issue.",
    )
    topic: str = Field(
        ...,
        description="Short topic label (e.g., 'refund', 'delivery_delay', 'technical_issue').",
    )
    urgency: str = Field(
        ...,
        description="One of: low, medium, high, critical.",
    )
    department: str = Field(
        ...,
        description="Target department (e.g., 'billing', 'support', 'technical', 'sales', 'general').",
    )
    status: str = Field(
        "open",
        description="Ticket status (e.g., open, pending, resolved, escalated).",
    )
    ticket_id: Optional[str] = Field(
        None,
        description="Existing ticket id to update, or leave empty to create a new one.",
    )


class UserProfileInput(BaseModel):
    user_id: str = Field(
        ...,
        description="Stable identifier for the user (same as passed to ticket tools).",
    )


class AnalyticsInput(BaseModel):
    ticket_id: str = Field(..., description="Ticket id for which to log analytics.")
    status: str = Field(
        ...,
        description="Final status of this interaction (e.g., 'resolved', 'escalated').",
    )
    escalated: bool = Field(
        ...,
        description="True if a human or email escalation was involved.",
    )
    response_time_sec: Optional[float] = Field(
        None,
        description="Optional response time in seconds, if available.",
    )
    rating: Optional[float] = Field(
        None,
        description="Optional satisfaction rating (1-5, etc.) provided by the user.",
    )


class CloseLastTicketInput(BaseModel):
    # Dummy field so the tool schema is non-empty; value is ignored.
    dummy: str = Field(
        "ok",
        description="Always 'ok'. No real arguments are needed; this just closes the last open ticket.",
    )


# ---------- Build agent ----------


def build_agent(model: str = "gpt-4o-mini"):
    # Multilingual OpenAI chat model (Arabic + English)
    llm = ChatOpenAI(model=model, temperature=0.2)

    tools = [
        StructuredTool.from_function(
            name="search_docs",
            func=lambda query, k=4: "\n\n".join(
                similarity_search(query=query, k=int(k))
            ),
            args_schema=SearchDocsInput,
            description=(
                "Retrieve relevant passages from the knowledge base (FAISS RAG). "
                "Use this to answer policy, FAQ, and how-to questions."
            ),
        ),
        StructuredTool.from_function(
            name="manage_ticket",
            func=lambda user_id, message, topic, urgency, department, status="open", ticket_id=None: upsert_ticket_tool(
                user_id=user_id,
                message=message,
                topic=topic,
                urgency=urgency,
                department=department,
                status=status,
                ticket_id=ticket_id,
            ),
            args_schema=TicketUpsertInput,
            description=(
                "Create or update a support ticket. "
                "Always include topic, urgency, and department. "
                "If updating, pass the existing ticket_id."
            ),
        ),
        StructuredTool.from_function(
            name="get_user_profile",
            func=lambda user_id: get_user_profile_tool(user_id=user_id),
            args_schema=UserProfileInput,
            description=(
                "Look up whether a user is new, returning, and whether they have open tickets, "
                "based on their user_id."
            ),
        ),
        StructuredTool.from_function(
            name="record_analytics",
            func=lambda ticket_id, status, escalated, response_time_sec=None, rating=None: record_analytics_tool(
                ticket_id=ticket_id,
                status=status,
                escalated=bool(escalated),
                response_time_sec=response_time_sec,
                rating=rating,
            ),
            args_schema=AnalyticsInput,
            description=(
                "Record analytics for a ticket: final status, whether it was escalated, "
                "optional response time, and optional rating."
            ),
        ),
        StructuredTool.from_function(
            name="close_last_ticket",
            func=lambda dummy="ok": close_last_open_ticket_tool(),
            args_schema=CloseLastTicketInput,
            description=(
                "Close the most recent open ticket by marking it as resolved. "
                "Use this whenever the user says things like 'close the ticket', "
                "'mark this as resolved', or clearly indicates the issue is solved. "
                "No ticket id is required."
            ),
        ),
        StructuredTool.from_function(
            name="save_text",
            func=lambda text, tag="note": save_text_tool(text=text, tag=tag),
            args_schema=SaveTextInput,
            description=(
                "Store snippets, FAQ candidates, or repeated issues. "
                "Use tag='faq_candidate' for potential new FAQ entries."
            ),
        ),
        StructuredTool.from_function(
            name="call_api",
            func=lambda endpoint, method="GET": call_api_tool(
                endpoint=endpoint, method=method
            ),
            args_schema=CallApiInput,
            description=(
                "Mock external API. Use for things like order status lookups "
                "(supports GET /orders/{id})."
            ),
        ),
        StructuredTool.from_function(
            name="escalate_to_support",
            func=lambda subject, body: send_email_tool(
                subject=subject,
                body=body,
                to=SUPPORT_EMAIL,
                meta={"channel": "agentx"},
            ),
            args_schema=EscalateToSupportInput,
            description=(
                "Escalate complex or urgent cases to human support by sending an email "
                f"to {SUPPORT_EMAIL}. Always include ticket_id and a short summary "
                "in the body."
            ),
        ),

    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
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
            "extra_prompt_messages": [
                MessagesPlaceholder(variable_name="chat_history")
            ],
        },
    )
    return agent
