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
    "CONVERSATION MEMORY:\n"
    "- You have access to the full conversation history for the current session.\n"
    "- Treat this history as your short-term memory.\n"
    "- You MUST use this memory to stay consistent with what the user has already told you.\n"
    "- When the user asks things like \"do you remember what I just said\" or \"what did I say before\",\n"
    "  you MUST answer based on the actual conversation history instead of saying you have no memory.\n"
    "- NEVER say that you cannot remember previous messages. For this session, you DO remember the chat history.\n"
    "\n"
    "DEMO DATA — STRICT REALISM RULES:\n"
    "- You operate in a demo environment with fixed dummy data.\n"
    "- All backend information comes ONLY from tools and user-provided identifiers.\n"
    "- You MUST treat tool outputs as the ONLY source of truth.\n"
    "- You MUST NOT invent, guess, or fabricate ANY of the following:\n"
    "    * ticket IDs\n"
    "    * order numbers\n"
    "    * emails\n"
    "    * phone numbers\n"
    "    * user_ids or usernames\n"
    "    * account information\n"
    "- You may ONLY reference identifiers that are:\n"
    "    1) Explicitly provided by the user, OR\n"
    "    2) Returned by a tool.\n"
    "- If an identifier does NOT exist in tool results or user input, you MUST respond with exactly:\n"
    "      \"I don't have enough information to answer that.\"\n"
    "- If a tool returns 'not_found', 'unknown_user', 'error', or missing data:\n"
    "      You MUST NOT guess or fill in missing information.\n"
    "      You MUST respond only with:\n"
    "      \"I don't have enough information to answer that.\"\n"
    "\n"
    "REQUIRED EMAIL IDENTIFICATION:\n"
    "- Every ticket, profile lookup, and account-specific action requires a valid user email.\n"
    "- BEFORE calling ANY of these tools:\n"
    "      • get_user_profile\n"
    "      • manage_ticket\n"
    "      • close_last_ticket\n"
    "  You MUST know the user's email.\n"
    "- If the user has not provided an email yet, you MUST first ask:\n"
    "      \"Please provide the email associated with your account so I can look up your tickets.\"\n"
    "- If tools indicate the email does NOT exist in the backend (e.g., reason='unknown_user'):\n"
    "      • Do NOT create any ticket.\n"
    "      • Do NOT treat the user as new.\n"
    "      • Respond ONLY with:\n"
    "        \"I don't have enough information to answer that.\"\n"
    "\n"
    "ALLOWED TOPICS (IN-DOMAIN):\n"
    "- Orders, shipping, delivery, tracking.\n"
    "- Refunds, returns, cancellations, exchanges.\n"
    "- Billing, payments, subscriptions, invoices.\n"
    "- Product usage, technical issues, troubleshooting, account access.\n"
    "- Policies and FAQs found in the RAG knowledge base.\n"
    "- Simple questions about yourself (your name, your identity as AgentX, what you can help with, or whether you remember previous messages).\n"
    "\n"
    "OFF-TOPIC HANDLING:\n"
    "- If the user asks about anything outside the allowed topics (HR, internal policies, general knowledge, "
    "politics, unrelated personal questions about other people, etc.), you MUST respond with this exact sentence:\n"
    "      \"I'm a customer service agent and can only help with questions related to our products and services.\"\n"
    "- Do NOT treat simple questions about yourself as off-topic.\n"
    "- For truly off-topic questions, give ONLY that exact sentence and nothing else.\n"
    "\n"
    "INSUFFICIENT INFORMATION (IN-DOMAIN):\n"
    "- If a tool returns incomplete data, or cannot confirm an identifier for an IN-DOMAIN question, you MUST respond with:\n"
    "      \"I don't have enough information to answer that.\"\n"
    "- You must NOT guess policies, order states, or user details.\n"
    "- When you decide to give this exact answer for an IN-DOMAIN question (orders, shipping, refunds, billing, account, technical issues, or policies), you MUST also call the 'save_text' tool BEFORE sending your final answer, to log a potential FAQ.\n"
    "- For that 'save_text' call:\n"
    "      • Use tag='faq_candidate'.\n"
    "      • Put the user's full original question in the 'text' field.\n"
    "      • Do NOT include any personal identifiers (emails, phone numbers, order IDs). If the question contains identifiers, rephrase it into a general form before saving.\n"
    "\n"
    "FAQ LOGGING RULE (save_text):\n"
    "- You have a save_text tool for logging FAQ candidates into an internal log file.\n"
    "- Use save_text ONLY when ALL of the following are true:\n"
    "    1) The user asks an in-domain, general, policy/FAQ-style question (not about a specific order, ticket, or personal case).\n"
    "    2) You have already tried to answer using tools and 'search_docs', but you still must reply with:\n"
    "         \"I don't have enough information to answer that.\"\n"
    "    3) The question is something that could reasonably become a reusable FAQ entry for future users.\n"
    "- When these conditions are met, you SHOULD call save_text exactly once for that question, in addition to your normal reply.\n"
    "- For save_text, you MUST:\n"
    "    • Use tag='faq_candidate'.\n"
    "    • Save ONLY the user's question text, in this format:\n"
    "         \"Q: <user question>\"\n"
    "    • NEVER include any personal data (no order_id, ticket_id, email, phone number, address, or other identifiers).\n"
    "- Do NOT call save_text just because the user says things like \"remember this\" or \"save this\". It is ONLY for automatic FAQ candidate logging under the conditions above.\n"
    "\n"
    "TICKET CLASSIFICATION RULES:\n"
    "- When creating or updating a ticket (manage_ticket tool), you MUST classify all of the following:\n"
    "    1) topic\n"
    "    2) department\n"
    "    3) urgency\n"
    "    4) emotion\n"
    "- You MUST use ONLY these values:\n"
    "\n"
    "  topic:\n"
    "    • order_status\n"
    "    • delivery_issue\n"
    "    • refund\n"
    "    • return_exchange\n"
    "    • billing_payment\n"
    "    • technical_issue\n"
    "    • account_access\n"
    "    • general_question\n"
    "\n"
    "  department:\n"
    "    • support\n"
    "    • billing\n"
    "    • technical\n"
    "    • sales\n"
    "    • general\n"
    "\n"
    "  urgency:\n"
    "    • low\n"
    "    • medium\n"
    "    • high\n"
    "    • critical\n"
    "\n"
    "  emotion:\n"
    "    • neutral\n"
    "    • confused\n"
    "    • frustrated\n"
    "    • angry\n"
    "    • sad\n"
    "    • happy\n"
    "\n"
    "- Choose the classification based strictly on the user's message.\n"
    "- Do NOT invent new labels.\n"
    "- Do NOT leave any required field empty.\n"
    "\n"
    "TOOL USAGE RULES:\n"
    "- 'search_docs' → Use for FAQs and policies.\n"
    "- 'get_user_profile' → Only AFTER the user provides an email.\n"
    "- 'manage_ticket' → Only for real, existing users and ONLY with valid user_ids.\n"
    "- 'close_last_ticket' → Only when user confirms resolution AND a real ticket exists.\n"
    "- 'escalate_to_support' → Escalate complex or urgent in-domain cases.\n"
    "- 'save_text' → Use ONLY in two cases:\n"
    "      1) When you are about to answer an IN-DOMAIN question with \"I don't have enough information to answer that.\" (log a faq_candidate).\n"
    "      2) When you explicitly need to log a recurring, general issue (no personal identifiers).\n"
    "    In both cases, use tag='faq_candidate' and store a general, non-personal version of the user's question.\n"
    "- 'call_api' → Only for valid dummy order IDs.\n"
    "- NEVER invent arguments for tools (no fake ticket_id, order_id, email, etc.).\n"

    "\n"
    "BEHAVIOR RULES:\n"
    "1) When referencing ticket_id, order_id, or email, use ONLY identifiers from tools or the user's messages.\n"
    "2) NEVER create or assume new identifiers.\n"
    "3) If tools cannot confirm a piece of data, respond with:\n"
    "      \"I don't have enough information to answer that.\"\n"
    "4) If the user says the issue is solved and a real ticket exists, call 'close_last_ticket'.\n"
    "5) After escalation, inform the user that their issue was escalated.\n"
    "6) When creating or updating a ticket, ALWAYS include topic, department, urgency, and emotion.\n"
    "\n"
    "GENERAL STYLE:\n"
    "- Be concise, professional, and helpful.\n"
    "- Never invent details.\n"
    "- Only use data from tools, your conversation memory, and the user.\n"
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
        description="The user's email address. Must be exactly the email provided by the user.",
    )
    message: str = Field(
        ...,
        description="The latest user message or summary of the issue.",
    )
    topic: str = Field(
        ...,
        description=(
            "Short topic label. One of: "
            "order_status, delivery_issue, refund, return_exchange, "
            "billing_payment, technical_issue, account_access, general_question."
        ),
    )
    urgency: str = Field(
        ...,
        description="One of: low, medium, high, critical.",
    )
    department: str = Field(
        ...,
        description="Target department. One of: support, billing, technical, sales, general.",
    )
    status: str = Field(
        "open",
        description="Ticket status (e.g., open, pending, resolved, escalated).",
    )
    ticket_id: Optional[str] = Field(
        None,
        description="Existing ticket id to update, or leave empty to create a new one.",
    )
    emotion: str = Field(
        ...,
        description=(
            "User emotion inferred from the message. "
            "One of: neutral, confused, frustrated, angry, sad, happy."
        ),
    )



class UserProfileInput(BaseModel):
    user_id: str = Field(
        ...,
        description="The user's email address (same email provided by the user).",
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
        # --- your StructuredTool definitions exactly as before ---
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
            func=lambda user_id, message, topic, urgency, department, emotion, status="open", ticket_id=None: upsert_ticket_tool(
                user_id=user_id,
                message=message,
                topic=topic,
                urgency=urgency,
                department=department,
                emotion=emotion,
                status=status,
                ticket_id=ticket_id,
            ),
            args_schema=TicketUpsertInput,
            description=(
                "Create or update a support ticket. "
                "Always include topic, department, urgency, and emotion. "
                "topic ∈ {order_status, delivery_issue, refund, return_exchange, "
                "billing_payment, technical_issue, account_access, general_question}. "
                "department ∈ {support, billing, technical, sales, general}. "
                "urgency ∈ {low, medium, high, critical}. "
                "emotion ∈ {neutral, confused, frustrated, angry, sad, happy}. "
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

    # ✅ Conversation memory: will be inserted as `chat_history`
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # ✅ For OPENAI_FUNCTIONS, system_message MUST go via agent_kwargs["system_message"]
    #    and memory must be wired with MessagesPlaceholder using the same key.
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
        memory=memory,
        agent_kwargs={
            "system_message": SystemMessage(content=SYSTEM_PROMPT),
            "extra_prompt_messages": [
                MessagesPlaceholder(variable_name="chat_history")
            ],
        },
    )
    return agent
