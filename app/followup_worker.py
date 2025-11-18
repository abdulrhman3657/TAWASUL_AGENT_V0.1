# app/followup_worker.py

import os
import time
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .tools import _read_ticket_events, send_email_tool, _load_dummy_users

load_dotenv(override=True)

# How “old” a ticket must be (seconds since last event) to get a follow-up
THRESHOLD_SECONDS = 24 * 60 * 60  # 24 hours


def _get_open_tickets_by_user() -> List[Dict[str, Any]]:
    """
    Return a list of latest open ticket records (one per ticket_id),
    each enriched with user info if available.
    """
    events = _read_ticket_events()
    if not events:
        return []

    # latest event per ticket_id
    latest_by_ticket: Dict[str, Dict[str, Any]] = {}
    for e in events:
        tid = str(e.get("ticket_id"))
        ts = e.get("ts", 0)
        prev = latest_by_ticket.get(tid)
        if prev is None or ts > prev.get("ts", 0):
            latest_by_ticket[tid] = e

    # Keep only open tickets
    open_tickets = [
        rec for rec in latest_by_ticket.values()
        if rec.get("status") not in {"resolved", "closed"}
    ]

    # Attach basic user profile if exists
    users = _load_dummy_users()
    for rec in open_tickets:
        email = str(rec.get("user_id", "")).strip().lower()
        rec["user_profile"] = users.get(email, {})
    return open_tickets


def _filter_stale_tickets(tickets: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
    """
    Filter tickets whose last event is older than THRESHOLD_SECONDS.
    """
    stale = []
    for rec in tickets:
        ts = rec.get("ts", 0)
        if now - ts >= THRESHOLD_SECONDS:
            stale.append(rec)
    return stale


def _build_followup_email_body(llm: ChatOpenAI, ticket: Dict[str, Any]) -> str:
    """
    Use the LLM to generate a short, friendly follow-up email body
    based on the latest ticket event.
    """
    user_email = str(ticket.get("user_id", "")).strip().lower()
    ticket_id = ticket.get("ticket_id", "UNKNOWN")
    topic = ticket.get("topic", "general_question")
    last_message = ticket.get("message", "")
    status = ticket.get("status", "open")

    profile = ticket.get("user_profile") or {}
    user_name = profile.get("name") or user_email or "customer"

    system_prompt = (
        "You are AgentX, a customer support assistant. "
        "You are writing short, polite follow-up emails to customers "
        "about their open support tickets.\n"
        "- Be concise and friendly.\n"
        "- Do NOT invent any details about orders or policies.\n"
        "- Ask if the issue is resolved or if they still need help.\n"
        "- Do NOT mention internal systems, tools, or logs.\n"
    )

    user_prompt = (
        f"Customer name: {user_name}\n"
        f"Customer email: {user_email}\n"
        f"Ticket ID: {ticket_id}\n"
        f"Topic: {topic}\n"
        f"Current ticket status: {status}\n"
        f"Last recorded customer message:\n"
        f"\"{last_message}\"\n\n"
        "Write a short follow-up email (plain text, no subject line) to the customer, "
        "asking if their issue has been resolved or if they still need assistance."
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return resp.content if hasattr(resp, "content") else str(resp)


def run_followup_once():
    """
    One pass of the follow-up logic:
      - Load open tickets
      - Filter to stale ones
      - Generate follow-up emails and queue them via send_email_tool
    """
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    now = time.time()
    open_tickets = _get_open_tickets_by_user()
    stale_tickets = _filter_stale_tickets(open_tickets, now=now)

    if not stale_tickets:
        print("No stale open tickets found. Nothing to do.")
        return

    print(f"Found {len(stale_tickets)} stale open tickets. Sending follow-ups...")

    for ticket in stale_tickets:
        user_email = str(ticket.get("user_id", "")).strip().lower()
        ticket_id = ticket.get("ticket_id", "UNKNOWN")

        if not user_email or "@" not in user_email:
            print(f"Skipping ticket {ticket_id}: invalid user email.")
            continue

        try:
            body = _build_followup_email_body(llm, ticket)
            subject = f"Checking in about your ticket {ticket_id}"

            result_msg = send_email_tool(
                subject=subject,
                body=body,
                to=user_email,  # send directly to the customer email in the demo outbox
                meta={
                    "source": "followup_worker",
                    "ticket_id": ticket_id,
                    "reason": "stale_open_ticket",
                },
            )
            print(f"[OK] Follow-up queued for {user_email}, ticket {ticket_id}: {result_msg}")
        except Exception as e:
            print(f"[ERR] Failed to send follow-up for ticket {ticket_id}: {e}")


if __name__ == "__main__":
    run_followup_once()
