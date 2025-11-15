# app/tools.py
import json
import os
import time
from typing import Any, Dict, List

# --------------------------------------------------------------------
# Paths & helpers
# --------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
STORAGE_DIR = os.path.join(BASE_DIR, "..", "storage")
OUTBOX_DIR = os.path.join(BASE_DIR, "..", "outbox")

LOGS_PATH = os.path.join(STORAGE_DIR, "logs.jsonl")
TICKETS_PATH = os.path.join(STORAGE_DIR, "tickets.jsonl")
ANALYTICS_PATH = os.path.join(STORAGE_DIR, "analytics.jsonl")
OUTBOX_PATH = os.path.join(OUTBOX_DIR, "emails.jsonl")

# Single support email for all escalations
SUPPORT_EMAIL = "support@tawasul31.com"

def _ensure_dirs() -> None:
    """Ensure storage + outbox folders exist."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(OUTBOX_DIR, exist_ok=True)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON record to a .jsonl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------
# Core tools: logs, mock API, email
# --------------------------------------------------------------------


def save_text_tool(
    text: str,
    tag: str = "note",
    meta: Dict[str, Any] | None = None,
) -> str:
    """
    Save conversation snippets, FAQ candidates, or insights to storage/logs.jsonl.

    Used for:
    - tracking common issues
    - suggesting new FAQs later
    - logging trends and bugs
    """
    _ensure_dirs()
    record = {
        "ts": time.time(),
        "tag": tag,
        "text": text,
        "meta": meta or {},
    }
    _append_jsonl(LOGS_PATH, record)
    return f"Saved text with tag='{tag}'."


def call_api_tool(
    endpoint: str,
    method: str = "GET",
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Mock external API call.

    Current behavior:
    - GET /orders/{id} => returns order status
      (odd IDs => 'processing', even IDs => 'shipped')
    """
    if method.upper() == "GET" and endpoint.startswith("/orders/"):
        order_id = endpoint.split("/")[-1]
        return {
            "status": "ok",
            "endpoint": endpoint,
            "data": {
                "order_id": order_id,
                "state": "processing"
                if order_id.isdigit() and int(order_id) % 2 == 1
                else "shipped",
            },
        }

    return {
        "status": "error",
        "message": f"Unknown endpoint: {method} {endpoint}",
        "payload": payload or {},
    }


def send_email_tool(
    to: str,
    subject: str,
    body: str,
    meta: Dict[str, Any] | None = None,
) -> str:
    """
    Escalate by sending an 'email' (append to outbox/emails.jsonl).

    Used for:
    - urgent / unclear cases
    - escalation to human support
    """
    _ensure_dirs()
    record = {
        "ts": time.time(),
        "to": to,
        "subject": subject,
        "body": body,
        "meta": meta or {},
    }
    _append_jsonl(OUTBOX_PATH, record)
    return f"Queued email to {to} with subject='{subject}'."


# --------------------------------------------------------------------
# Ticketing & routing
# --------------------------------------------------------------------


def upsert_ticket_tool(
    user_id: str,
    message: str,
    topic: str,
    urgency: str,
    department: str,
    status: str = "open",
    ticket_id: str | None = None,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Create or update a support ticket with topic, urgency, and routing.

    - If ticket_id is None => create new ticket.
    - Otherwise => append an update entry (audit log style).

    urgency: one of ["low", "medium", "high", "critical"]
    status:  e.g. ["open", "pending", "resolved", "escalated"]
    """
    _ensure_dirs()
    now = time.time()

    is_new = ticket_id is None
    if is_new:
        ticket_id = f"T{int(now * 1000)}"

    record = {
        "ts": now,
        "ticket_id": ticket_id,
        "user_id": user_id,
        "message": message,
        "topic": topic,
        "urgency": urgency,
        "department": department,
        "status": status,
        "meta": meta or {},
        "event": "created" if is_new else "updated",
    }
    _append_jsonl(TICKETS_PATH, record)

    return {
        "ticket_id": ticket_id,
        "operation": "created" if is_new else "updated",
        "topic": topic,
        "urgency": urgency,
        "department": department,
        "status": status,
    }


def get_user_profile_tool(user_id: str) -> Dict[str, Any]:
    """
    Inspect tickets.jsonl and derive a simple user profile:

    - is_new_user: bool
    - has_open_tickets: bool
    - total_tickets: int
    - open_tickets: int
    - last_ticket_ts: float | None
    """
    _ensure_dirs()
    if not os.path.exists(TICKETS_PATH):
        return {
            "user_id": user_id,
            "is_new_user": True,
            "has_open_tickets": False,
            "total_tickets": 0,
            "open_tickets": 0,
            "last_ticket_ts": None,
        }

    total = 0
    open_count = 0
    last_ts = None

    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("user_id") != user_id:
                continue

            total += 1
            if rec.get("status") not in {"resolved", "closed"}:
                open_count += 1

            ts = rec.get("ts")
            if isinstance(ts, (int, float)):
                last_ts = ts if last_ts is None else max(last_ts, ts)

    return {
        "user_id": user_id,
        "is_new_user": total == 0,
        "has_open_tickets": open_count > 0,
        "total_tickets": total,
        "open_tickets": open_count,
        "last_ticket_ts": last_ts,
    }


# --------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------


def record_analytics_tool(
    ticket_id: str,
    status: str,
    escalated: bool,
    response_time_sec: float | None = None,
    rating: float | None = None,
    meta: Dict[str, Any] | None = None,
) -> str:
    """
    Record basic analytics for a conversation / ticket:

    - status: 'resolved', 'escalated', etc.
    - escalated: whether a human was involved.
    - response_time_sec: optional latency metric.
    - rating: optional user satisfaction score (1–5, etc.).
    """
    _ensure_dirs()
    record = {
        "ts": time.time(),
        "ticket_id": ticket_id,
        "status": status,
        "escalated": escalated,
        "response_time_sec": response_time_sec,
        "rating": rating,
        "meta": meta or {},
    }
    _append_jsonl(ANALYTICS_PATH, record)
    return f"Recorded analytics for ticket {ticket_id} with status='{status}'."


# --------------------------------------------------------------------
# Deterministic ticket closing helper
# --------------------------------------------------------------------


def close_last_open_ticket_tool() -> Dict[str, Any]:
    """
    Close the most recent non-resolved ticket (no args required).

    Intended for simple demos / single-user flows:
    - Scan tickets.jsonl
    - Find the latest record whose status is not in {'resolved', 'closed'}
    - Append a new record with status='resolved' and event='updated'
    """
    _ensure_dirs()

    if not os.path.exists(TICKETS_PATH):
        return {"ok": False, "reason": "no_tickets_file"}

    last_open_record: Dict[str, Any] | None = None

    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            status = rec.get("status")
            if status in {"resolved", "closed"}:
                continue

            # keep the latest open/non-resolved record
            if last_open_record is None or rec.get("ts", 0) > last_open_record.get("ts", 0):
                last_open_record = rec

    if last_open_record is None:
        return {"ok": False, "reason": "no_open_ticket"}

    # Create a new "updated" record marking it resolved
    now = time.time()
    ticket_id = last_open_record.get("ticket_id")
    user_id = last_open_record.get("user_id")
    topic = last_open_record.get("topic")
    department = last_open_record.get("department")

    updated = {
        "ts": now,
        "ticket_id": ticket_id,
        "user_id": user_id,
        "message": "Ticket closed by close_last_open_ticket_tool.",
        "topic": topic,
        "urgency": "low",
        "department": department,
        "status": "resolved",
        "meta": {"auto_closed": True},
        "event": "updated",
    }
    _append_jsonl(TICKETS_PATH, updated)

    return {
        "ok": True,
        "ticket_id": ticket_id,
        "status": "resolved",
        "closed_ts": now,
    }
