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
OUTBOX_PATH = os.path.join(OUTBOX_DIR, "emails.jsonl")

# Mock data directory (static dummy backend data)
MOCK_DATA_DIR = os.path.join(BASE_DIR, "..", "mock_data")
ORDERS_PATH = os.path.join(MOCK_DATA_DIR, "orders.json")
USERS_PATH = os.path.join(MOCK_DATA_DIR, "users.json")

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


# In-memory cache for dummy orders
_DUMMY_ORDERS_CACHE: Dict[str, Any] | None = None


def _load_dummy_orders() -> Dict[str, Any]:
    """
    Load dummy orders from mock_data/orders.json into memory.
    The JSON file should map order_id -> { ...fields... }.
    Example:
        {
          "1001": {"state": "processing"},
          "1002": {"state": "shipped"}
        }
    """
    global _DUMMY_ORDERS_CACHE
    if _DUMMY_ORDERS_CACHE is not None:
        return _DUMMY_ORDERS_CACHE

    if not os.path.exists(ORDERS_PATH):
        # No mock orders file; return empty dict so calls fail gracefully
        _DUMMY_ORDERS_CACHE = {}
        return _DUMMY_ORDERS_CACHE

    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure keys are strings
        _DUMMY_ORDERS_CACHE = {str(k): v for k, v in data.items()}
    except Exception:
        _DUMMY_ORDERS_CACHE = {}

    return _DUMMY_ORDERS_CACHE

# In-memory cache for dummy users (keyed by email)
_DUMMY_USERS_CACHE: Dict[str, Any] | None = None


def _load_dummy_users() -> Dict[str, Any]:
    """
    Load dummy users from mock_data/users.json into memory.

    The JSON should map email -> profile dict, e.g.:
        {
          "alice@example.com": {"name": "Alice", "segment": "vip"},
          "bob@example.com":   {"name": "Bob", "segment": "standard"}
        }
    """
    global _DUMMY_USERS_CACHE
    if _DUMMY_USERS_CACHE is not None:
        return _DUMMY_USERS_CACHE

    if not os.path.exists(USERS_PATH):
        _DUMMY_USERS_CACHE = {}
        return _DUMMY_USERS_CACHE

    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize keys to lowercase emails
        _DUMMY_USERS_CACHE = {str(k).strip().lower(): v for k, v in data.items()}
    except Exception:
        _DUMMY_USERS_CACHE = {}

    return _DUMMY_USERS_CACHE



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
    Mock external API call using static JSON data in mock_data/.

    - GET /orders/{id} => looks up the order in mock_data/orders.json.
      If the order_id is not present, returns status='not_found'.
    """
    method = method.upper()

    # ---- Orders lookup ----
    if method == "GET" and endpoint.startswith("/orders/"):
        orders = _load_dummy_orders()
        order_id = endpoint.split("/")[-1]

        if order_id in orders:
            # Merge base fields with whatever is in the JSON
            order_data = {"order_id": order_id}
            extra = orders.get(order_id) or {}
            if isinstance(extra, dict):
                order_data.update(extra)

            return {
                "status": "ok",
                "endpoint": endpoint,
                "data": order_data,
            }

        # Order does not exist in dummy data
        return {
            "status": "not_found",
            "endpoint": endpoint,
            "message": "Order not found in demo data.",
        }

    # ---- Unknown endpoint ----
    return {
        "status": "error",
        "message": f"Unknown endpoint: {method} {endpoint}",
        "payload": payload or {},
    }



def send_email_tool(
    subject: str,
    body: str,
    to: str = SUPPORT_EMAIL,
    meta: Dict[str, Any] | None = None,
) -> str:
    """
    Escalate by sending an 'email' (append to outbox/emails.jsonl).

    NOTE:
    - All support escalations use the fixed SUPPORT_EMAIL by default.
    - The 'to' parameter is optional; if omitted, SUPPORT_EMAIL is used.
    """
    _ensure_dirs()
    record = {
        "ts": time.time(),
        "to": to,
        "subject": subject,
        "body": body,
        "meta": meta or {},
    }
    with open(OUTBOX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"Queued email to {to} with subject='{subject}'."

# --------------------------------------------------------------------
# Ticketing helpers
# --------------------------------------------------------------------

def _read_ticket_events() -> List[Dict[str, Any]]:
    """
    Read all ticket events from tickets.jsonl.
    Returns an empty list if the file does not exist or is unreadable.
    """
    _ensure_dirs()
    events: List[Dict[str, Any]] = []

    if not os.path.exists(TICKETS_PATH):
        return events

    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(rec)
    return events


def _generate_ticket_id() -> str:
    """
    Generate a new, deterministic ticket_id of the form 'T-000001', 'T-000002', ...

    It scans existing ticket_ids in tickets.jsonl, finds the maximum numeric suffix,
    and returns the next one. This keeps IDs stable and non-random for the demo.
    """
    events = _read_ticket_events()
    prefix = "T-"
    max_num = 0

    for rec in events:
        tid = str(rec.get("ticket_id", ""))
        if not tid.startswith(prefix):
            continue
        suffix = tid[len(prefix):]
        try:
            n = int(suffix)
        except ValueError:
            continue
        if n > max_num:
            max_num = n

    return f"{prefix}{max_num + 1:06d}"  # e.g., T-000001


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
    emotion: str | None = None,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Create or update a support ticket linked to a known user email.

    - user_id is treated as the user's email address.
    - The email MUST exist in mock_data/users.json; otherwise, no ticket is created.

    Ticket events are stored line-by-line in storage/tickets.jsonl with the schema:
        {
          "schema": 1,
          "type": "ticket_event",
          "ts": <float>,
          "ticket_id": "T-000001",
          "user_id": "<user_email>",
          "message": "<latest user message or summary>",
          "topic": "<topic label>",
          "urgency": "low|medium|high|critical",
          "department": "<billing|support|technical|sales|general>",
          "status": "open|pending|resolved|escalated|closed",
          "emotion": "neutral|confused|frustrated|angry|sad|happy",
          "event": "created|updated",
          "meta": {...}
        }

    """
    _ensure_dirs()
    now = time.time()

    # Validate user email against mock_data/users.json
    users = _load_dummy_users()
    email = str(user_id).strip().lower()
    if email not in users:
        return {
            "ok": False,
            "reason": "unknown_user",
            "message": "User email not found in demo data.",
        }

    is_new = ticket_id is None
    if is_new:
        ticket_id = _generate_ticket_id()

    record: Dict[str, Any] = {
        "schema": 1,
        "type": "ticket_event",
        "ts": now,
        "ticket_id": ticket_id,
        "user_id": email,  # store normalized email
        "message": message,
        "topic": topic,
        "urgency": urgency,
        "department": department,
        "status": status,
        "emotion": emotion,
        "meta": meta or {},
        "event": "created" if is_new else "updated",
    }

    _append_jsonl(TICKETS_PATH, record)

    # Auto-escalate high/critical tickets via email
    if urgency in {"high", "critical"} and status not in {"resolved", "closed"}:
        subject = f"[AgentX] {urgency.upper()} ticket {ticket_id} for {email}"
        body_lines = [
            f"Ticket ID: {ticket_id}",
            f"User: {email}",
            f"Topic: {topic}",
            f"Urgency: {urgency}",
            f"Emotion: {emotion}",
            "",
            "Latest message:",
            message,
        ]
        body = "\n".join(body_lines)

        # This uses the same send_email_tool the agent tool wraps
        send_email_tool(
            subject=subject,
            body=body,
            to=SUPPORT_EMAIL,
            meta={
                "auto_escalated": True,
                "source": "upsert_ticket_tool",
                "operation": "created" if is_new else "updated",
            },
        )

    return {
        "ok": True,
        "ticket_id": ticket_id,
        "operation": "created" if is_new else "updated",
        "topic": topic,
        "urgency": urgency,
        "department": department,
        "status": status,
        "emotion": emotion,
        "user_email": email,
    }






def get_user_profile_tool(user_id: str) -> Dict[str, Any]:
    """
    Inspect tickets.jsonl and derive a simple user profile based on email.

    - user_id is treated as the user's email address.
    - If the email is not present in mock_data/users.json, the user is considered unknown.

    Returns:
    - ok: bool (True if user is known, False otherwise)
    - reason: str | None (e.g. 'unknown_user' on failure)
    - user_id: the normalized email
    - known_user: bool (email exists in users.json)
    - is_new_user: bool (known user but no tickets yet)
    - has_open_tickets: bool
    - total_tickets: int (unique ticket_ids for this user)
    - open_tickets: int (unique tickets whose latest status is not resolved/closed)
    - last_ticket_ts: float | None
    - profile: dict | None (raw profile from users.json, if any)
    """
    email = str(user_id).strip().lower()
    users = _load_dummy_users()
    profile = users.get(email)

    events = _read_ticket_events()
    user_events = [
        e for e in events
        if str(e.get("user_id", "")).strip().lower() == email
    ]

    # Unknown user in dummy backend
    if profile is None:
        return {
            "ok": False,
            "reason": "unknown_user",
            "message": "Email not found in user data database.",
            "user_id": email,
            "known_user": False,
            "is_new_user": True,
            "has_open_tickets": False,
            "total_tickets": 0,
            "open_tickets": 0,
            "last_ticket_ts": None,
            "profile": None,
        }

    # Known user, but no tickets yet
    if not user_events:
        return {
            "ok": True,
            "reason": None,
            "user_id": email,
            "known_user": True,
            "is_new_user": True,
            "has_open_tickets": False,
            "total_tickets": 0,
            "open_tickets": 0,
            "last_ticket_ts": None,
            "profile": profile,
        }

    # Track latest status per ticket_id
    latest_by_ticket: Dict[str, Dict[str, Any]] = {}
    for e in user_events:
        tid = str(e.get("ticket_id"))
        ts = e.get("ts", 0)
        prev = latest_by_ticket.get(tid)
        if prev is None or ts > prev.get("ts", 0):
            latest_by_ticket[tid] = e

    total_tickets = len(latest_by_ticket)
    open_tickets = sum(
        1
        for e in latest_by_ticket.values()
        if e.get("status") not in {"resolved", "closed"}
    )
    last_ticket_ts = max(e.get("ts", 0) for e in user_events) if user_events else None

    return {
        "ok": True,
        "reason": None,
        "user_id": email,
        "known_user": True,
        "is_new_user": total_tickets == 0,
        "has_open_tickets": open_tickets > 0,
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "last_ticket_ts": last_ticket_ts,
        "profile": profile,
    }


# --------------------------------------------------------------------
# Deterministic ticket closing helper
# --------------------------------------------------------------------


def close_last_open_ticket_tool(user_id: str) -> Dict[str, Any]:
    """
    Close the most recent non-resolved ticket for a specific user.

    - user_id is treated as the user's email address.
    - The email MUST exist in mock_data/users.json; otherwise, no ticket is closed.

    Behavior:
    - Scan tickets.jsonl via _read_ticket_events()
    - Filter to events for this user
    - Find the latest record whose status is not in {'resolved', 'closed'}
    - Append a new record with status='resolved' and event='updated'
    """
    email = str(user_id).strip().lower()

    # Ensure this is a known user in the dummy backend
    users = _load_dummy_users()
    if email not in users:
        return {
            "ok": False,
            "reason": "unknown_user",
            "message": "User email not found in demo data.",
            "user_id": email,
        }

    events = _read_ticket_events()
    if not events:
        return {
            "ok": False,
            "reason": "no_tickets_file",
            "message": "No tickets file found.",
            "user_id": email,
        }

    # Filter events for this user only
    user_events: List[Dict[str, Any]] = [
        e for e in events
        if str(e.get("user_id", "")).strip().lower() == email
    ]

    if not user_events:
        return {
            "ok": False,
            "reason": "no_tickets_for_user",
            "message": "No tickets found for this user.",
            "user_id": email,
        }

    last_open_record: Dict[str, Any] | None = None

    for rec in user_events:
        status = rec.get("status")
        if status in {"resolved", "closed"}:
            continue

        # keep the latest open/non-resolved record for this user
        if last_open_record is None or rec.get("ts", 0) > last_open_record.get("ts", 0):
            last_open_record = rec

    if last_open_record is None:
        return {
            "ok": False,
            "reason": "no_open_ticket",
            "message": "No open tickets found for this user.",
            "user_id": email,
        }

    now = time.time()
    ticket_id = last_open_record.get("ticket_id")
    topic = last_open_record.get("topic")
    department = last_open_record.get("department")

    # Preserve urgency & emotion from the last open record
    urgency = last_open_record.get("urgency", "low")
    emotion = last_open_record.get("emotion", "neutral")

    updated = {
        "schema": 1,
        "type": "ticket_event",
        "ts": now,
        "ticket_id": ticket_id,
        "user_id": email,
        "message": "Ticket closed by close_last_open_ticket_tool.",
        "topic": topic,
        "urgency": urgency,
        "department": department,
        "status": "resolved",
        "emotion": emotion,
        "meta": {
            **(last_open_record.get("meta") or {}),
            "auto_closed": True,
        },
        "event": "updated",
    }
    _append_jsonl(TICKETS_PATH, updated)

    return {
        "ok": True,
        "ticket_id": ticket_id,
        "status": "resolved",
        "closed_ts": now,
        "user_id": email,
    }



