# These are the concrete implementations of the tools the agent calls.
import json
import os
import time
from typing import Any, Dict

# Paths
LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "logs.jsonl")
OUTBOX_PATH = os.path.join(os.path.dirname(__file__), "..", "outbox", "emails.jsonl")

# creates these parent directories if needed.
def _ensure_dirs():
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTBOX_PATH), exist_ok=True)

# Ensures directories exist.
# Creates a record:
# {
#     "ts": time.time(),
#     "tag": tag,
#     "text": text,
#     "meta": meta or {},
# }
# Appends it as a JSON line to logs.jsonl.
# Returns a friendly status string: "Saved text with tag='...'."
# The agent can use this to log snippets, FAQs, or notes from conversations.
def save_text_tool(text: str, tag: str = "note", meta: Dict[str, Any] | None = None) -> str:
    # "\"\"Save conversation snippets, FAQ candidates, or insights to storage/logs.jsonl.\"\"\"
    _ensure_dirs()
    record = {
        "ts": time.time(),
        "tag": tag,
        "text": text,
        "meta": meta or {},
    }
    with open(LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return f"Saved text with tag='{tag}'."

# Mock external API:
# If method == "GET" and endpoint starts with /orders/:
# Extracts order_id.
# Returns:
# Odd order IDs → "processing".
# Even order IDs → "shipped".
# Otherwise returns an error dict: {"status": "error", "message": "Unknown endpoint: ..."}.
# This lets the agent simulate API calls for order status queries.
def call_api_tool(endpoint: str, method: str = "GET", payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # \"\"\"Mock external API call. Supports GET /orders/{id}.\"\"\"
    if method.upper() == "GET" and endpoint.startswith("/orders/"):
        order_id = endpoint.split("/")[-1]
        return {
            "status": "ok",
            "endpoint": endpoint,
            "data": {
                "order_id": order_id,
                "state": "processing" if order_id.isdigit() and int(order_id) % 2 == 1 else "shipped"
            }
        }
    return {"status": "error", "message": f"Unknown endpoint: {method} {endpoint}"}

# Ensures directories exist.
# Appends a JSONL record to outbox/emails.jsonl:
# {
#     "ts": time.time(),
#     "to": to,
#     "subject": subject,
#     "body": body,
#     "meta": meta or {},
# }
# Returns a status message: Queued email to {to} with subject='...'.
# This is the “escalation” mechanism the system prompt refers to.
def send_email_tool(to: str, subject: str, body: str, meta: Dict[str, Any] | None = None) -> str:
    # \"\"\"Escalate by sending an 'email' (append to outbox/emails.jsonl).\"\"\"
    _ensure_dirs()

    to = "support@tawasul31.com"

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
