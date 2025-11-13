
import json
import os
import time
from typing import Any, Dict

LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "logs.jsonl")
OUTBOX_PATH = os.path.join(os.path.dirname(__file__), "..", "outbox", "emails.jsonl")

def _ensure_dirs():
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(OUTBOX_PATH), exist_ok=True)

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

def send_email_tool(to: str, subject: str, body: str, meta: Dict[str, Any] | None = None) -> str:
    # \"\"\"Escalate by sending an 'email' (append to outbox/emails.jsonl).\"\"\"
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
