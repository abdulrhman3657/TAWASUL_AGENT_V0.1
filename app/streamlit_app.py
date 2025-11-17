# This is a chat UI using Streamlit, with a bit of path-shim magic to make imports robust.
import os
import json
import time
import uuid
import streamlit as st

# --- robust import path shim (works no matter your cwd) ---
import sys
from pathlib import Path

FILE = Path(__file__).resolve()     # .../project1/app/streamlit_app.py
ROOT = FILE.parents[1]              # .../project1
APP_DIR = ROOT / "app"              # .../project1/app

# ensure project root and app/ are importable
for p in (str(ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# defensive sanity checks (will raise a clear error if something is off)
assert (APP_DIR / "__init__.py").exists(), f"Missing __init__.py at {APP_DIR}"
assert (APP_DIR / "agent.py").exists(), f"Missing agent.py at {APP_DIR}"
# --- end shim ---

from app.agent import build_agent
from app.rag import build_vectorstore
from app.tools import LOGS_PATH, OUTBOX_PATH, TICKETS_PATH

# -----------------------------
#  Conversation logging (Python-controlled)
#  One JSON file per session_id, updated on each turn
# -----------------------------
CONVERSATIONS_DIR = os.path.join(ROOT, "storage", "conversations")


def save_conversation_json(session_id: str, history):
    """
    Save the entire conversation for a session into a single JSON file.
    The same file is overwritten on each call (no extra JSON objects).
    """
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    path = os.path.join(CONVERSATIONS_DIR, f"{session_id}.json")
    record = {
        "ts": time.time(),
        "session_id": session_id,
        "messages": [
            {"role": role, "content": msg}
            for role, msg in history
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Agentic AI — Round 1", page_icon="🤖")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    if st.button("Rebuild RAG index"):
        try:
            build_vectorstore()
            st.success("RAG index rebuilt successfully.")
        except Exception as e:
            st.error(f"Error building index: {e}")

    # 🆕 Start a completely new conversation (new session_id, new agent, cleared history)
    if st.button("🆕 Start New Conversation"):
        # reset conversation-specific state
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.agent = build_agent(model=model)
        st.session_state.history = []

        # support both new and old Streamlit versions
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    st.markdown("**Paths**")
    st.code(f"Logs: {LOGS_PATH}", language="bash")
    st.code(f"Outbox: {OUTBOX_PATH}", language="bash")
    st.code(f"Tickets: {TICKETS_PATH}", language="bash")
    st.code(f"Conversations dir: {CONVERSATIONS_DIR}", language="bash")

# -----------------------------------
# Session state init (agent + history)
# -----------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "agent" not in st.session_state:
    st.session_state.agent = build_agent(model=model)
    st.session_state.history = []

st.title("Tawasul AI")

# Chat history render
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# -----------------------------
# Chat input & main interaction
# -----------------------------
user_input = st.chat_input("Type your message…")
if user_input:
    # 1) Add user message
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Agent reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.agent.invoke({"input": user_input})
                reply = result.get("output", str(result))
            except Exception as e:
                reply = f"Error: {e}"

            st.markdown(reply)
            st.session_state.history.append(("assistant", reply))

    # 3) 💾 Save/overwrite this session's conversation JSON
    try:
        save_conversation_json(
            st.session_state.session_id,
            st.session_state.history,
        )
    except Exception as e:
        st.warning(f"Failed to save conversation: {e}")
