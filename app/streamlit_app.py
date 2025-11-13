import os
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
from app.tools import LOGS_PATH, OUTBOX_PATH
import streamlit as st
import os


from app.agent import build_agent
from app.rag import build_vectorstore
from app.tools import LOGS_PATH, OUTBOX_PATH  # just to surface paths

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

    st.markdown("**Paths**")
    st.code(f"Logs: {LOGS_PATH}", language="bash")
    st.code(f"Outbox: {OUTBOX_PATH}", language="bash")

# Build agent once and keep in session
if "agent" not in st.session_state:
    st.session_state.agent = build_agent(model=model)
    st.session_state.history = []

st.title("🤖 Agentic AI (LangChain + Chroma + Streamlit)")

# Chat history render
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# Chat input
user_input = st.chat_input("Type your message…")
if user_input:
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.agent.invoke({"input": user_input})
                reply = result.get("output", str(result))
            except Exception as e:
                reply = f"Error: {e}"
            st.markdown(reply)
            st.session_state.history.append(("assistant", reply))
