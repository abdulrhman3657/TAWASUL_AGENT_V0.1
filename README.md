# AgentX – LangChain + FAISS + Streamlit Agent

An interactive, memory-aware AI agent built with **LangChain**, **FAISS**, and **Streamlit**.
Supports retrieval-augmented generation (RAG) over local files and structured tools
like API calls, text logging, and email escalation.

---

##  Features
-  **OpenAI Agent** with structured tools
-  **FAISS** vector search (RAG)
-  **Conversation memory**
-  **Streamlit UI** with persistent context
-  **Email + logging tools** for escalation and tracking

---

##  Python Version

This project **requires Python 3.11**.

Other Python versions (such as 3.10 or 3.12) may cause dependency issues with FAISS, LangChain, and certain Windows-specific libraries.  

##  Quick Start

```bash

# create env
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt

# set your OpenAI key
setx OPENAI_API_KEY "sk-..."  # Windows
# or use a .env file

# build the FAISS index
python -m app.rag

# run Streamlit UI
python -m streamlit run app/streamlit_app.py

```

## ⚠️ Note on Python Wheel Issues
Some dependencies — especially vector DB backends like FAISS or Chroma — may fail to install on certain systems due to missing pre-built Python wheels or platform-specific limitations.

If you see errors about wheel builds or DLL load failures, try:

```bash
pip install -U pip setuptools wheel

