# Agentic AI Support System

A fully agentic customer support assistant powered by LangChain, OpenAI models, FAISS RAG, JSONL-based ticketing, analytics logging, and Streamlit/FastAPI interfaces.

---

##  Features

###  Autonomous AI Support Agent
The agent can:
- Create, update, and close support tickets  
- Retrieve knowledge via RAG  
- Escalate critical issues to human support  
- Save conversation notes  

---

##  RAG Knowledge Base
All `.txt` files in the `data/` directory are embedded into a FAISS vectorstore.

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

