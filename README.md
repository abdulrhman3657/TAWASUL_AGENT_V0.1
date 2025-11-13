# AgentX – LangChain + FAISS + Streamlit Agent

An interactive, memory-aware AI agent built with **LangChain**, **FAISS**, and **Streamlit**.
Supports retrieval-augmented generation (RAG) over local files and structured tools
like API calls, text logging, and email escalation.

---

## 🚀 Features
- 🤖 **OpenAI Agent** with structured tools
- 💾 **FAISS** vector search (RAG)
- 🧠 **Conversation memory**
- 🪄 **Streamlit UI** with persistent context
- 📬 **Email + logging tools** for escalation and tracking

---

## 🧰 Quick Start

```bash
# clone the repo
git clone https://github.com/<your-username>/agentx.git
cd agentx

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
