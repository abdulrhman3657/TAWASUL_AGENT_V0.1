# This module handles document loading, embedding, and similarity search using FAISS.
# app/rag.py
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index")

# Also loads .env for the OpenAI embeddings API key.
from dotenv import load_dotenv
load_dotenv(override=True)


# Ensures DATA_DIR exists.
# Iterates over all .txt files in data/.
# Reads each file into a single Document:
# Collects them into a list.
def _load_txt_documents() -> List[Document]:
    docs: List[Document] = []
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in os.listdir(DATA_DIR):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(DATA_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": name}))
        except Exception as e:
            print(f"[RAG] Skipped {name}: {e}")
    return docs

# Ensure storage/ exists.
# Load docs from _load_txt_documents().
# Create OpenAIEmbeddings().
# Build a FAISS index:
# This is what the “Rebuild RAG index” button in Streamlit triggers.
def build_vectorstore() -> FAISS:
    os.makedirs(DB_DIR, exist_ok=True)
    docs = _load_txt_documents()
    embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    return vs

# Creates an OpenAIEmbeddings object.
# Loads the local FAISS index:
# This is for local trusted use only (as the comment warns).
def get_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings()
    # allow_dangerous_deserialization=True for local use; avoid for untrusted indexes
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Loads the vector store.
# Calls vs.similarity_search(query, k=k).
# Returns just the page_content for each matched Document.
# This is what the search_docs tool uses inside the agent.
def similarity_search(query: str, k: int = 4) -> List[str]:
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]


# If you run:
# python -m app.rag
# It will build the index and print where it’s saved.
if __name__ == "__main__":
    build_vectorstore()
    print("FAISS index saved to:", FAISS_INDEX_PATH)
