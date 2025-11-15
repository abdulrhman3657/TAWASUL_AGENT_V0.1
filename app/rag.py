# app/rag.py
import os
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index")


def _load_txt_documents() -> List[Document]:
    """
    Load all .txt files from data/ as Documents.

    These represent the knowledge base: FAQs, policies, how-tos, etc.
    """
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


def build_vectorstore() -> FAISS:
    """
    Build and persist a FAISS index from the txt documents.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    docs = _load_txt_documents()
    embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    return vs


def get_vectorstore() -> FAISS:
    embeddings = OpenAIEmbeddings()
    # allow_dangerous_deserialization=True for local use; avoid for untrusted indexes
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def similarity_search(query: str, k: int = 4) -> List[str]:
    """
    Return the top-k matching passages for the given query.
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]


if __name__ == "__main__":
    build_vectorstore()
    print("FAISS index saved to:", FAISS_INDEX_PATH)
