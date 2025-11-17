# app/rag.py
import os
import json
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index")

# Optional: cache embeddings so we don't recreate them every call
_EMBEDDINGS = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = OpenAIEmbeddings()  # requires OPENAI_API_KEY
    return _EMBEDDINGS


def _load_json_documents() -> List[Document]:
    """
    Load all .json files from data/ as Documents.

    Each JSON file can be:
      - a list of items, OR
      - an object with an "items" list, OR
      - a single object/string.

    For each item we try, in order:
      - item["text"]
      - item["content"]
      - "Q: {question}\\nA: {answer}" if both 'question' and 'answer' exist
      - json.dumps(item) as a fallback
    """
    docs: List[Document] = []
    os.makedirs(DATA_DIR, exist_ok=True)

    for name in os.listdir(DATA_DIR):
        if not name.lower().endswith(".json"):
            continue

        path = os.path.join(DATA_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[RAG] Skipped {name}: {e}")
            continue

        # Normalize into a list of records
        if isinstance(raw, list):
            records = raw
        elif isinstance(raw, dict):
            if isinstance(raw.get("items"), list):
                records = raw["items"]
            else:
                records = [raw]
        else:
            records = [raw]

        for rec in records:
            text = None

            if isinstance(rec, str):
                text = rec
            elif isinstance(rec, dict):
                if "text" in rec:
                    text = rec["text"]
                elif "content" in rec:
                    text = rec["content"]
                elif "question" in rec and "answer" in rec:
                    text = f"Q: {rec['question']}\nA: {rec['answer']}"
                else:
                    # Fallback: serialize dict
                    text = json.dumps(rec, ensure_ascii=False)
            else:
                text = str(rec)

            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": name}
                    )
                )

    return docs


def _load_and_chunk_documents() -> List[Document]:
    """
    Load JSON documents and chunk them into smaller passages for better RAG.

    Chunking fixes the 'big blob' issue: long documents are split into
    ~1000-character pieces with overlap, so retrieval is more precise.
    """
    raw_docs = _load_json_documents()
    if not raw_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )

    chunked_docs: List[Document] = []
    for doc in raw_docs:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(
                Document(
                    page_content=chunk,
                    metadata=doc.metadata,
                )
            )

    return chunked_docs


def build_vectorstore() -> FAISS:
    """
    Build and persist a FAISS index from the JSON documents in data/.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    docs = _load_and_chunk_documents()
    if not docs:
        raise RuntimeError(f"No JSON documents found in {DATA_DIR}")

    embeddings = _get_embeddings()
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    return vs


def get_vectorstore() -> FAISS:
    """
    Load the FAISS index from disk. If it doesn't exist, build it once.
    """
    embeddings = _get_embeddings()
    try:
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,  # demo only
        )
    except Exception as e:
        print(f"[RAG] Failed to load FAISS index ({e}), rebuilding...")
        return build_vectorstore()


def similarity_search(query: str, k: int = 4) -> List[str]:
    """
    Return the top-k matching passages for the given query.
    """
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]


if __name__ == "__main__":
    vs = build_vectorstore()
    print("FAISS index saved to:", FAISS_INDEX_PATH)
    print("Number of chunks indexed:", len(vs.docstore._dict))
