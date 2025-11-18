# app/fallback_detector.py
from functools import lru_cache
from typing import Tuple, List

from langchain_openai import OpenAIEmbeddings
import math

# new faqs logger

# Canonical fallback replies the model is supposed to use
FALLBACK_SENTENCES = [
    "I don't have enough information to answer that.",
    "لا أملك معلومات كافية للإجابة على ذلك.",
]
# you can add "I'm a customer service agent and can only help with questions related to our products and services."
# for more general logging


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@lru_cache(maxsize=1)
def _get_model_and_fallback_vectors() -> Tuple[OpenAIEmbeddings, List[List[float]]]:
    """
    Lazily instantiate OpenAIEmbeddings and precompute embeddings
    for all canonical fallback sentences. Cached across calls.
    """
    model = OpenAIEmbeddings()
    fallback_vecs = model.embed_documents(FALLBACK_SENTENCES)
    return model, fallback_vecs


def is_semantic_fallback(reply: str, threshold: float = 0.92) -> Tuple[bool, float]:
    """
    Return (is_fallback_like, similarity_score).

    - Embeds the reply.
    - Computes cosine similarity against each fallback sentence.
    - If max similarity >= threshold => treat as fallback.
    """
    reply = (reply or "").strip()
    if not reply:
        return False, 0.0

    model, fallback_vecs = _get_model_and_fallback_vectors()
    reply_vec = model.embed_query(reply)

    sims = [_cosine_similarity(reply_vec, fv) for fv in fallback_vecs]
    best = max(sims) if sims else 0.0
    return best >= threshold, best
