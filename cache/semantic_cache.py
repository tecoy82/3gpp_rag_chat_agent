"""
Layer 5: Semantic Cache
-----------------------
Exact string caches are useless for NLP: "What is 5G?" and "Can you explain 5G?"
are the same question but different strings.

A semantic cache embeds the incoming query and checks if a SIMILAR query was
already answered. If the new query's embedding is within a cosine distance
threshold of a cached one, we return the cached answer — no LLM call needed.

This is a simplified in-memory version. In production you'd:
  - Use Redis with the redis-py client for persistence across restarts
  - Or LangChain's built-in RedisSemanticCache / GPTCache integrations
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EMBEDDING_MODEL, CACHE_TTL_SECONDS

SIMILARITY_THRESHOLD = 0.92  # cosine similarity above this = cache hit


class SemanticCache:
    def __init__(self):
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._entries: list[dict] = []  # {embedding, answer, query, timestamp}

    def _embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # normalized vectors: dot == cosine

    def _is_expired(self, entry: dict) -> bool:
        return time.time() - entry["timestamp"] > CACHE_TTL_SECONDS

    def get(self, query: str) -> str | None:
        q_emb = self._embed(query)
        for entry in self._entries:
            if self._is_expired(entry):
                continue
            sim = self._cosine_similarity(q_emb, entry["embedding"])
            if sim >= SIMILARITY_THRESHOLD:
                print(f"  [cache hit] similarity={sim:.3f} for cached query: '{entry['query']}'")
                return entry["answer"]
        return None

    def set(self, query: str, answer: str):
        self._entries.append({
            "query": query,
            "embedding": self._embed(query),
            "answer": answer,
            "timestamp": time.time(),
        })

    def evict_expired(self):
        self._entries = [e for e in self._entries if not self._is_expired(e)]

    @property
    def size(self) -> int:
        return len(self._entries)
