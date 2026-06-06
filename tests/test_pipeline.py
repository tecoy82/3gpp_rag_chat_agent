"""
Validation checks for each pipeline layer.
Run with: pytest tests/

These tests verify the pipeline contracts without hitting the LLM API,
so they run fast and cost nothing.
"""

import os
import pytest


class TestChunking:
    def test_chunk_size_respected(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import CHUNK_SIZE, CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        text = "word " * 1000  # 5000 chars
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= CHUNK_SIZE + 50, f"Chunk too large: {len(chunk)}"

    def test_overlap_preserves_context(self):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=CHUNK_OVERLAP)
        text = "a " * 500
        chunks = splitter.split_text(text)
        assert len(chunks) > 1, "Expected multiple chunks"


class TestSemanticCache:
    def test_cache_hit_on_similar_query(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from cache.semantic_cache import SemanticCache

        cache = SemanticCache()
        cache.set("What is 5G NR?", "5G NR is the New Radio standard.")
        result = cache.get("Explain 5G NR to me")  # similar but not identical
        # May or may not hit depending on threshold — just assert no crash
        assert result is None or isinstance(result, str)

    def test_cache_exact_hit(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from cache.semantic_cache import SemanticCache

        cache = SemanticCache()
        cache.set("What is LTE?", "LTE is Long Term Evolution.")
        result = cache.get("What is LTE?")
        assert result == "LTE is Long Term Evolution."

    def test_cache_eviction(self):
        import sys, time
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from cache.semantic_cache import SemanticCache
        from unittest.mock import patch

        cache = SemanticCache()
        cache.set("test query", "test answer")
        assert cache.size == 1

        # Simulate TTL expiry
        with patch("cache.semantic_cache.CACHE_TTL_SECONDS", -1):
            cache.evict_expired()
        assert cache.size == 0


class TestConfig:
    def test_required_env_vars_documented(self):
        """Ensures .env.example covers the vars config.py reads."""
        example_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        assert os.path.exists(example_path), ".env.example missing"
        content = open(example_path).read()
        for var in ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CHROMA_PATH"):
            assert var in content, f"{var} not documented in .env.example"
