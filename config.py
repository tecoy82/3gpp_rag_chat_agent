import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

CHROMA_PATH = os.getenv("CHROMA_PATH", "vectorstore/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))

# Chunking strategy — these are the knobs you'll tune
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # overlap between chunks (preserves context at boundaries)

# How many chunks to retrieve per query
RETRIEVER_TOP_K = 5

DATA_RAW_PATH = "data/raw"
DATA_CLEANED_PATH = "data/cleaned"
DATA_CHUNKS_PATH = "data/chunks"
