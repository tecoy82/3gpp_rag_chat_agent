# 3GPP RAG Chat Agent

A learning project demonstrating a production-style RAG pipeline over 3GPP LTE and 5G standards, with a LangGraph agent and Flask chatbot UI.

## Architecture

```
scripts/fetch_specs.py   →  data/raw/         (download 3GPP PDFs)
scripts/clean_docs.py    →  data/cleaned/     (parse + strip boilerplate)
scripts/embed.py         →  vectorstore/      (chunk + embed + index in ChromaDB)
agent/rag_agent.py       →  LangGraph ReAct agent
cache/semantic_cache.py  →  in-memory semantic cache
ui/app.py                →  Flask chatbot server
ui/static/index.html     →  browser chat UI
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY
```

## Run the pipeline (one-time)

```bash
python scripts/fetch_specs.py   # download specs
python scripts/clean_docs.py    # parse + clean
python scripts/embed.py         # build vector index
```

## Start the app

```bash
python ui/app.py
# Open http://localhost:5000
```

## Run tests

```bash
pytest tests/ -v
```

## Key design decisions

- **ChromaDB** persisted locally — zero infrastructure, swap to Qdrant (Docker) for a more enterprise look
- **sentence-transformers/all-MiniLM-L6-v2** for embeddings — free, local, no API key
- **LangGraph ReAct agent** — decides when/how to retrieve rather than a fixed chain
- **Semantic cache** — cosine similarity threshold (0.92) so near-duplicate queries hit cache
- **Claude (claude-sonnet-4-6)** as LLM via `langchain-anthropic`
