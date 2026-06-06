"""
Layer 3: Chunking + Embedding + Indexing
-----------------------------------------
This is the heart of RAG. Three sub-steps:

1. CHUNKING: Split long documents into overlapping windows.
   Why overlap? So a sentence at the end of chunk N also appears at the start
   of chunk N+1. A query hitting that sentence retrieves coherent context.

2. EMBEDDING: Convert each chunk to a dense vector (list of floats).
   Similar meaning -> similar vectors -> close in vector space.
   We use sentence-transformers which runs 100% locally.

3. INDEXING: Store vectors + metadata in ChromaDB so we can do
   approximate nearest-neighbour search at query time.

RAG concept: at query time, we embed the user's question the same way,
then find the K closest chunk vectors. Those chunks = the "context" we
hand to the LLM.
"""

import os
import sys
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_CLEANED_PATH, DATA_CHUNKS_PATH, CHROMA_PATH,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_TOP_K
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Prefer Apple MPS (Metal) → fall back to CPU. Add "cuda" before "mps" if on Linux/Windows with Nvidia.
def _best_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_cleaned_docs(cleaned_dir: str) -> list[Document]:
    docs = []
    for filename in os.listdir(cleaned_dir):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(cleaned_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Metadata travels with each chunk — crucial for citations in answers
        docs.append(Document(page_content=text, metadata={"source": filename}))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split {len(docs)} doc(s) into {len(chunks)} chunks")
    return chunks


def save_chunks_to_disk(chunks: list[Document], chunks_dir: str):
    """Write chunks to disk as JSON for inspection and debugging.

    Opens data/chunks/chunks.json to see exactly what text went into each
    vector — useful for tuning CHUNK_SIZE and CHUNK_OVERLAP.
    """
    os.makedirs(chunks_dir, exist_ok=True)
    out_path = os.path.join(chunks_dir, "chunks.json")
    records = [
        {
            "index": i,
            "source": doc.metadata.get("source", "unknown"),
            "char_count": len(doc.page_content),
            "text": doc.page_content,
        }
        for i, doc in enumerate(chunks)
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Chunks written to {out_path} ({len(records)} entries)")


def build_vectorstore(chunks: list[Document]) -> Chroma:
    device = _best_device()
    print(f"  Loading embedding model: {EMBEDDING_MODEL} (device: {device})")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
    )

    print(f"  Building ChromaDB index at {CHROMA_PATH} ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"  Indexed {len(chunks)} chunks")
    return vectorstore


def smoke_test(vectorstore: Chroma):
    """Quick sanity check — retrieve top chunks for a test query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    results = retriever.invoke("What is the architecture of 5G NR?")
    print(f"\n  Smoke test — top {len(results)} chunks for '5G NR architecture':")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i}] ({doc.metadata['source']}) {preview}...")


def main():
    docs = load_cleaned_docs(DATA_CLEANED_PATH)
    if not docs:
        print(f"No cleaned docs in {DATA_CLEANED_PATH}/ — run clean_docs.py first.")
        return

    chunks = chunk_documents(docs)
    save_chunks_to_disk(chunks, DATA_CHUNKS_PATH)
    vectorstore = build_vectorstore(chunks)
    smoke_test(vectorstore)
    print("\nDone. Next step: run the agent or app.py")


if __name__ == "__main__":
    main()
