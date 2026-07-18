"""
Minimal in-memory RAG. No external vector DB needed for a hackathon demo.
build_index(docs, client) -> index
retrieve(query, index, client, k) -> list[str] of top-k chunks
Swap in Chroma/FAISS later only if you actually need scale.
"""

import numpy as np
from openai import OpenAI

# Gemini embedding model (works via the OpenAI-compatible endpoint).
# If you switch the app back to plain OpenAI, use "text-embedding-3-small".
EMBED_MODEL = "text-embedding-004"


def _chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Naive character chunking with overlap. Good enough for a demo."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if c.strip()]


def _embed(texts: list[str], client: OpenAI) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def build_index(docs: list[str], client: OpenAI) -> dict:
    """Chunk all docs, embed them, return an in-memory index."""
    all_chunks = []
    for d in docs:
        all_chunks.extend(_chunk(d))
    if not all_chunks:
        return {"chunks": [], "vectors": np.zeros((0, 1))}
    vectors = _embed(all_chunks, client)
    return {"chunks": all_chunks, "vectors": vectors}


def retrieve(query: str, index: dict, client: OpenAI, k: int = 3) -> list[str]:
    """Return top-k chunks by cosine similarity."""
    if not index["chunks"]:
        return []
    q = _embed([query], client)[0]
    mat = index["vectors"]
    # cosine similarity
    sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
    top = np.argsort(sims)[::-1][:k]
    return [index["chunks"][i] for i in top]
