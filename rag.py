"""
Minimal in-memory RAG with graceful fallback.

If the provider supports embeddings (OpenAI, Gemini), we use semantic search.
If it does NOT (Groq only does chat), we fall back to keyword overlap scoring,
so file upload + retrieval still work for the demo — just less "semantic".

Toggle EMBED_ENABLED to match your provider:
  - Groq            -> EMBED_ENABLED = False  (uses keyword fallback)
  - OpenAI / Gemini -> EMBED_ENABLED = True   (uses embeddings, set EMBED_MODEL)
"""

import re
import numpy as np
from openai import OpenAI

# Groq has no embeddings endpoint, so keep this False while on Groq.
EMBED_ENABLED = False

# Only used when EMBED_ENABLED = True.
# OpenAI: "text-embedding-3-small"   Gemini: "text-embedding-004"
EMBED_MODEL = "text-embedding-3-small"


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
    """Chunk all docs. Embed only if embeddings are enabled."""
    all_chunks = []
    for d in docs:
        all_chunks.extend(_chunk(d))
    index = {"chunks": all_chunks, "vectors": None}
    if EMBED_ENABLED and all_chunks:
        index["vectors"] = _embed(all_chunks, client)
    return index


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, index: dict, client: OpenAI, k: int = 3) -> list[str]:
    """Top-k chunks. Semantic if embeddings available, else keyword overlap."""
    chunks = index["chunks"]
    if not chunks:
        return []

    if EMBED_ENABLED and index.get("vectors") is not None:
        q = _embed([query], client)[0]
        mat = index["vectors"]
        sims = mat @ q / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
        top = np.argsort(sims)[::-1][:k]
        return [chunks[i] for i in top]

    # Keyword fallback: score chunks by shared-word overlap with the query.
    q_tokens = _tokens(query)
    scored = []
    for c in chunks:
        overlap = len(q_tokens & _tokens(c))
        scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    # If nothing overlaps, just return the first k chunks so context isn't empty.
    top = [c for score, c in scored if score > 0][:k]
    return top if top else chunks[:k]
