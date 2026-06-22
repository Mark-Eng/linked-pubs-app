"""
embeddings.py
─────────────
OpenAI text embeddings and FAISS index management.

The FAISS index enables fast approximate nearest-neighbour (ANN) search so we
can retrieve a small candidate set (e.g. 100 records) from 20,000 in <1 second,
without comparing every pair.

Index storage
─────────────
  index/publications.faiss  — the FAISS index (float32 vectors)
  index/id_map.json         — list mapping FAISS row position → DB publication id
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import numpy as np
import faiss
from openai import OpenAI

import config as cfg

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _openai_client


# ── Embedding generation ───────────────────────────────────────────────────────

EMBED_VERSION = "v2-with-authors"   # bump this if _text_for_embedding changes


def _text_for_embedding(record: dict) -> str:
    """
    Build the text string used to generate an embedding for a record.

    Authors are included first because author overlap is a primary signal of
    linked publications, and the ANN search must be sensitive to it at the
    retrieval stage — not just at the later feature-scoring stage.

    By the time this function is called, `record["authors"]` is already a
    plain list of strings (one per author), regardless of how the source file
    stored them.  We join them with "; " purely to form a single text string
    for the embedding API; no assumptions are made about the original format.
    """
    authors  = record.get("authors") or []
    title    = record.get("title")   or ""
    abstract = record.get("abstract") or ""

    author_str       = "; ".join(authors) if authors else ""
    abstract_snippet = abstract[:1000]

    parts = [p for p in [author_str, title, abstract_snippet] if p]
    return "\n\n".join(parts).strip()


def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed a list of strings using the OpenAI embeddings API.
    Processes in batches to respect API limits.
    Returns an (N, EMBEDDING_DIM) float32 array.
    """
    from tqdm import tqdm

    client = _get_client()
    all_embeddings = []
    batches = range(0, len(texts), batch_size)

    for i in tqdm(batches, desc="Generating embeddings", unit="batch",
                  total=len(batches)):
        batch = texts[i : i + batch_size]
        # Replace None/empty with a placeholder
        batch = [t if t else "[empty]" for t in batch]
        response = client.embeddings.create(
            model=cfg.OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)
        if i + batch_size < len(texts):
            time.sleep(0.1)   # gentle rate limiting

    return np.array(all_embeddings, dtype=np.float32)


def embed_single(text: str) -> np.ndarray:
    """Embed a single string, return a (1, DIM) float32 array."""
    return embed_texts([text])


# ── FAISS index management ─────────────────────────────────────────────────────

def build_faiss_index(records: list[dict]) -> tuple[faiss.Index, list]:
    """
    Build a FAISS inner-product index from a list of publication records.
    Vectors are L2-normalised so inner product == cosine similarity.

    Returns (index, id_map) where id_map[i] is the DB id of FAISS row i.
    Saves both to disk.
    """
    texts = [_text_for_embedding(r) for r in records]
    id_map = [r["id"] for r in records]

    logger.info("Generating embeddings for %d records…", len(texts))
    vecs = embed_texts(texts)

    # L2-normalise for cosine similarity via inner product
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(cfg.EMBEDDING_DIM)
    index.add(vecs)

    faiss.write_index(index, str(cfg.FAISS_INDEX_PATH))
    with open(cfg.ID_MAP_PATH, "w") as fh:
        json.dump(id_map, fh)

    logger.info(
        "FAISS index built: %d vectors. Saved to %s",
        index.ntotal,
        cfg.FAISS_INDEX_PATH,
    )
    return index, id_map


def load_faiss_index() -> tuple[faiss.Index, list] | tuple[None, None]:
    """Load the FAISS index and id_map from disk. Returns (None, None) if not found."""
    if not Path(cfg.FAISS_INDEX_PATH).exists():
        logger.warning("FAISS index not found — run build_index.py first")
        return None, None
    index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))
    with open(cfg.ID_MAP_PATH) as fh:
        id_map = json.load(fh)
    logger.info("Loaded FAISS index with %d vectors", index.ntotal)
    return index, id_map


def query_index(
    query_text: str,
    index: faiss.Index,
    id_map: list,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """
    Query the FAISS index for the nearest neighbours of query_text.

    Returns a list of (publication_id, cosine_similarity) sorted by
    descending similarity.
    """
    if top_k is None:
        top_k = cfg.ANN_TOP_K

    vec = embed_single(query_text)
    faiss.normalize_L2(vec)

    k = min(top_k, index.ntotal)
    sims, indices = index.search(vec, k)

    results = []
    for sim, idx in zip(sims[0], indices[0]):
        if idx >= 0:
            results.append((id_map[idx], float(sim)))
    return results


# ── In-memory index for batch comparisons ─────────────────────────────────────

def build_in_memory_index(records: list[dict]) -> tuple[faiss.Index, list]:
    """
    Build a FAISS index in memory (no disk write) from a list of records.
    Used for standalone file-based comparisons.
    """
    texts = [_text_for_embedding(r) for r in records]
    id_map = list(range(len(records)))   # use list index as id

    vecs = embed_texts(texts)
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(cfg.EMBEDDING_DIM)
    index.add(vecs)
    return index, id_map
