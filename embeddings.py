"""
embeddings.py
─────────────
OpenAI text embeddings and FAISS query helpers.

The FAISS index enables fast approximate nearest-neighbour (ANN) search so we
can retrieve a small candidate set (e.g. 100 records) from 20,000 in <1 second,
without comparing every pair.

This module generates vectors and queries an index it is given. Persisting an
index to disk and reading it back is base_store.py's job (save_base /
load_base), which keeps the FAISS index, id map, records, raw vectors, and
TF-IDF model versioned together as one set.
"""

from __future__ import annotations
import logging
import time

import numpy as np
import faiss
from openai import OpenAI

import config as cfg
from errors import ConfigurationError
from llm_io import retry_after_seconds
from progress import progress

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not cfg.OPENAI_API_KEY:
            raise ConfigurationError(
                "OPENAI_API_KEY is not set — it is required for embeddings. "
                "Add it to .env or the environment."
            )
        # max_retries=0: disable the SDK's own retry so our backoff (below) is
        # the only mechanism — mirrors the pattern used for LLM calls in
        # llm_scoring.py.
        _openai_client = OpenAI(api_key=cfg.OPENAI_API_KEY, max_retries=0)
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
    client = _get_client()
    all_embeddings = []
    batches = range(0, len(texts), batch_size)
    max_attempts = cfg.LLM_MAX_RETRIES     # clamped to >= 1 in config

    from openai import RateLimitError, InternalServerError

    for i in progress(batches, desc="Generating embeddings", unit="batch",
                      total=len(batches)):
        batch = texts[i : i + batch_size]
        # Replace None/empty with a placeholder
        batch = [t if t else "[empty]" for t in batch]

        # Reset per batch. Leaving it bound across iterations would mean a
        # batch that somehow fell through the retry loop silently reused the
        # previous batch's vectors — embeddings for the wrong texts, with no
        # error to show it.
        response = None
        delay = cfg.LLM_RETRY_BASE_DELAY
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.embeddings.create(
                    model=cfg.OPENAI_EMBEDDING_MODEL,
                    input=batch,
                )
                break
            except RateLimitError as exc:
                if attempt >= max_attempts:
                    raise
                wait = retry_after_seconds(exc) or delay
                logger.warning("Embeddings rate limit (attempt %d/%d) — retrying in %.1fs",
                                attempt, max_attempts, wait)
                time.sleep(wait)
                delay *= 2
            except InternalServerError:
                if attempt >= max_attempts:
                    raise
                logger.warning("Embeddings server error (attempt %d/%d) — retrying in %.1fs",
                                attempt, max_attempts, delay)
                time.sleep(delay)
                delay *= 2

        if response is None:   # defensive: the loop above either breaks or raises
            raise RuntimeError(f"Embedding batch at offset {i} produced no response")

        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)
        if i + batch_size < len(texts):
            time.sleep(0.1)   # gentle rate limiting

    return np.array(all_embeddings, dtype=np.float32)


def embed_single(text: str) -> np.ndarray:
    """Embed a single string, return a (1, DIM) float32 array."""
    return embed_texts([text])


# ── FAISS index management ─────────────────────────────────────────────────────

# Building and loading a persistent index lives in base_store.py
# (save_base / load_base) — there is deliberately no second implementation
# here. This module only generates vectors and queries an index it is handed.


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
