"""
text_processing.py
──────────────────
Title tokenisation, TF-IDF model building, and weighted cosine similarity.

The TF-IDF model is fitted on the corpus of all publication titles in the DB,
so that rare, domain-specific terms (e.g. "deworming", "Odisha") receive high
weight and common words receive low weight, regardless of stoplist coverage.
The fitted model is persisted to disk and reloaded for real-time queries.
"""

from __future__ import annotations
import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ── Extended stopword list for academic/development sector titles ──────────────
_BASE_STOPWORDS = {
    # English function words
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "that", "this", "these",
    "those", "it", "its", "we", "our", "their", "than", "into", "up",
    "out", "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "each", "all", "any", "some", "such", "own", "between", "through",
    "during", "before", "after", "above", "below", "about", "against",
    # Academic / impact evaluation boilerplate
    "evidence", "evaluation", "evaluations", "study", "studies", "analysis",
    "paper", "working", "report", "results", "findings", "impact", "impacts",
    "effect", "effects", "effectiveness", "systematic", "review", "reviews",
    "experimental", "quasi", "randomised", "randomized", "rct", "trial",
    "intervention", "interventions", "programme", "program", "project",
    "policy", "policies", "using", "based", "new", "case",
}


def _tokenise(text: str) -> list[str]:
    """Lowercase, remove punctuation, tokenise, drop short/stop tokens."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[''`]s\b", "", text)          # possessives
    text = re.sub(r"[^\w\s-]", " ", text)          # punctuation → space
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in _BASE_STOPWORDS]


# ── TF-IDF vectoriser ──────────────────────────────────────────────────────────

def fit_tfidf(titles: list[str]) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectoriser on a list of title strings. No disk I/O — pair
    with save_tfidf() to persist it.

    This is the single definition of the TF-IDF configuration. Every caller
    must go through it: a model fitted with different parameters produces
    title scores that are not comparable with those from the base index.
    """
    vectoriser = TfidfVectorizer(
        analyzer="word",
        tokenizer=_tokenise,
        token_pattern=None,       # tokenizer overrides pattern
        min_df=2,                 # ignore terms appearing in only 1 doc
        max_df=0.85,              # ignore very common terms
        sublinear_tf=True,        # log(1 + tf) smoothing
        ngram_range=(1, 2),       # unigrams and bigrams
    )
    vectoriser.fit(titles)
    logger.info("TF-IDF model fitted on %d titles", len(titles))
    return vectoriser


def save_tfidf(vectoriser: TfidfVectorizer, path: Path | str) -> None:
    """Serialise a fitted vectoriser's vocabulary and IDF weights to JSON."""
    data = {
        "params": {
            "min_df":      vectoriser.min_df,
            "max_df":      vectoriser.max_df,
            "sublinear_tf": vectoriser.sublinear_tf,
            "ngram_range": list(vectoriser.ngram_range),
        },
        "vocabulary": vectoriser.vocabulary_,
        "idf":        vectoriser.idf_.tolist(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    logger.info("TF-IDF model saved to %s", path)


def load_tfidf(path: Path | str) -> TfidfVectorizer | None:
    """
    Load a fitted TF-IDF model from the given path, or None if not found.

    The JSON file stores the vocabulary and IDF weights; the vectoriser is
    reconstructed from these so that transform() works identically to the
    original fitted model.
    """
    if not Path(path).exists():
        logger.warning("TF-IDF model not found at %s", path)
        return None
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    from sklearn.feature_extraction.text import TfidfTransformer

    params = data["params"]
    vectoriser = TfidfVectorizer(
        analyzer="word",
        tokenizer=_tokenise,
        token_pattern=None,
        min_df=params["min_df"],
        max_df=params["max_df"],
        sublinear_tf=params["sublinear_tf"],
        ngram_range=tuple(params["ngram_range"]),
    )
    # Restore fitted state directly, bypassing a re-fit
    vectoriser.vocabulary_ = {k: int(v) for k, v in data["vocabulary"].items()}
    vectoriser.fixed_vocabulary_ = True
    vectoriser._tfidf = TfidfTransformer(sublinear_tf=params["sublinear_tf"])
    vectoriser._tfidf.idf_ = np.array(data["idf"], dtype=np.float64)
    return vectoriser


def title_similarity(title_a: str, title_b: str, vectoriser: TfidfVectorizer) -> float:
    """
    Compute TF-IDF weighted cosine similarity between two title strings.
    Returns a float in [0, 1].
    """
    if not title_a or not title_b:
        return 0.0
    vecs = vectoriser.transform([title_a, title_b])
    sim = cosine_similarity(vecs[0], vecs[1])[0, 0]
    return float(np.clip(sim, 0.0, 1.0))


def title_similarity_fallback(title_a: str, title_b: str) -> float:
    """
    Token-overlap Jaccard similarity used when no TF-IDF model is available
    (e.g. for standalone file-based comparisons with no pre-built corpus).
    All tokens weighted equally.
    """
    toks_a = set(_tokenise(title_a))
    toks_b = set(_tokenise(title_b))
    if not toks_a or not toks_b:
        return 0.0
    intersection = toks_a & toks_b
    union = toks_a | toks_b
    return len(intersection) / len(union)


def get_title_tokens(title: str) -> set[str]:
    """Return the set of meaningful tokens from a title (for blocking)."""
    return set(_tokenise(title))
