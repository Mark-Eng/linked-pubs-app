"""
author_matching.py
──────────────────
Fuzzy author name matching using Jaro-Winkler similarity.

Strategy
────────
1. Normalise each name string: lowercase, strip punctuation, standardise
   initials so "Smith, J." and "Smith, John" can be compared fairly.
2. For each author in the smaller set, find its best match in the larger set
   using Jaro-Winkler similarity (good for names: handles transpositions,
   abbreviations, and prefix weighting).
   Token-sorted forms are also compared so that transposed surname/given-name
   order (e.g. "Shilpa Aggarwal" vs "Aggarwal Shilpa") scores correctly.
3. The author_overlap_score is a weighted sum where rarer names contribute
   more to the score than common ones (controlled by NAME_FREQUENCY_SOURCE).

Name frequency weighting
────────────────────────
Set NAME_FREQUENCY_SOURCE in config.py or .env:
  "none"          — all matched names count equally (default fallback)
  "names-dataset" — uses the names-dataset package for global name frequencies
  "orcid"         — uses a pre-built frequency table from the ORCID public data
                    dump (index/name_frequencies.json, built by build_name_frequencies.py)
"""

from __future__ import annotations
import json
import logging
import math
import re
import unicodedata

from rapidfuzz.distance import JaroWinkler

import config as cfg

logger = logging.getLogger(__name__)


# ── Name normalisation ─────────────────────────────────────────────────────────

def _remove_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _normalise_name(name: str) -> str:
    """
    Normalise a single author name string for fuzzy comparison.

    Transformations applied:
    - Unicode decomposition (accents stripped)
    - Lowercase
    - Remove punctuation except hyphens within words
    - Collapse whitespace
    - Move surname to front if comma-separated ("Smith, John" → "smith john")
    - Expand initials: treat "J" the same as "john" by not removing them
    """
    if not name:
        return ""
    name = _remove_accents(name)
    name = name.lower()
    # Handle "Surname, Given" → standardise to "surname given" for consistency.
    if "," in name:
        parts = name.split(",", 1)
        name = parts[0].strip() + " " + parts[1].strip()
    # Remove punctuation except spaces and hyphens
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def normalise_author_list(authors: list[str]) -> list[str]:
    """Normalise a list of author name strings."""
    return [_normalise_name(a) for a in authors if a and a.strip()]


# ── Pairwise similarity ────────────────────────────────────────────────────────

def _name_similarity(name_a: str, name_b: str) -> float:
    """
    Jaro-Winkler similarity between two normalised name strings.

    Also compares token-sorted forms to handle transpositions where surname
    and given name are swapped between records (e.g. "shilpa aggarwal" vs
    "aggarwal shilpa").  Returns the higher of the two scores.
    """
    if not name_a or not name_b:
        return 0.0
    score_as_is = JaroWinkler.normalized_similarity(name_a, name_b)
    sorted_a = " ".join(sorted(name_a.split()))
    sorted_b = " ".join(sorted(name_b.split()))
    score_sorted = JaroWinkler.normalized_similarity(sorted_a, sorted_b)
    return max(score_as_is, score_sorted)


def _best_match_score(name: str, candidates: list[str]) -> float:
    """Return the highest similarity score for `name` against any candidate."""
    if not candidates:
        return 0.0
    return max(_name_similarity(name, c) for c in candidates)


# ── Name frequency / rarity weighting ─────────────────────────────────────────

class _FrequencyDB:
    """Base class for name frequency sources."""
    def get_weight(self, normalised_name: str) -> float:
        """Return a rarity multiplier ≥ 1.0 for a normalised name string."""
        return 1.0


class _NullFrequencyDB(_FrequencyDB):
    """No weighting — all names count equally."""
    pass


class _OrcidFrequencyDB(_FrequencyDB):
    """
    Rarity weights derived from the ORCID public data dump.
    Expects index/name_frequencies.json built by build_name_frequencies.py.
    JSON structure: {"given": {name: count, ...}, "family": {name: count, ...}, "total": int}
    """
    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self._given:  dict[str, int] = data["given"]
        self._family: dict[str, int] = data["family"]
        self._total:  int            = data["total"]
        logger.info(
            "Loaded ORCID name frequencies (%d given, %d family, %d total records)",
            len(self._given), len(self._family), self._total,
        )

    def get_weight(self, normalised_name: str) -> float:
        tokens = [t for t in normalised_name.split() if len(t) >= 3]
        if not tokens:
            return 1.0
        # Use the rarest token to drive the weight (minimum combined frequency)
        best = 1.0
        for token in tokens:
            count = self._given.get(token, 0) + self._family.get(token, 0)
            if count == 0:
                # Not in dataset — treat as maximally rare
                best = cfg.NAME_RARITY_MAX_WEIGHT
                break
            fraction = count / self._total
            # log scale: fraction=0.01 (common) → ~1.0; fraction=0.0001 (rare) → ~5.6
            w = math.log(0.01 / fraction) + 1.0
            best = max(best, w)
        return min(cfg.NAME_RARITY_MAX_WEIGHT, max(1.0, best))


class _NamesDatasetDB(_FrequencyDB):
    """
    Rarity weights from the names-dataset package (pip install names-dataset).
    Uses summed country percentages as a global frequency proxy.
    """
    def __init__(self):
        from names_dataset import NameDataset
        self._nd = NameDataset()
        logger.info("Loaded names-dataset for author name frequency weighting")

    def get_weight(self, normalised_name: str) -> float:
        tokens = [t for t in normalised_name.split() if len(t) >= 3]
        if not tokens:
            return 1.0
        best = 1.0
        for token in tokens:
            info = self._nd.search(token.title())
            if not info:
                # Unknown name — treat as moderately rare
                best = max(best, 2.0)
                continue
            # Find the minimum rank across both roles and all countries.
            # Minimum rank = "how common is this name in ANY role, anywhere?"
            # Common names (Smith as surname, rank=1) → min_rank=1 → weight=1.0
            # Rare names (Aggarwal as surname, rank=412) → high weight
            all_ranks = []
            for role in ("first_name", "last_name"):
                role_data = info.get(role) or {}
                all_ranks += [
                    v for v in role_data.get("rank", {}).values()
                    if v is not None
                ]
            if not all_ranks:
                best = max(best, 2.0)
                continue
            min_rank = min(all_ranks)
            # log scale normalised so rank ≤ 10 → weight ≤ 1.0:
            #   rank=1   → 0.29  → clamped to 1.0
            #   rank=10  → 1.00
            #   rank=100 → 1.93
            #   rank=412 → 2.52
            #   rank=2000→ 3.24  → capped at NAME_RARITY_MAX_WEIGHT
            w = math.log1p(min_rank) / math.log1p(10)
            best = max(best, w)
        return min(cfg.NAME_RARITY_MAX_WEIGHT, max(1.0, best))


_freq_db: _FrequencyDB | None = None


def _get_freq_db() -> _FrequencyDB:
    global _freq_db
    if _freq_db is not None:
        return _freq_db

    source = cfg.NAME_FREQUENCY_SOURCE

    if source == "orcid":
        path = cfg.BASE_DIR / "index" / "name_frequencies.json"
        if path.exists():
            try:
                _freq_db = _OrcidFrequencyDB(path)
                return _freq_db
            except Exception as exc:
                logger.warning("Failed to load ORCID name frequencies (%s) — falling back to unweighted", exc)
        else:
            logger.warning(
                "ORCID name frequency file not found at %s "
                "— run build_name_frequencies.py or set NAME_FREQUENCY_SOURCE=none",
                path,
            )

    elif source == "names-dataset":
        try:
            _freq_db = _NamesDatasetDB()
            return _freq_db
        except ImportError:
            logger.warning(
                "names-dataset not installed — falling back to unweighted. "
                "Run: pip install names-dataset"
            )
        except Exception as exc:
            logger.warning("Failed to load names-dataset (%s) — falling back to unweighted", exc)

    _freq_db = _NullFrequencyDB()
    return _freq_db


# ── Author overlap score ───────────────────────────────────────────────────────

def author_overlap_score(authors_a: list[str], authors_b: list[str]) -> float:
    """
    Compute a normalised author overlap score in [0, 1].

    Algorithm:
    - Normalise both lists.
    - For each author in the SMALLER list, find its best match in the larger.
    - A match counts if similarity >= cfg.AUTHOR_MATCH_THRESHOLD.
    - Each matched name contributes its rarity weight rather than a flat 1.
    - Score = (sum of weights of matched names) / (sum of weights of all names
      in the smaller list) × coverage_weight.

    The coverage_weight dampens the score when both lists are very short
    (two single-author papers sharing that one author is less informative
    than six-author papers sharing five authors).
    """
    norm_a = normalise_author_list(authors_a)
    norm_b = normalise_author_list(authors_b)

    if not norm_a or not norm_b:
        return 0.0

    if len(norm_a) >= len(norm_b):
        smaller, larger = norm_b, norm_a
    else:
        smaller, larger = norm_a, norm_b

    freq_db = _get_freq_db()
    weights = [freq_db.get_weight(name) for name in smaller]
    total_weight = sum(weights)

    matched = [
        (name, w)
        for name, w in zip(smaller, weights)
        if _best_match_score(name, larger) >= cfg.AUTHOR_MATCH_THRESHOLD
    ]
    matched_weight = sum(w for _, w in matched)

    if total_weight == 0:
        return 0.0

    base_score = matched_weight / total_weight

    # Coverage weight: dampens score for very short author lists
    n_matched = len(matched)
    coverage_weight = 1.0 - (1.0 / (1.0 + n_matched))
    score = 0.70 * base_score + 0.30 * coverage_weight * base_score

    return round(min(score, 1.0), 4)


def matched_author_pairs(authors_a: list[str], authors_b: list[str]) -> list[tuple[str, str, float]]:
    """
    Return a list of (name_from_a, best_match_in_b, similarity) for all
    authors in A that have a match above threshold in B.
    Useful for generating human-readable rationale.
    """
    norm_a = normalise_author_list(authors_a)
    norm_b = normalise_author_list(authors_b)
    pairs = []
    for na, orig_a in zip(norm_a, authors_a):
        if not norm_b:
            continue
        sims = [(_name_similarity(na, nb), orig_b) for nb, orig_b in zip(norm_b, authors_b)]
        best_sim, best_name = max(sims, key=lambda x: x[0])
        if best_sim >= cfg.AUTHOR_MATCH_THRESHOLD:
            pairs.append((orig_a, best_name, round(best_sim, 3)))
    return pairs
