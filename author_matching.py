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
  "none"          — all matched names count equally
  "names-dataset" — uses the names-dataset package for global name frequencies
  "orcid"         — uses a pre-built frequency table from the ORCID public data
                    dump (index/name_frequencies.json, built by build_name_frequencies.py)

Loading the source is not free or certain: names-dataset expands to ~1.7 GB
resident and takes ~5s, and a memory-capped container can fail to load it.
Because a failure disables weighting — changing the author component, which
carries 30% of the composite — it must not happen invisibly:

* Call init_frequency_db() at start-up rather than letting the first author
  comparison trigger the load, so a failure happens before any embedding or
  LLM spend rather than after it.
* Record active_frequency_source() with the results. It distinguishes a
  deliberately unweighted run from one that wanted weighting and didn't get
  it.
* Set NAME_FREQUENCY_STRICT=true in served or automated deployments to turn a
  load failure into a hard error instead of a silent fallback.
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata

from rapidfuzz.distance import JaroWinkler

import config as cfg
from errors import ConfigurationError

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

def _comma_surname(name: str) -> str | None:
    """
    Normalised surname of a "Surname, Given" name, or None if there is no comma
    (in which case which token is the surname is unknown).
    """
    if "," not in name:
        return None
    return _normalise_name(name.split(",", 1)[0]) or None


def _name_similarity(
    name_a: str,
    name_b: str,
    surname_a: str | None = None,
    surname_b: str | None = None,
) -> float:
    """
    Jaro-Winkler similarity between two normalised name strings.

    Also compares token-sorted forms to handle transpositions where surname
    and given name are swapped between records (e.g. "shilpa aggarwal" vs
    "aggarwal shilpa"), and, when cfg.AUTHOR_INITIAL_MATCHING is set, applies
    the initial-aware rule in _initial_aware_similarity, which uses the
    comma-marked surnames (see _comma_surname) when available.  Returns the
    highest of the scores.
    """
    if not name_a or not name_b:
        return 0.0
    score_as_is = JaroWinkler.normalized_similarity(name_a, name_b)
    sorted_a = " ".join(sorted(name_a.split()))
    sorted_b = " ".join(sorted(name_b.split()))
    score_sorted = JaroWinkler.normalized_similarity(sorted_a, sorted_b)
    score = max(score_as_is, score_sorted)
    if cfg.AUTHOR_INITIAL_MATCHING:
        score = max(score, _initial_aware_similarity(name_a, name_b, surname_a, surname_b))
    return score


# A full token must be this close to its counterpart to count as the same name
# part in the initial-aware rule. Stricter than AUTHOR_MATCH_THRESHOLD because
# it applies to a single token, not a whole name.
_INITIAL_RULE_TOKEN_THRESHOLD = 0.92

# Score given to a name pair that passes the initial-aware rule: just above the
# default AUTHOR_MATCH_THRESHOLD (0.88), so these matches are recognisable in
# the "Matched Authors" output column.
_INITIAL_RULE_SCORE = 0.90


def _initial_rule_token_pair(t1: str, t2: str) -> str | None:
    """'full' if two full tokens agree, 'initial' if one is an initial of the other."""
    if len(t1) > 1 and len(t2) > 1:
        similar = JaroWinkler.normalized_similarity(t1, t2) >= _INITIAL_RULE_TOKEN_THRESHOLD
        return "full" if similar else None
    return "initial" if t1[0] == t2[0] else None


def _initial_aware_similarity(
    name_a: str,
    name_b: str,
    surname_a: str | None = None,
    surname_b: str | None = None,
) -> float:
    """
    Match names where one side abbreviates a part to an initial.

    The token-sort comparison only catches "faiela c" vs "candido faiela" when
    the initial happens to sort after the given name ("david s" vs "scott david"
    matches; "faiela c" vs "candido faiela" does not). This rule removes that
    alphabet dependence.

    Every token of the shorter name must pair with a distinct token of the
    longer one, in any order: two full tokens pair if near-identical, and a
    single letter pairs with any token starting with it. At least one pair
    must be two full tokens, so initials alone never match, and both names
    need at least two tokens. Tokens of the longer name may be left unpaired
    only when both names are comma-anchored (below), and then only initials:
    otherwise "Lowe, John B." would match "Busienei, John" (busienei↔b,
    john↔john, lowe ignored) and "Ahmed U. A." would match "Arshad, Ahmed".

    When both names are "Surname, Given", the surnames must also agree and the
    first given names must be compatible, so a surname cannot pair with a given
    name ("Thuy, Tran T." ≠ "Tran, Thanh T.") and a middle initial cannot stand
    in for the first given name ("Li, Z. S." ≠ "Li, Shixiang"). This rejects
    true matches whose source has surname and given name swapped; the
    order-free rule still applies when either name lacks a comma.

    Returns _INITIAL_RULE_SCORE on success, else 0.0. Only applies when at
    least one side contains a single-letter token.
    """
    tokens_a, tokens_b = name_a.split(), name_b.split()
    if not any(len(t) == 1 for t in tokens_a + tokens_b):
        return 0.0
    # A one-token name ("Li,") would match every name containing that token.
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return 0.0

    anchored = bool(surname_a and surname_b)
    if anchored:
        if JaroWinkler.normalized_similarity(surname_a, surname_b) < _INITIAL_RULE_TOKEN_THRESHOLD:
            return 0.0
        # _normalise_name puts the surname first, so the given names follow it.
        given_a = tokens_a[len(surname_a.split()):]
        given_b = tokens_b[len(surname_b.split()):]
        if given_a and given_b and not _initial_rule_token_pair(given_a[0], given_b[0]):
            return 0.0

    if len(tokens_a) > len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a
    pair_kind = _initial_rule_token_pair

    # Backtracking search over assignments; names have few tokens, so this is cheap.
    def assign(i: int, used: frozenset[int], has_full: bool) -> bool:
        if i == len(tokens_a):
            leftover = [t for j, t in enumerate(tokens_b) if j not in used]
            if anchored:
                # Unmatched middle initials are fine once surname and first given name agree.
                return has_full and all(len(t) == 1 for t in leftover)
            return has_full and not leftover
        for j, token_b in enumerate(tokens_b):
            if j in used:
                continue
            kind = pair_kind(tokens_a[i], token_b)
            if kind and assign(i + 1, used | {j}, has_full or kind == "full"):
                return True
        return False

    return _INITIAL_RULE_SCORE if assign(0, frozenset(), False) else 0.0


def _parse_names(authors: list[str]) -> list[tuple[str, str | None]]:
    """(normalised name, comma-marked surname) for each non-blank author."""
    return [(_normalise_name(a), _comma_surname(a)) for a in authors if a and a.strip()]


def _best_match_score(
    name: tuple[str, str | None], candidates: list[tuple[str, str | None]]
) -> float:
    """Return the highest similarity score for `name` against any candidate."""
    if not candidates:
        return 0.0
    norm, surname = name
    return max(_name_similarity(norm, c, surname, c_surname) for c, c_surname in candidates)


# ── Name frequency / rarity weighting ─────────────────────────────────────────

class _FrequencyDB:
    """Base class for name frequency sources."""
    def get_weight(self, _normalised_name: str) -> float:
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
        from names_dataset import NameDataset  # type: ignore[import-not-found]
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
_freq_source: str = "not loaded"


def _resolve_freq_db() -> tuple[_FrequencyDB, str, str | None]:
    """
    Load the configured frequency source.

    Returns (database, effective_source, failure_reason). failure_reason is
    None on success; when set, the caller decides whether to degrade to
    unweighted matching or refuse to run (see cfg.NAME_FREQUENCY_STRICT).
    """
    source = cfg.NAME_FREQUENCY_SOURCE

    if source == "orcid":
        path = cfg.ORCID_FREQUENCIES_PATH
        if not path.exists():
            return _NullFrequencyDB(), "none", (
                f"ORCID name frequency file not found at {path} — run "
                f"build_name_frequencies.py or set NAME_FREQUENCY_SOURCE=none"
            )
        try:
            return _OrcidFrequencyDB(path), "orcid", None
        except (OSError, ValueError, KeyError) as exc:
            # OSError: unreadable file; ValueError (incl. JSONDecodeError): malformed
            # JSON; KeyError: missing "given"/"family"/"total" keys. We control this
            # file's format (built by build_name_frequencies.py), so these cover the
            # realistic failure modes without masking genuinely unexpected errors.
            return _NullFrequencyDB(), "none", f"failed to load ORCID name frequencies: {exc}"

    if source == "names-dataset":
        try:
            return _NamesDatasetDB(), "names-dataset", None
        except ImportError:
            return _NullFrequencyDB(), "none", (
                "names-dataset not installed — run: pip install names-dataset"
            )
        except Exception as exc:  # noqa: BLE001 — third-party init, and the most
            # likely failure is MemoryError: the package expands to ~1.7 GB
            # resident, so a memory-capped container lands here.
            return _NullFrequencyDB(), "none", f"failed to load names-dataset: {exc}"

    if source != "none":
        return _NullFrequencyDB(), "none", (
            f"unknown NAME_FREQUENCY_SOURCE {source!r} — "
            f"expected 'none', 'names-dataset', or 'orcid'"
        )

    return _NullFrequencyDB(), "none", None


def init_frequency_db() -> str:
    """
    Load the name-frequency database now and report the source in effect.

    Call this once at start-up, before any expensive stage. Two reasons:

    * names-dataset takes ~5s and ~1.7 GB. Loading it lazily on the first
      author comparison means a memory failure surfaces *after* embeddings
      have been generated and paid for.
    * A failure silently disables rarity weighting, changing the author score
      that carries 30% of the composite. Callers should record the returned
      value alongside their results so a run scored without weighting can be
      identified afterwards.

    Raises ConfigurationError instead of degrading when
    cfg.NAME_FREQUENCY_STRICT is set — see that setting's comment for why a
    served deployment wants that.
    """
    global _freq_db, _freq_source

    requested = cfg.NAME_FREQUENCY_SOURCE
    db, effective, reason = _resolve_freq_db()

    if reason is not None:
        if cfg.NAME_FREQUENCY_STRICT:
            raise ConfigurationError(
                f"Author name weighting source {requested!r} is unavailable: {reason}. "
                f"NAME_FREQUENCY_STRICT is set, so refusing to run with weighting "
                f"silently disabled."
            )
        logger.warning(
            "%s — falling back to UNWEIGHTED author matching. Scores will differ "
            "from a weighted run; set NAME_FREQUENCY_STRICT=true to fail instead.",
            reason,
        )
        effective = f"none (requested {requested}, unavailable)"

    _freq_db = db
    _freq_source = effective
    logger.info("Author name weighting: %s", effective)
    return effective


def active_frequency_source() -> str:
    """
    The name-frequency source actually in effect, loading it if necessary.

    Worth recording in run metadata: it distinguishes a deliberately
    unweighted run ("none") from one that wanted weighting and didn't get it
    ("none (requested names-dataset, unavailable)").
    """
    if _freq_db is None:
        init_frequency_db()
    return _freq_source


def _get_freq_db() -> _FrequencyDB:
    if _freq_db is None:
        init_frequency_db()
    assert _freq_db is not None   # narrowed by init_frequency_db above
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
    parsed_a = _parse_names(authors_a)
    parsed_b = _parse_names(authors_b)

    if not parsed_a or not parsed_b:
        return 0.0

    if len(parsed_a) >= len(parsed_b):
        smaller, larger = parsed_b, parsed_a
    else:
        smaller, larger = parsed_a, parsed_b

    freq_db = _get_freq_db()
    weights = [freq_db.get_weight(norm) for norm, _ in smaller]
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
    paired_a = [(_normalise_name(a), _comma_surname(a), a) for a in authors_a if a and a.strip()]
    paired_b = [(_normalise_name(b), _comma_surname(b), b) for b in authors_b if b and b.strip()]
    pairs = []
    for na, sa, orig_a in paired_a:
        if not paired_b:
            continue
        sims = [(_name_similarity(na, nb, sa, sb), orig_b) for nb, sb, orig_b in paired_b]
        best_sim, best_name = max(sims, key=lambda x: x[0])
        if best_sim >= cfg.AUTHOR_MATCH_THRESHOLD:
            pairs.append((orig_a, best_name, round(best_sim, 3)))
    return pairs
