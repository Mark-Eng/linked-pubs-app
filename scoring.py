"""
scoring.py
──────────
Composite scoring pipeline.

For each candidate pair (retrieved via FAISS ANN), this module:
  1. Computes feature-based scores (author overlap, title similarity, year gap).
  2. Applies the country signal as a modifier.
  3. Optionally calls the LLM for semantic scoring.
  4. Combines all signals into a final composite score in [0, 1].

The composite formula is:
  raw = (WEIGHT_AUTHOR * author_score
       + WEIGHT_TITLE  * title_score
       + WEIGHT_LLM    * llm_score         # 0 if LLM not run
       + WEIGHT_EMBEDDING * embedding_sim
       + WEIGHT_YEAR   * year_score)
  composite = clip(raw + country_modifier, 0, 1)

A disjoint-country penalty is scaled by COUNTRY_PENALTY_LLM_RELIEF_FACTOR when
the LLM score is at least COUNTRY_PENALTY_LLM_RELIEF_THRESHOLD (recorded as
PairScore.country_penalty_relaxed).

When LLM scoring is skipped (e.g. pair below pre-filter threshold),
its weight is redistributed to the other components proportionally.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np

import config as cfg
from author_matching import author_overlap_score, matched_author_pairs
from country_extraction import EXTRACTION_VERSION, country_signal, extract_countries
from text_processing import title_similarity, title_similarity_fallback
from trial_registration import extract_trial_ids, trial_id_signal
from llm_scoring import ScoringResult

logger = logging.getLogger(__name__)

# Set by the caller before scoring: every entry point fits or loads a TF-IDF
# model for the corpus in play (the base index's model for comparisons against
# a base, a freshly fitted one for within-file checks) and assigns it here.
# When it is None, title scoring degrades to the token-overlap fallback rather
# than silently using a model fitted on the wrong corpus.
_tfidf_model = None
_tfidf_warned = False


def _get_tfidf():
    global _tfidf_warned
    if _tfidf_model is None and not _tfidf_warned:
        _tfidf_warned = True
        logger.warning(
            "No TF-IDF model set — title similarity will use the token-overlap "
            "fallback. Callers should assign scoring._tfidf_model first."
        )
    return _tfidf_model


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PairScore:
    id_a: int | str
    id_b: int | str
    composite: float
    author_score: float
    title_score: float
    embedding_sim: float
    year_score: float
    country_signal_value: int          # -1, 0, or +1
    country_description: str
    llm_score: float | None            # None if LLM not called
    llm_rationale: str
    llm_key_evidence: list[str]        = field(default_factory=list)
    llm_confidence: str                = ""
    llm_error: str | None             = None
    matched_authors: list[tuple]       = field(default_factory=list)
    countries_a: list[str]             = field(default_factory=list)
    countries_b: list[str]             = field(default_factory=list)
    trial_id_signal_value: int         = 0   # +1 shared, -1 conflict, 0 absent
    trial_id_description: str          = ""
    country_penalty_relaxed: bool      = False   # disjoint penalty eased by a high LLM score


# ── Individual feature scorers ─────────────────────────────────────────────────

def scoring_config_snapshot(name_frequency: str, **extra) -> dict:
    """
    Every setting that can change a PairScore, for cache invalidation.

    Callers that cache computed scores must store this alongside them and
    discard the cache when it differs. Without it, changing a weight in .env
    and re-running the same input returns the *old* scores from cache with no
    indication anything is stale — which matters more once weights and
    thresholds are exposed as admin settings in a web interface.

    name_frequency is passed in rather than read here so that building a
    snapshot cannot trigger the ~1.7 GB name-frequency load as a side effect;
    pass the value returned by author_matching.init_frequency_db(). The
    *effective* source belongs in the key because a run that silently fell
    back to unweighted matching produces different author scores from a
    weighted one.

    OUTPUT_THRESHOLD and the LLM pre-filter settings are deliberately absent:
    they filter and select pairs after scoring rather than changing any score.
    """
    snapshot = {
        "weight_author":            cfg.WEIGHT_AUTHOR,
        "weight_title":             cfg.WEIGHT_TITLE,
        "weight_llm":               cfg.WEIGHT_LLM,
        "weight_embedding":         cfg.WEIGHT_EMBEDDING,
        "weight_year":              cfg.WEIGHT_YEAR,
        "country_overlap_bonus":    cfg.COUNTRY_OVERLAP_BONUS,
        "country_disjoint_penalty": cfg.COUNTRY_DISJOINT_PENALTY,
        "country_penalty_llm_relief_threshold": cfg.COUNTRY_PENALTY_LLM_RELIEF_THRESHOLD,
        "country_penalty_llm_relief_factor":    cfg.COUNTRY_PENALTY_LLM_RELIEF_FACTOR,
        "country_extraction_version": EXTRACTION_VERSION,
        "trial_id_match_floor":    cfg.TRIAL_ID_MATCH_FLOOR,
        "trial_id_conflict_ceil":   cfg.TRIAL_ID_CONFLICT_CEIL,
        "year_gap_threshold":       cfg.YEAR_GAP_THRESHOLD,
        "year_gap_penalty":         cfg.YEAR_GAP_PENALTY,
        "author_match_threshold":   cfg.AUTHOR_MATCH_THRESHOLD,
        "author_initial_matching":  cfg.AUTHOR_INITIAL_MATCHING,
        "name_rarity_max_weight":   cfg.NAME_RARITY_MAX_WEIGHT,
        "name_frequency":           name_frequency,
        "ann_top_k":                cfg.ANN_TOP_K,
    }
    snapshot.update(extra)
    return snapshot


def describe_config_change(cached: dict | None, current: dict) -> str:
    """
    Human-readable summary of why a cached snapshot no longer applies.

    Names the settings that actually changed, so a cache miss is explicable
    rather than mysterious.
    """
    if not cached:
        return "no scoring configuration recorded in cache"
    differences = [
        f"{key}: {cached.get(key, '(absent)')!r} → {current[key]!r}"
        for key in sorted(current)
        if cached.get(key) != current[key]
    ]
    removed = [key for key in sorted(cached) if key not in current]
    if removed:
        differences.append(f"no longer tracked: {', '.join(removed)}")
    return "; ".join(differences) or "settings match"


# ── Individual feature scorers ─────────────────────────────────────────────────

def _year_score(year_a, year_b) -> float:
    """1.0 if same year or either missing; decays with gap; penalty if gap > threshold."""
    try:
        ya, yb = int(year_a), int(year_b)
    except (TypeError, ValueError):
        return 0.5   # missing year: neutral
    gap = abs(ya - yb)
    if gap == 0:
        return 1.0
    if gap > cfg.YEAR_GAP_THRESHOLD:
        return max(0.0, 1.0 - cfg.YEAR_GAP_PENALTY - (gap - cfg.YEAR_GAP_THRESHOLD) * 0.02)
    return max(0.0, 1.0 - gap * 0.08)


def _resolve_countries(record: dict) -> list[str]:
    """
    Get the country list for a record: the country field if populated,
    otherwise extracted from title + abstract.

    This is a per-pair fallback, and deliberately regex-only. The LLM tier
    (country_extraction.resolve_countries_with_fallback) belongs upstream,
    where each record is resolved exactly once — calling it here would mean
    an API request per *pair*, so a record appearing in 100 candidate pairs
    would be sent 100 times.

    Every CLI pre-resolves before scoring, so in those paths this returns the
    stored value and the fallback never fires. It exists for callers that
    score a record they have just been handed — api_service scoring a record
    the user is still typing — and extraction is memoised so repeated pairs
    for the same record cost one pass.
    """
    db_countries = record.get("country") or []
    if db_countries:
        return db_countries
    # Extract from text
    text = " ".join(filter(None, [record.get("title"), record.get("abstract")]))
    return extract_countries(text)


def _compute_composite(
    author_sc: float,
    title_sc: float,
    emb_sc: float,
    year_sc: float,
    country_mod: float,
    llm_sc: float | None,
) -> float:
    """Compute composite score, redistributing LLM weight if LLM not run."""
    if llm_sc is None:
        # Redistribute LLM weight proportionally among the other components
        scale = 1.0 / (1.0 - cfg.WEIGHT_LLM)
        raw = (
            cfg.WEIGHT_AUTHOR    * author_sc * scale
            + cfg.WEIGHT_TITLE   * title_sc  * scale
            + cfg.WEIGHT_EMBEDDING * emb_sc  * scale
            + cfg.WEIGHT_YEAR    * year_sc   * scale
        )
    else:
        raw = (
            cfg.WEIGHT_AUTHOR    * author_sc
            + cfg.WEIGHT_TITLE   * title_sc
            + cfg.WEIGHT_LLM     * llm_sc
            + cfg.WEIGHT_EMBEDDING * emb_sc
            + cfg.WEIGHT_YEAR    * year_sc
        )
    return float(np.clip(raw + country_mod, 0.0, 1.0))


# ── Main scoring function ──────────────────────────────────────────────────────

def score_pair(
    rec_a: dict,
    rec_b: dict,
    embedding_sim: float = 0.0,
    llm_result: ScoringResult | None = None,
    *,
    use_country: bool,
) -> PairScore:
    """
    Compute a full PairScore for two publication records.

    Parameters
    ----------
    rec_a, rec_b    : publication dicts (must have id, title, authors, year, country)
    embedding_sim   : cosine similarity from FAISS (already computed during retrieval)
    llm_result      : pre-computed LLM result, or None if LLM was not called
    use_country     : whether the country modifier applies. Required and
                      keyword-only rather than defaulting to True: the default
                      caused two real bugs, applying the -0.40 country-mismatch
                      penalty to SR pairs in batch_check and api_service, where
                      country is not a meaningful signal. Pass
                      `use_country=(dataset == "ie")`.
    """
    # ── Feature scores ─────────────────────────────────────────────────────────
    author_sc = author_overlap_score(
        rec_a.get("authors", []),
        rec_b.get("authors", []),
    )

    tfidf = _get_tfidf()
    if tfidf:
        title_sc = title_similarity(
            rec_a.get("title", ""),
            rec_b.get("title", ""),
            tfidf,
        )
    else:
        title_sc = title_similarity_fallback(
            rec_a.get("title", ""),
            rec_b.get("title", ""),
        )

    year_sc = _year_score(rec_a.get("year"), rec_b.get("year"))

    countries_a = _resolve_countries(rec_a)
    countries_b = _resolve_countries(rec_b)
    if use_country:
        c_signal, c_desc = country_signal(countries_a, countries_b)
        country_mod = {
             1: cfg.COUNTRY_OVERLAP_BONUS,
             0: 0.0,
            -1: cfg.COUNTRY_DISJOINT_PENALTY,
        }[c_signal]
    else:
        c_signal, c_desc, country_mod = 0, "country signal disabled", 0.0

    # ── LLM result ─────────────────────────────────────────────────────────────
    llm_sc = None
    llm_rationale = ""
    llm_evidence = []
    llm_confidence = ""
    llm_error = None

    if llm_result is not None:
        if llm_result.error:
            llm_error = llm_result.error
        else:
            llm_sc = llm_result.score
            llm_rationale = llm_result.rationale
            llm_evidence = llm_result.key_evidence
            llm_confidence = llm_result.confidence

    # A near-certain LLM verdict outweighs a country mismatch, which is more
    # often a country fill-in error than a genuinely different study country.
    # Applied before the trial-ID override so a conflicting ID still caps.
    country_penalty_relaxed = (
        c_signal == -1
        and llm_sc is not None
        and llm_sc >= cfg.COUNTRY_PENALTY_LLM_RELIEF_THRESHOLD
    )
    if country_penalty_relaxed:
        country_mod *= cfg.COUNTRY_PENALTY_LLM_RELIEF_FACTOR

    # ── Trial registration number signal ───────────────────────────────────────
    trial_ids_a = extract_trial_ids(rec_a.get("abstract", "") or "")
    trial_ids_b = extract_trial_ids(rec_b.get("abstract", "") or "")
    tr_signal, tr_desc = trial_id_signal(trial_ids_a, trial_ids_b)

    # ── Composite ──────────────────────────────────────────────────────────────
    composite = _compute_composite(
        author_sc, title_sc, embedding_sim, year_sc, country_mod, llm_sc
    )

    # Near-override: shared trial ID floors the composite; conflicting IDs cap it
    if tr_signal == 1:
        composite = max(composite, cfg.TRIAL_ID_MATCH_FLOOR)
    elif tr_signal == -1:
        composite = min(composite, cfg.TRIAL_ID_CONFLICT_CEIL)

    return PairScore(
        id_a=rec_a["id"],
        id_b=rec_b["id"],
        composite=round(composite, 4),
        author_score=round(author_sc, 4),
        title_score=round(title_sc, 4),
        embedding_sim=round(embedding_sim, 4),
        year_score=round(year_sc, 4),
        country_signal_value=c_signal,
        country_description=c_desc,
        llm_score=round(llm_sc, 4) if llm_sc is not None else None,
        llm_rationale=llm_rationale,
        llm_key_evidence=llm_evidence,
        llm_confidence=llm_confidence,
        llm_error=llm_error,
        matched_authors=matched_author_pairs(
            rec_a.get("authors", []),
            rec_b.get("authors", []),
        ),
        countries_a=countries_a,
        countries_b=countries_b,
        trial_id_signal_value=tr_signal,
        trial_id_description=tr_desc,
        country_penalty_relaxed=country_penalty_relaxed,
    )


def score_candidates(
    query_record: dict,
    candidates: list[tuple[dict, float]],   # (record, embedding_sim)
    run_llm: bool = True,
    *,
    use_country: bool,
) -> list[PairScore]:
    """
    Score a list of (candidate_record, embedding_sim) tuples against a query record.

    Applies the LLM pre-filter: only the top LLM_TOP_K candidates (by stage-2
    score) that exceed LLM_PREFILTER_THRESHOLD are sent to the LLM.
    Returns all PairScores sorted by composite score descending.

    use_country must be False for SR datasets, matching the CLI scripts —
    country overlap is only a meaningful signal for impact evaluations.
    """
    from llm_scoring import score_pairs

    # ── Stage-2: feature scores without LLM ───────────────────────────────────
    stage2 = [
        score_pair(query_record, cand, embedding_sim=emb_sim, llm_result=None,
                   use_country=use_country)
        for cand, emb_sim in candidates
    ]

    if not run_llm:
        stage2.sort(key=lambda s: s.composite, reverse=True)
        return [s for s in stage2 if s.composite >= cfg.OUTPUT_THRESHOLD]

    # ── LLM pre-filter ────────────────────────────────────────────────────────
    candidate_map = {cand["id"]: cand for cand, _ in candidates}
    llm_eligible = sorted(
        [s for s in stage2
         if (s.author_score >= cfg.LLM_PREFILTER_THRESHOLD
             or s.title_score >= cfg.LLM_PREFILTER_THRESHOLD)],
        key=lambda s: s.author_score + s.title_score,
        reverse=True,
    )[: cfg.LLM_TOP_K]

    if llm_eligible:
        llm_pairs = [
            (query_record, candidate_map[s.id_b]) for s in llm_eligible
        ]
        llm_results = score_pairs(llm_pairs)
        llm_map = {s.id_b: llm_res for s, llm_res in zip(llm_eligible, llm_results)}
    else:
        llm_map = {}

    # ── Final composite scores ─────────────────────────────────────────────────
    final_scores = []
    for stage2_score in stage2:
        cand = candidate_map[stage2_score.id_b]
        emb_sim = stage2_score.embedding_sim
        llm_res = llm_map.get(stage2_score.id_b)
        final_scores.append(
            score_pair(query_record, cand, emb_sim, llm_res, use_country=use_country)
        )

    final_scores.sort(key=lambda s: s.composite, reverse=True)
    return [s for s in final_scores if s.composite >= cfg.OUTPUT_THRESHOLD]
