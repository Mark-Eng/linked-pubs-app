"""
app_runner.py
─────────────
Pipeline wrapper for the Streamlit app.

Extracts the file_check pipeline into a callable function that accepts a
config dict (from the UI) and returns raw Excel bytes, instead of writing
to a file path and reading CLI args.

Thread safety
─────────────
scoring.py, llm_scoring.py, and config.py use module-level globals.  A
single threading.Lock (_run_lock) serialises concurrent Streamlit sessions
for the prototype.  This is acceptable given the expected low concurrency
(handful of simultaneous users at most).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path

import config as cfg

logger = logging.getLogger(__name__)

_run_lock = threading.Lock()


def run_pipeline(
    input_path: str,
    app_config: dict,
    log_callback,
) -> bytes:
    """
    Run the file_check pipeline with settings from the Streamlit UI.

    Parameters
    ----------
    input_path   : path to the uploaded RIS file (in a temp directory)
    app_config   : dict with keys:
                     WEIGHT_TITLE, WEIGHT_AUTHOR, WEIGHT_EMBEDDING,
                     WEIGHT_YEAR, WEIGHT_LLM
                     use_country        : bool
                     COUNTRY_OVERLAP_BONUS, COUNTRY_DISJOINT_PENALTY : float
                     run_llm            : bool
                     LLM_PROVIDER       : "anthropic" | "openai"
                     llm_min, llm_max   : float | None
                     llm_prompt         : str | None  (None = default)
                     OUTPUT_THRESHOLD   : float
    log_callback : callable(str) — receives progress messages

    Returns
    -------
    bytes of the output Excel workbook
    """
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer

    import embeddings as emb_mod
    import llm_scoring as llm_mod
    import scoring as sc_mod
    import text_processing as tp_mod
    from embeddings import embed_texts, _text_for_embedding
    from file_check import _parse_ris
    from llm_scoring import score_pairs as llm_score_pairs
    from output_writer import write_excel
    from scoring import score_pair
    from text_processing import _tokenise

    # ── Patch config and acquire lock ─────────────────────────────────────────
    _run_lock.acquire()

    _saved = {
        "WEIGHT_TITLE":             cfg.WEIGHT_TITLE,
        "WEIGHT_AUTHOR":            cfg.WEIGHT_AUTHOR,
        "WEIGHT_EMBEDDING":         cfg.WEIGHT_EMBEDDING,
        "WEIGHT_YEAR":              cfg.WEIGHT_YEAR,
        "WEIGHT_LLM":               cfg.WEIGHT_LLM,
        "COUNTRY_OVERLAP_BONUS":    cfg.COUNTRY_OVERLAP_BONUS,
        "COUNTRY_DISJOINT_PENALTY": cfg.COUNTRY_DISJOINT_PENALTY,
        "LLM_PROVIDER":             cfg.LLM_PROVIDER,
        "ANTHROPIC_API_KEY":        cfg.ANTHROPIC_API_KEY,
        "OPENAI_API_KEY":           cfg.OPENAI_API_KEY,
    }
    _saved_prompt = llm_mod._PROMPT_TEMPLATE
    _saved_emb_client = emb_mod._openai_client

    try:
        emb_mod._openai_client = None  # force fresh client with current API key
        # Apply weight + country overrides
        cfg.WEIGHT_TITLE     = app_config["WEIGHT_TITLE"]
        cfg.WEIGHT_AUTHOR    = app_config["WEIGHT_AUTHOR"]
        cfg.WEIGHT_EMBEDDING = app_config["WEIGHT_EMBEDDING"]
        cfg.WEIGHT_YEAR      = app_config["WEIGHT_YEAR"]
        cfg.WEIGHT_LLM       = app_config["WEIGHT_LLM"]
        cfg.COUNTRY_OVERLAP_BONUS    = app_config["COUNTRY_OVERLAP_BONUS"]
        cfg.COUNTRY_DISJOINT_PENALTY = app_config["COUNTRY_DISJOINT_PENALTY"]

        # LLM provider
        cfg.LLM_PROVIDER = app_config["LLM_PROVIDER"]
        # Inject API keys from Streamlit secrets / env
        cfg.ANTHROPIC_API_KEY = app_config.get("ANTHROPIC_API_KEY", cfg.ANTHROPIC_API_KEY)
        cfg.OPENAI_API_KEY    = app_config.get("OPENAI_API_KEY",    cfg.OPENAI_API_KEY)

        # Custom LLM prompt
        if app_config.get("llm_prompt"):
            llm_mod._PROMPT_TEMPLATE = app_config["llm_prompt"]

        return _execute_pipeline(
            input_path=input_path,
            app_config=app_config,
            log_callback=log_callback,
            sc_mod=sc_mod,
            tp_mod=tp_mod,
            faiss=faiss,
            TfidfVectorizer=TfidfVectorizer,
            _tokenise=_tokenise,
            embed_texts=embed_texts,
            _text_for_embedding=_text_for_embedding,
            _parse_ris=_parse_ris,
            llm_score_pairs=llm_score_pairs,
            write_excel=write_excel,
            score_pair=score_pair,
        )

    finally:
        # Restore original config values
        for k, v in _saved.items():
            setattr(cfg, k, v)
        llm_mod._PROMPT_TEMPLATE = _saved_prompt
        emb_mod._openai_client = _saved_emb_client
        _run_lock.release()


def _execute_pipeline(
    input_path,
    app_config,
    log_callback,
    sc_mod,
    tp_mod,
    faiss,
    TfidfVectorizer,
    _tokenise,
    embed_texts,
    _text_for_embedding,
    _parse_ris,
    llm_score_pairs,
    write_excel,
    score_pair,
) -> bytes:
    """Internal: run the three-stage pipeline and return Excel bytes."""

    use_country     = app_config["use_country"]
    run_llm         = app_config["run_llm"]
    llm_min         = app_config.get("llm_min")
    llm_max         = app_config.get("llm_max")
    output_threshold = app_config["OUTPUT_THRESHOLD"]

    # ── Parse RIS ─────────────────────────────────────────────────────────────
    records = _parse_ris(input_path)
    if len(records) < 2:
        raise ValueError("The uploaded file contains fewer than 2 records with titles. At least 2 are needed.")
    log_callback(f"Parsed {len(records)} records.")

    records_by_id = {r["id"]: r for r in records}

    # ── Build TF-IDF model ────────────────────────────────────────────────────
    titles = [r["title"] for r in records if r.get("title")]
    tfidf = TfidfVectorizer(
        analyzer="word", tokenizer=_tokenise, token_pattern=None,
        min_df=1, sublinear_tf=True, ngram_range=(1, 2),
    )
    tfidf.fit(titles)
    tp_mod._tfidf_model = tfidf
    sc_mod._tfidf_model = tfidf

    # ── Stage 1: Embeddings + FAISS ───────────────────────────────────────────
    log_callback("Stage 1: generating embeddings…")
    texts = [_text_for_embedding(r) for r in records]
    try:
        vecs = embed_texts(texts)
        faiss.normalize_L2(vecs)
    except Exception as exc:
        import numpy as np
        log_callback(
            f"WARNING: Embeddings failed ({exc}). Results will not include embedding "
            f"similarity — all other scores are unaffected."
        )
        vecs = np.zeros((len(texts), cfg.EMBEDDING_DIM), dtype=np.float32)
    log_callback(f"Stage 1: embeddings ready ({len(vecs)} vectors). Building FAISS index…")

    index = faiss.IndexFlatIP(cfg.EMBEDDING_DIM)
    index.add(vecs)
    idx_id_map = [r["id"] for r in records]

    top_k = min(cfg.ANN_TOP_K, len(records))
    ann_sim_lookup: dict[tuple, float] = {}
    n_records = len(records)
    _report_every_rec = max(1, n_records // 4)
    for i, rec in enumerate(records):
        vec = vecs[i: i + 1]
        sims, idxs = index.search(vec, top_k)
        for sim, idx in zip(sims[0], idxs[0]):
            if idx >= 0:
                cand_id = idx_id_map[idx]
                if cand_id != rec["id"]:
                    pair = (max(rec["id"], cand_id), min(rec["id"], cand_id))
                    ann_sim_lookup[pair] = max(ann_sim_lookup.get(pair, 0.0), float(sim))
        if (i + 1) % _report_every_rec == 0:
            log_callback(f"Stage 1: searched {i + 1}/{n_records} records…")

    all_pairs = list(ann_sim_lookup.keys())
    log_callback(f"Stage 1 complete: {len(all_pairs)} candidate pairs found.")

    # ── Stage 2: Feature scoring ──────────────────────────────────────────────
    n_pairs = len(all_pairs)
    log_callback(f"Stage 2: scoring {n_pairs} candidate pairs…")
    stage2 = []
    _report_every_pair = max(1, n_pairs // 20)
    for i, pair in enumerate(all_pairs):
        id_a, id_b = pair
        rec_a = records_by_id.get(id_a)
        rec_b = records_by_id.get(id_b)
        if not rec_a or not rec_b:
            continue
        emb_sim = ann_sim_lookup.get(pair, 0.0)
        ps = score_pair(rec_a, rec_b, embedding_sim=emb_sim, llm_result=None, use_country=use_country)
        stage2.append(ps)
        if (i + 1) % _report_every_pair == 0:
            log_callback(f"Stage 2: {i + 1}/{n_pairs} pairs scored.")
            log_callback(f"PROGRESS:{i + 1}:{n_pairs}")
    log_callback(f"Stage 2 complete: {len(stage2)} pairs scored.")
    log_callback(f"PROGRESS:{n_pairs}:{n_pairs}")

    # ── Stage 3: LLM scoring ──────────────────────────────────────────────────
    if run_llm:
        llm_eligible = sorted(
            [ps for ps in stage2
             if (ps.author_score >= cfg.LLM_PREFILTER_THRESHOLD
                 or ps.title_score >= cfg.LLM_PREFILTER_THRESHOLD)
             and (llm_min is None or ps.composite >= llm_min)
             and (llm_max is None or ps.composite <= llm_max)],
            key=lambda s: s.author_score + s.title_score,
            reverse=True,
        )
        n_llm = len(llm_eligible)
        log_callback(f"Stage 3: sending {n_llm} pairs to LLM…")
        log_callback(f"PROGRESS:0:{n_llm}")
        if llm_eligible:
            llm_pairs = [(records_by_id[ps.id_a], records_by_id[ps.id_b]) for ps in llm_eligible]
            _report_every_llm = max(1, n_llm // 10)

            def _llm_progress(done: int, total: int) -> None:
                if done % _report_every_llm == 0 or done == total:
                    log_callback(f"Stage 3: {done}/{total} pairs scored by LLM.")
                    log_callback(f"PROGRESS:{done}:{total}")

            llm_results = llm_score_pairs(llm_pairs, progress_callback=_llm_progress)
            llm_map = {(ps.id_a, ps.id_b): res for ps, res in zip(llm_eligible, llm_results)}
        else:
            llm_map = {}

        final_scores = []
        n_final = len(stage2)
        _report_every_final = max(1, n_final // 20)
        log_callback(f"Stage 3: final scoring {n_final} pairs…")
        log_callback(f"PROGRESS:0:{n_final}")
        for i, ps in enumerate(stage2):
            llm_res = llm_map.get((ps.id_a, ps.id_b))
            emb_sim = ann_sim_lookup.get(
                (max(ps.id_a, ps.id_b), min(ps.id_a, ps.id_b)), 0.0
            )
            final_scores.append(
                score_pair(records_by_id[ps.id_a], records_by_id[ps.id_b], emb_sim, llm_res, use_country=use_country)
            )
            if (i + 1) % _report_every_final == 0 or (i + 1) == n_final:
                log_callback(f"Stage 3: {i + 1}/{n_final} pairs final-scored.")
                log_callback(f"PROGRESS:{i + 1}:{n_final}")
        log_callback("Stage 3 complete.")
    else:
        final_scores = stage2

    # ── Filter, sort ──────────────────────────────────────────────────────────
    final_scores = [ps for ps in final_scores if ps.composite >= output_threshold]
    final_scores.sort(key=lambda s: s.composite, reverse=True)
    log_callback(f"{len(final_scores)} pairs above threshold {output_threshold:.2f}.")

    # ── Write output to temp file, return bytes ───────────────────────────────
    import datetime
    metadata = {
        "Input file": Path(input_path).name,
        "Total records": len(records),
        "Candidate pairs checked": len(all_pairs),
        "LLM scoring": "Yes" if run_llm else "No",
        "Score threshold": output_threshold,
        "Run date": datetime.date.today().isoformat(),
    }

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        write_excel(final_scores, records_by_id, tmp_path, run_metadata=metadata)
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
