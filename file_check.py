"""
file_check.py
─────────────
Standalone linked-publication detection for RIS or Excel files.
No database connection required.

Usage:
  python file_check.py --input references.ris [--no-llm] [--threshold 0.4] [--output results.xlsx]
  python file_check.py --input references.xlsx [--sheet "Sheet1"] [--no-llm]
  python file_check.py --input references.ris --no-cache   # force re-embedding

Input format
─────────────
RIS file: standard RIS format. Fields used:
  TI / T1  — title
  AU / A1  — author (one per line)
  AB / N2  — abstract
  PY / Y1  — year
  ID       — record ID (generated if absent)

Excel file: must have columns named (case-insensitive):
  title (required), authors (semicolon-delimited), abstract, year
  A numeric ID column is auto-generated.

Any record with no country — all RIS records, and Excel records with no
country column or a blank cell — has it extracted from the title and abstract.
Extraction uses the regex gazetteer, with an LLM fallback for records where
the gazetteer finds nothing (IE datasets only; disable with --no-llm-country).

Output
──────
Excel workbook with Summary and Pairs sheets (same format as batch_check.py).

Caches
──────
Three caches sit alongside the input file, so a re-run skips the expensive
stages instead of paying for them again:

  <input>.embedcache.json    OpenAI embeddings, keyed on the record ids and
                             the embedding text format (EMBED_VERSION)
  <input>.countrycache.json  countries extracted from record text, keyed on the
                             record ids and the extraction mode
  <input>.stage2cache.json   candidate pairs and their stage-2 scores

All three are invalidated when the input file is modified or its record ids
change. The stage-2 cache additionally records every setting that feeds into a
composite score — the weights, the thresholds, the country mode, and the author
name weighting actually in effect — and regenerates when any of them differs,
naming what changed. Without that, editing a weight in .env and re-running the
same file would silently return the previous scores.

Use --no-cache to force regeneration of all three.
"""

import argparse
import base64
import dataclasses
import datetime
import json
import logging
from pathlib import Path

import config as cfg
from errors import ConfigurationError, InputError, cli_entry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Input parsers ──────────────────────────────────────────────────────────────

def _parse_ris(path: str) -> list[dict]:
    """Parse a RIS file into a list of record dicts."""
    try:
        import rispy
    except ImportError as exc:
        raise ConfigurationError("rispy not installed — run: pip install rispy") from exc

    with open(path, encoding="utf-8-sig") as fh:
        entries = rispy.load(fh)

    records = []
    for i, entry in enumerate(entries):
        title = entry.get("title") or entry.get("primary_title") or ""
        authors_raw = entry.get("authors") or entry.get("first_authors") or []
        abstract = entry.get("abstract") or ""

        year_raw = entry.get("year") or entry.get("publication_year") or ""
        try:
            year = int(str(year_raw)[:4])
        except (ValueError, TypeError):
            year = None

        # Use U1 as the record ID when the entry is tagged as an Eppi Reviewer export.
        # rispy routes unrecognised tags (including U1) into entry["unknown_tag"][tag][0].
        keywords = entry.get("keywords") or []
        is_eppi = any(str(k).strip().lower() == "eppi-reviewer" for k in keywords)
        if is_eppi:
            u1_vals = (entry.get("unknown_tag") or {}).get("U1") or []
            u1 = str(u1_vals[0]).strip() if u1_vals else ""
            rec_id = u1 if u1 else (entry.get("id") or f"ris_{i}")
        else:
            rec_id = entry.get("id") or f"ris_{i}"

        # Some RIS exports (e.g. this EPPI-Reviewer format) put every author on a
        # single AU line as a semicolon-joined string instead of one AU line per
        # author. Split on ";" to recover individual names either way — a plain
        # name with no embedded ";" just passes through unchanged.
        authors = [
            part.strip()
            for a in authors_raw
            for part in str(a).split(";")
            if part.strip()
        ]

        records.append({
            "id":             rec_id,
            "title":          title.strip(),
            "authors":        authors,
            "abstract":       abstract.strip(),
            "year":           year,
            "country":        [],
            "source_project": "NA (data sourced from RIS file)",
            "status":         "NA (data sourced from RIS file)",
        })

    logger.info("Parsed %d records from RIS file", len(records))
    return [r for r in records if r["title"]]


def _parse_excel(path: str, sheet: str | None = None) -> list[dict]:
    """Parse an Excel file into a list of record dicts."""
    import pandas as pd

    df = pd.read_excel(path, sheet_name=sheet or 0)
    df.columns = df.columns.str.lower().str.strip()

    if "title" not in df.columns:
        found = ", ".join(str(c) for c in df.columns[:12]) or "(no columns)"
        raise InputError(
            f"{Path(path).name}: Excel file must have a 'title' column. "
            f"Columns found: {found}"
        )

    # Detect optional column aliases for flexibility with different export formats
    year_col = "year" if "year" in df.columns else (
        "year_of_publication" if "year_of_publication" in df.columns else None
    )
    has_id_col = "id" in df.columns
    has_country_col = "country" in df.columns
    has_source_project_col = "source_project" in df.columns
    has_status_col = "status" in df.columns

    def _author_names(row) -> list[str]:
        authors_raw = str(row.get("authors", "")).strip()
        return [a.strip() for a in authors_raw.split(";") if a.strip() and authors_raw != "nan"]

    # Some exports (e.g. the 3ie DEP) use a layout with
    # one row per author, where only the FIRST row of each publication has the
    # title/abstract/year/etc. populated — subsequent rows for the same
    # publication have a blank title and just one more author in the authors
    # column. Treat a blank-title row as a continuation of the previous record
    # (append its author) rather than as a separate (and skippable) record.
    records = []
    current = None
    n_continuation_rows = 0
    for i, row in df.iterrows():
        title = str(row.get("title", "")).strip()

        if not title or title == "nan":
            if current is not None:
                current["authors"].extend(_author_names(row))
                n_continuation_rows += 1
            continue

        if current is not None:
            records.append(current)
            current = None

        authors = _author_names(row)

        abstract = str(row.get("abstract", "")).strip()
        if abstract == "nan":
            abstract = ""

        year_raw = str(row.get(year_col, "")) if year_col else ""
        try:
            year = int(float(year_raw)) if year_raw not in ("", "nan") else None
        except (ValueError, TypeError):
            year = None

        # Use the 'id' column value when present; fall back to row index
        if has_id_col:
            try:
                rec_id = int(row["id"])
            except (ValueError, TypeError):
                rec_id = int(i)
        else:
            rec_id = int(i)

        # Parse country into a list (semicolon-delimited or single value)
        if has_country_col:
            country_raw = str(row.get("country", "")).strip()
            country = [c.strip() for c in country_raw.split(";") if c.strip() and country_raw != "nan"]
        else:
            country = []

        if has_source_project_col:
            sp = str(row.get("source_project", "")).strip()
            source_project = sp if sp and sp != "nan" else None
        else:
            source_project = None

        if has_status_col:
            st = str(row.get("status", "")).strip()
            status = st if st and st != "nan" else None
        else:
            status = None

        current = {
            "id":             rec_id,
            "title":          title,
            "authors":        authors,
            "abstract":       abstract,
            "year":           year,
            "country":        country,
            "source_project": source_project,
            "status":         status,
        }

    if current is not None:
        records.append(current)

    logger.info(
        "Parsed %d records from Excel file (%d continuation rows merged in as additional authors)",
        len(records), n_continuation_rows,
    )
    return records


def load_records(input_path: str, sheet: str | None = None) -> list[dict]:
    """
    Parse an input file into record dicts, dispatching on extension.

    Raises InputError for an unsupported extension or malformed content, so
    that callers other than the CLI — a web import adapter, for instance —
    can report the problem instead of being terminated by it.
    """
    if not Path(input_path).exists():
        raise InputError(f"Input file not found: {input_path}")

    ext = Path(input_path).suffix.lower()
    if ext in (".ris", ".txt"):
        return _parse_ris(input_path)
    if ext in (".xlsx", ".xls"):
        return _parse_excel(input_path, sheet)
    raise InputError(
        f"Unsupported file type '{ext}' for {Path(input_path).name} "
        f"(use .ris, .txt, .xlsx, or .xls)"
    )


# ── Country extraction ─────────────────────────────────────────────────────────

def _load_countries(records: list[dict], input_path: str, no_cache: bool, use_llm: bool) -> None:
    """
    Populate rec["country"] in place for records that have none, extracting
    country names from the title + abstract. Records with a populated country
    field keep it. Applies to every input format: RIS has no country field,
    and Excel files may lack the column or leave cells blank.

    Results are cached to <input>.countrycache.json so the LLM fallback isn't
    paid for again on a re-run. The cache is invalidated if the input file is
    newer, the record IDs differ, or the extraction mode changed.
    """
    from country_extraction import (
        EXTRACTION_VERSION, extract_countries, resolve_countries_with_fallback,
    )

    cache_path = Path(input_path).with_suffix(".countrycache.json")
    input_mtime = Path(input_path).stat().st_mtime
    current_ids = [r["id"] for r in records]
    mode = f"{'llm' if use_llm else 'regex'}:v{EXTRACTION_VERSION}"

    if not no_cache and cache_path.exists() and cache_path.stat().st_mtime >= input_mtime:
        with open(cache_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("ids") == current_ids and cached.get("mode") == mode:
            for rec, countries in zip(records, cached["countries"]):
                rec["country"] = countries
            logger.info("Loaded extracted countries from cache (%s)", cache_path.name)
            return
        logger.info("Country cache doesn't match input — re-extracting…")

    needs_country = [rec for rec in records if not rec.get("country")]
    texts = [(rec.get("title") or "") + " " + (rec.get("abstract") or "") for rec in needs_country]
    if use_llm:
        resolved = resolve_countries_with_fallback(texts)
    else:
        resolved = [extract_countries(t) for t in texts]
    for rec, countries in zip(needs_country, resolved):
        rec["country"] = countries

    n_identified = sum(1 for r in records if r.get("country"))
    logger.info("Country identified for %d/%d records", n_identified, len(records))

    cache_data = {
        "mode":      mode,
        "ids":       current_ids,
        "countries": [r.get("country") or [] for r in records],
    }
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache_data, fh)
    logger.info("Extracted countries cached to %s", cache_path)


# ── Embedding cache ────────────────────────────────────────────────────────────

def _load_embeddings(records: list[dict], input_path: str, no_cache: bool):
    """
    Return a float32 numpy array of L2-normalised embeddings, one row per record.
    Loads from cache if available and current; otherwise calls the OpenAI API
    and saves a fresh cache.
    """
    import faiss
    import numpy as np

    from embeddings import EMBED_VERSION, _text_for_embedding, embed_texts

    cache_path = Path(input_path).with_suffix(".embedcache.json")
    input_mtime = Path(input_path).stat().st_mtime
    current_ids = [r["id"] for r in records]

    if not no_cache and cache_path.exists():
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= input_mtime:
            with open(cache_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            if cached.get("ids") == current_ids and cached.get("version") == EMBED_VERSION:
                raw = base64.b64decode(cached["vecs_b64"])
                vecs = np.frombuffer(raw, dtype=np.float32).reshape(cached["shape"])
                logger.info("Loaded embeddings from cache (%d vectors)", len(vecs))
                return vecs
            elif cached.get("version") != EMBED_VERSION:
                logger.info("Embedding version changed (%s → %s) — regenerating…",
                            cached.get("version", "v1"), EMBED_VERSION)
            else:
                logger.info("Cache record IDs don't match input — regenerating…")
        else:
            logger.info("Input file is newer than cache — regenerating…")

    logger.info("Generating embeddings for %d records (this may take a few minutes)…", len(records))
    texts = [_text_for_embedding(r) for r in records]
    vecs = embed_texts(texts)
    faiss.normalize_L2(vecs)

    cache_data = {
        "version":  EMBED_VERSION,
        "ids":      current_ids,
        "shape":    list(vecs.shape),
        "vecs_b64": base64.b64encode(vecs.tobytes()).decode("ascii"),
    }
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache_data, fh)
    logger.info("Embeddings cached to %s", cache_path)

    return vecs


# ── Main ───────────────────────────────────────────────────────────────────────

def _run():
    parser = argparse.ArgumentParser(description="File-based linked-publication detection")
    parser.add_argument("--input",     required=True,  help="Input RIS or Excel file")
    parser.add_argument("--sheet",     default=None,   help="Excel sheet name (default: first sheet)")
    parser.add_argument("--no-llm",    action="store_true")
    parser.add_argument(
        "--no-llm-country", action="store_true",
        help="Disable LLM fallback for country extraction (fires only when regex extraction "
             "finds nothing, and only for IE datasets; default: enabled)",
    )
    parser.add_argument("--no-cache",  action="store_true", help="Force re-generation of embeddings")
    parser.add_argument("--threshold", type=float, default=cfg.OUTPUT_THRESHOLD)
    parser.add_argument("--output",    type=str,   default=None)
    parser.add_argument(
        "--name-freq",
        choices=["none", "names-dataset", "orcid"],
        default=None,
        help="Author name frequency source (overrides NAME_FREQUENCY_SOURCE in .env)",
    )
    parser.add_argument(
        "--llm-min", type=float, default=None,
        help="Skip LLM for pairs whose stage-2 composite is below this value (already unlikely)",
    )
    parser.add_argument(
        "--llm-max", type=float, default=None,
        help="Skip LLM for pairs whose stage-2 composite is above this value (already confident)",
    )
    parser.add_argument(
        "--dataset", choices=["ie", "sr"], default="ie",
        help="Dataset type: 'ie' uses country signal in scoring, 'sr' disables it (default: ie)",
    )
    parser.add_argument(
        "--status", nargs="+", default=None, metavar="VALUE",
        help="Only include records whose status matches one of these values (Excel input only)",
    )
    args = parser.parse_args()

    use_country = (args.dataset == "ie")

    if args.name_freq is not None:
        cfg.NAME_FREQUENCY_SOURCE = args.name_freq

    output_file = args.output or (
        cfg.oneoff_output_dir()
        / f"Within batch - linked_pairs_{Path(args.input).stem} "
          f"({datetime.date.today():%Y-%m-%d}).xlsx"  # noqa: DTZ011
    )

    import faiss
    from progress import progress

    from llm_scoring import score_pairs as llm_score_pairs
    from output_writer import write_excel
    from scoring import score_pair

    records = load_records(args.input, args.sheet)

    if args.status and Path(args.input).suffix.lower() in (".xlsx", ".xls"):
        status_set = {s.strip() for s in args.status}
        before = len(records)
        records = [r for r in records if r.get("status") in status_set]
        logger.info("Status filter %s: %d → %d records", status_set, before, len(records))

    if len(records) < 2:
        raise InputError(
            f"Need at least 2 records to compare, found {len(records)} "
            f"in {Path(args.input).name}."
        )

    # Both are start-up conditions that would otherwise fail late or not at all.
    # The LLM key is checked because neither consumer fails loudly without it:
    # the country fallback catches its own errors and returns no countries, and
    # stage-3 scoring records a per-pair error and carries on — so a missing key
    # yields a complete-looking run with those signals silently absent.
    #
    # The embeddings key is deliberately NOT checked here: a fully cached re-run
    # makes no API call, and requiring the key would break working offline. The
    # guard in embeddings._get_client() covers the case where one is really
    # needed, before any spend.
    llm_country = use_country and not args.no_llm_country
    cfg.check_api_keys(need_llm=not args.no_llm or llm_country)

    from author_matching import init_frequency_db
    name_weighting = init_frequency_db()

    # Records with no country (RIS, an Excel file without a country column, or
    # a blank cell) get it extracted from title + abstract, whatever the format
    _load_countries(records, args.input, args.no_cache, llm_country)

    records_by_id = {r["id"]: r for r in records}

    # ── Build TF-IDF model ─────────────────────────────────────────────────────
    logger.info("Building TF-IDF model from input titles…")
    titles = [r["title"] for r in records if r.get("title")]
    from sklearn.feature_extraction.text import TfidfVectorizer

    from text_processing import _tokenise
    tfidf = TfidfVectorizer(
        analyzer="word", tokenizer=_tokenise, token_pattern=None,
        min_df=1, sublinear_tf=True, ngram_range=(1, 2),
    )
    tfidf.fit(titles)

    # Monkey-patch tfidf into text_processing and scoring for this run
    import scoring as sc_module
    import text_processing as tp
    tp._tfidf_model = tfidf
    sc_module._tfidf_model = tfidf

    # ── Embeddings (cached) and FAISS index ────────────────────────────────────
    vecs = _load_embeddings(records, args.input, args.no_cache)

    logger.info("Building FAISS index…")
    index = faiss.IndexFlatIP(cfg.EMBEDDING_DIM)
    index.add(vecs)
    idx_id_map = [r["id"] for r in records]

    # ── Stages 1 & 2 (cached) ──────────────────────────────────────────────────
    stage2_cache_path = Path(args.input).with_suffix(".stage2cache.json")
    input_mtime = Path(args.input).stat().st_mtime
    stage2_loaded = False

    # The cache holds computed composite scores, so it is only reusable while
    # every setting that feeds into them is unchanged — weights, thresholds, the
    # country mode, and the author name weighting actually in effect.
    from scoring import describe_config_change, scoring_config_snapshot
    country_mode = f"{args.dataset}:{'llm' if llm_country else 'regex'}"
    scoring_config = scoring_config_snapshot(name_weighting, country_mode=country_mode)

    if not args.no_cache and stage2_cache_path.exists() and \
            stage2_cache_path.stat().st_mtime >= input_mtime:
        with open(stage2_cache_path, encoding="utf-8") as fh:
            _s2 = json.load(fh)
        if _s2.get("scoring_config") != scoring_config:
            logger.info(
                "Stage 1+2 cache was built with different scoring settings — "
                "regenerating (%s)",
                describe_config_change(_s2.get("scoring_config"), scoring_config),
            )
        else:
            # ann_sim_lookup: stored as [[id_a, id_b, sim], ...] — restore as dict with tuple keys
            ann_sim_lookup = {(row[0], row[1]): row[2] for row in _s2["ann_sim_lookup"]}
            # stage2: stored as list of dicts — restore as PairScore dataclasses
            from scoring import PairScore
            stage2 = [
                PairScore(**{
                    **d,
                    "matched_authors": [tuple(t) for t in d.get("matched_authors", [])],
                })
                for d in _s2["stage2"]
            ]
            all_pairs = list(ann_sim_lookup.keys())
            logger.info("Loaded Stage 1+2 results from cache (%d pairs)", len(all_pairs))
            stage2_loaded = True

    if not stage2_loaded:
        # ── Stage 1: ANN candidate pairs ───────────────────────────────────────
        logger.info("ANN candidate retrieval…")
        ann_sim_lookup: dict[tuple, float] = {}
        top_k = min(cfg.ANN_TOP_K, len(records))

        for i, rec in enumerate(progress(records, desc="Stage 1: ANN search")):
            vec = vecs[i : i + 1]   # reuse pre-computed, already normalised
            sims, idxs = index.search(vec, top_k)
            for sim, idx in zip(sims[0], idxs[0]):
                if idx >= 0:
                    cand_id = idx_id_map[idx]
                    if cand_id != rec["id"]:
                        pair = (max(rec["id"], cand_id), min(rec["id"], cand_id))
                        ann_sim_lookup[pair] = max(ann_sim_lookup.get(pair, 0.0), float(sim))

        all_pairs = list(ann_sim_lookup.keys())
        logger.info("%d candidate pairs", len(all_pairs))

        # ── Stage 2: Feature scoring ────────────────────────────────────────────
        stage2 = []
        for pair in progress(all_pairs, desc="Stage 2: feature scoring"):
            id_a, id_b = pair
            rec_a = records_by_id.get(id_a)
            rec_b = records_by_id.get(id_b)
            if not rec_a or not rec_b:
                continue
            emb_sim = ann_sim_lookup.get(pair, 0.0)
            ps = score_pair(rec_a, rec_b, embedding_sim=emb_sim, llm_result=None, use_country=use_country)
            stage2.append(ps)

        cache_data = {
            "scoring_config": scoring_config,
            # JSON object keys must be strings; store pairs as [[id_a, id_b, sim], ...]
            "ann_sim_lookup": [[k[0], k[1], v] for k, v in ann_sim_lookup.items()],
            "stage2":         [dataclasses.asdict(ps) for ps in stage2],
        }
        with open(stage2_cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache_data, fh)
        logger.info("Stage 1+2 results cached to %s", stage2_cache_path)

    # ── Stage 3: LLM scoring ───────────────────────────────────────────────────
    if not args.no_llm:
        llm_eligible = sorted(
            [ps for ps in stage2
             if (ps.author_score >= cfg.LLM_PREFILTER_THRESHOLD
                 or ps.title_score >= cfg.LLM_PREFILTER_THRESHOLD)
             and (args.llm_min is None or ps.composite >= args.llm_min)
             and (args.llm_max is None or ps.composite <= args.llm_max)],
            key=lambda s: s.author_score + s.title_score,
            reverse=True,
        )
        logger.info("Sending %d pairs to LLM…", len(llm_eligible))
        if llm_eligible:
            llm_pairs = [(records_by_id[ps.id_a], records_by_id[ps.id_b]) for ps in llm_eligible]
            llm_results = llm_score_pairs(llm_pairs)
            llm_map = {(ps.id_a, ps.id_b): res for ps, res in zip(llm_eligible, llm_results)}
        else:
            llm_map = {}

        final_scores = []
        for ps in progress(stage2, desc="Stage 3: final scoring"):
            llm_res = llm_map.get((ps.id_a, ps.id_b))
            emb_sim = ann_sim_lookup.get((max(ps.id_a, ps.id_b), min(ps.id_a, ps.id_b)), 0.0)
            final_scores.append(score_pair(records_by_id[ps.id_a], records_by_id[ps.id_b], emb_sim, llm_res, use_country=use_country))
    else:
        final_scores = stage2

    # ── Filter, sort, write ────────────────────────────────────────────────────
    final_scores = [ps for ps in final_scores if ps.composite >= args.threshold]
    final_scores.sort(key=lambda s: s.composite, reverse=True)
    logger.info("%d pairs above threshold %.2f", len(final_scores), args.threshold)

    metadata = {
        "Input file": args.input,
        "Total records": len(records),
        "Status filter": ", ".join(args.status) if args.status else "None",
        "Candidate pairs checked": len(all_pairs),
        "LLM scoring": "Yes" if not args.no_llm else "No",
        "LLM country fallback": "Yes" if llm_country else "No",
        "Author name weighting": name_weighting,
        "Score threshold": args.threshold,
    }
    write_excel(final_scores, records_by_id, output_file, run_metadata=metadata)
    logger.info("Done — results written to %s", output_file)


def main():
    cli_entry(_run, logger)


if __name__ == "__main__":
    main()
