# Linked Publications Detection Tool

[Link to the online app](https://find-linked-pubs.streamlit.app/)

## Overview

This tool scans a set of research publications and flags pairs that are likely reporting on the same underlying study — known as "linked publications". It combines author name matching, title similarity, and AI-based semantic comparison to produce a ranked list of candidate pairs for human review.

The tool is tailored for use in maintaining a living evidence repository in international development, but many parts of the tool can be adapted for more general use.

It is designed to work with two types of studies — **impact evaluations** (IEs)  and **systematic reviews** (SRs) — with customised logic to identify linkages among each type of study.

---

## Use cases

**Use Case 1 — One-time audit of an existing dataset**
You have a large set of records in a spreadsheet or RIS file and want to find any pairs that may be linked. Run `file_check.py` on the file. Results are written to an Excel workbook for human review.

**Use Case 2 — Ongoing batch workflow for living evidence products**
You have an existing database of records and want to periodically add new batches of records, while capturing any linked relationships between the existing database and the new batch. You can build a "base database" from your existing records with `build_base.py`. Before entering a new batch into the database, you can compare it against the existing set using `compare_to_base.py` to flag any potential links. After review, you can append the newly approved records to the base with `build_base.py --append`, ready for the next round.

---

## How it works

Records are compared in three stages, balancing speed and accuracy:

```
Input records
      │
      ▼
[Stage 1] FAISS semantic search     ← vector embeddings; retrieves top-100
      │   (~1 second total)             most similar records per query
      ▼
[Stage 2] Feature scoring           ← author fuzzy match (Jaro-Winkler),
      │   (<2 seconds total)            TF-IDF title similarity, year gap,
      │                                 country signal (IE only)
      ▼
[Stage 3] LLM semantic scoring      ← Claude or GPT-4o-mini reads title +
      │   (rate-limited; see below)     abstract and gives a holistic judgment
      ▼
Composite score + ranked Excel output
```

**Stage 1** converts each record's title, authors, and abstract into a numeric "embedding" — a list of numbers that represents the text's meaning — using OpenAI's `text-embedding-3-small` model. These embeddings are stored in a FAISS index (FAISS is a library by Meta for fast search over large collections of embeddings). FAISS finds the ~100 most semantically similar records for each query in about a second across 20,000 records, avoiding the need to compare every possible pair.

**Stage 2** scores each candidate pair on several features:

- *Author overlap* — weighted by name rarity, so a shared unusual surname counts for more than a shared common one. Name frequency is looked up using the `names-dataset` library (a global name frequency database) by default; alternatively, a frequency table built from the public ORCID researcher registry (~20 million records) can be used (see `--name-freq` in the CLI reference). Before comparison, author names are normalised: accents are removed, initials are standardised, and names stored as "Surname, Firstname" are rewritten to "Firstname Surname". To handle transposed names, each pair is compared both as-is and with the name tokens sorted alphabetically — the best of the two scores is used, so "Aggarwal, Shilpa" and "Shilpa Aggarwal" will match correctly regardless of which order they appear in each record.

- *Title similarity* — TF-IDF cosine similarity (a standard measure of word-overlap between texts, adjusted for how common each word is across all records).

- *Year gap* — penalises large differences in publication year.

- *Country signal (IE only)* — whether the two records mention the same country. For Excel input, the tool looks for country information in a dedicated `country` column; for RIS input, which has no country field, country names are automatically extracted from the title and abstract using a built-in list of ~180 country names, aliases, and demonyms (e.g. "Kenyan" → Kenya, "DRC" → Congo). Note that country is used as a score modifier only for IEs (where country overlap is a meaningful signal of linked publications). Since most SRs have an international focus, country information is not used in generating SR similarity scores.

- *Registration number* If both records share a registration ID number (e.g. ISRCTN, NCT, PROSPERO), the composite score is floored at a high value regardless of other signals; conflicting trial IDs cap it at a low value. These signals are combined into a weighted composite score.

**Stage 3 (optional)** sends pairs within a user-specified range of base composite scores to a large language model, which reads the titles and abstracts and gives an independent score with a rationale. The LLM result is incorporated into the final composite.

**Country signal** 

**Composite score formula** (weights configurable in `.env`):
```
composite = clip(raw + country_modifier, 0, 1)

where:
  raw = WEIGHT_AUTHOR×author + WEIGHT_TITLE×title + WEIGHT_LLM×llm
      + WEIGHT_EMBEDDING×embedding + WEIGHT_YEAR×year
```
Default weights: Author 30%, LLM 35%, Title 20%, Embedding 10%, Year 5%.

The `country_modifier` is an additive adjustment applied after the weighted sum, then the result is clipped to [0, 1]:

| Country signal | Condition | Default modifier |
|---|---|---|
| +1 (overlap) | Both records have countries and share ≥ 1 in common | **+0.05** |
| 0 (unknown) | At least one record has no identified country | **0.00** |
| −1 (disjoint) | Both records have countries but share none | **−0.40** |

The modifier is intentionally asymmetric: a matching country provides a small +5 percentage-point nudge, while conflicting countries impose a strong −40 percentage-point penalty. This reflects the prior that linked publications nearly always share a study country, so a mismatch is strong evidence against a link — but a match alone is a weaker signal (many unrelated studies share the same country). Modifier values are configurable via `COUNTRY_OVERLAP_BONUS` and `COUNTRY_DISJOINT_PENALTY` in `.env`.

**LLM rate limits** — the default concurrency and delay settings in `.env` are conservative (concurrency=3, 1s delay between requests). At these settings, Stage 3 processes roughly 60–90 pairs per minute. Use `--llm-min`/`--llm-max` to restrict LLM calls to a specific score range and reduce cost and runtime.

**Caching** — embeddings and stage-2 scores are cached to disk alongside the input file (`.embedcache.json` and `.stage2cache.json`). Re-running the same file skips the expensive steps and completes in under a minute.

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All configuration (reads from `.env`) |
| `text_processing.py` | Tokenisation, TF-IDF model, title similarity |
| `author_matching.py` | Fuzzy name matching (Jaro-Winkler) |
| `country_extraction.py` | Gazetteer-based country extraction and comparison |
| `embeddings.py` | OpenAI embeddings + FAISS index management |
| `llm_scoring.py` | LLM calls (Anthropic/OpenAI), async batching |
| `scoring.py` | Composite scoring pipeline |
| `output_writer.py` | Excel output (shared by all scripts) |
| `build_base.py` | **CLI**: build or append to a persistent base index from Excel |
| `compare_to_base.py` | **CLI**: compare a new batch against the base index |
| `file_check.py` | **CLI**: within-file all-vs-all check (one-off runs) |
| `prompts/semantic_similarity.txt` | LLM prompt for abstract comparison |

---

## Directory structure

```
data/
  ie/
    base_sources/   ← ier-published Excel exports (source of truth for IE base)
    batches/        ← incoming IE RIS/Excel batches to compare
    results/        ← IE comparison output Excel files
  sr/
    base_sources/   ← srr-published Excel exports (source of truth for SR base)
    batches/        ← incoming SR RIS/Excel batches to compare
    results/        ← SR comparison output Excel files
  oneoff/
    oneoff_input/   ← input files for ad-hoc within-file checks
    oneoff_output/  ← results from ad-hoc checks

base/
  ie/   ← IE persistent index (FAISS, TF-IDF, records, embeddings)
  sr/   ← SR persistent index
```

Large data files (Excel exports, RIS files, cache files, output workbooks) are excluded from git. Only the directory structure (`.gitkeep` files) and code are tracked.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and scoring settings
```

Key `.env` variables:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | For embeddings (`text-embedding-3-small`) |
| `ANTHROPIC_API_KEY` | For LLM scoring (Claude) |
| `OUTPUT_THRESHOLD` | Minimum composite score to include in output (default `0.40`) |
| `LLM_PREFILTER_THRESHOLD` | Min author or title score to send a pair to LLM (default `0.15`) |
| `LLM_CONCURRENCY` | Parallel LLM requests (default `3`) |
| `LLM_REQUEST_DELAY` | Seconds between LLM requests (default `1.0`) |
| `WEIGHT_AUTHOR/TITLE/LLM/EMBEDDING/YEAR` | Composite score weights |
| `COUNTRY_OVERLAP_BONUS` / `COUNTRY_DISJOINT_PENALTY` | Country modifier values |

---

## Workflows

### Phase A — Within-dataset audit (one-time)

Check all records in the base Excel file against each other:

```bash
# IE
python file_check.py \
  --input "data/ie/base_sources/ier-published_2026-05-17.xlsx" \
  --llm-min 0.4 --llm-max 0.6 \
  --output "data/ie/results/within_base_2026-05-17.xlsx"

# SR
python file_check.py \
  --input "data/sr/base_sources/srr-published_2026-05-19.xlsx" \
  --dataset sr \
  --llm-min 0.4 --llm-max 0.6 \
  --output "data/sr/results/within_base_2026-05-19.xlsx"
```

**Expected runtime** (~21k records, IE):
- First run: ~45–70 min (20 min embedding + 5 min scoring + 20–45 min LLM depending on `--llm-min`/`--llm-max` range)
- Re-runs with cache: ~5 min

---

### Phase B — Build the persistent base index (one-time, then updated each round)

Run after Phase A. Reuses the embedding cache so no new API calls are needed.

```bash
# IE
python build_base.py --input "data/ie/base_sources/ier-published_2026-05-17.xlsx"

# SR
python build_base.py \
  --input "data/sr/base_sources/srr-published_2026-05-19.xlsx" \
  --base-dir base/sr
```

Saves to `base/ie/` or `base/sr/`: FAISS index, TF-IDF model, raw record data, and embedding matrix.

---

### Phase C — Recurring batch comparison

Repeat these steps each time a new batch arrives.

**Step C1 — Compare batch against base**

```bash
# IE batch
python compare_to_base.py \
  --input "data/ie/batches/<batch_file>" \
  --llm-min 0.4 --llm-max 0.6

# SR batch
python compare_to_base.py \
  --input "data/sr/batches/<batch_file>" \
  --dataset sr --base-dir base/sr
```

Output defaults to `compare_<stem>_<date>.xlsx` in the working directory; use `--output` to override.

**Step C2 — Review results**
Open the output Excel. Each flagged pair shows scores, matched authors, LLM rationale, and key evidence. Adjudicate which records are genuinely linked.

**Step C3 — Enter non-linked records**
Add approved (non-linked) records to the database via the normal workflow.

**Step C4 — Export new records**
Export the newly added records from the database as an `ier-published`-format Excel file.

**Step C5 — Append to base**

```bash
# IE
python build_base.py --append \
  --input "data/ie/base_sources/approved_batch_YYYYMMDD.xlsx"

# SR
python build_base.py --append \
  --input "data/sr/base_sources/approved_batch_YYYYMMDD.xlsx" \
  --base-dir base/sr
```

Only new records (by ID) are embedded; existing vectors are preserved. TF-IDF is refit on all titles.

**Step C6** — Repeat from C1 for the next batch.

---

### One-off within-file check

For ad-hoc checks that don't involve the base database:

```bash
python file_check.py --input "data/oneoff/oneoff_input/my_refs.ris"
python file_check.py --input "data/oneoff/oneoff_input/my_refs.txt"    # .txt = RIS
python file_check.py --input "data/oneoff/oneoff_input/my_refs.xlsx"
python file_check.py --input "data/oneoff/oneoff_input/my_refs.ris" --dataset sr
```

Output goes to `data/oneoff/oneoff_output/` by default.

---

## Input formats

**Excel** — must have a `title` column. Optional columns: `id`, `authors` (semicolon-delimited), `abstract`, `year` or `year_of_publication`, `country` (semicolon-delimited), `status`. If a `country` column is present, those values are used directly as the country signal for scoring. If a `status` column is present, the values are carried through to the output and can be used to restrict the analysis with `--status` (see CLI reference). Author name order (e.g. "Smith, John" vs. "John Smith") is handled automatically.

**RIS / TXT** — standard RIS format. Both `.ris` and `.txt` extensions are accepted (RIS files are frequently saved with a `.txt` extension by reference managers). RIS has no dedicated country field; the tool extracts country names automatically from the title and abstract (see Stage 2 description). Author name order is handled automatically. Fields used:

| RIS tag | Field |
|---------|-------|
| `TI` / `T1` | Title |
| `AU` / `A1` | Authors (one per line) |
| `AB` | Abstract |
| `PY` / `Y1` | Year |
| `ID` | Record ID (auto-generated if absent) |

---

## CLI reference

All three main scripts share these options:

| Flag | Description |
|------|-------------|
| `--no-llm` | Skip LLM scoring; use stage-2 composite scores only |
| `--threshold FLOAT` | Minimum composite score to include in output (default: `OUTPUT_THRESHOLD` from `.env`) |
| `--output PATH` | Output Excel file path |
| `--llm-min FLOAT` | Skip LLM for pairs with composite **below** this value |
| `--llm-max FLOAT` | Skip LLM for pairs with composite **above** this value |
| `--dataset ie\|sr` | Dataset type — `sr` disables country scoring (default: `ie`) |
| `--name-freq none\|names-dataset\|orcid` | Author name frequency source: `none` = all names weighted equally; `names-dataset` = global name frequency database (default); `orcid` = frequency table pre-built from the public ORCID researcher registry (requires running `build_name_frequencies.py` first) |
| `--sheet NAME` | Excel sheet name (default: first sheet) |

Additional flags:

| Script | Flag | Description |
|--------|------|-------------|
| `file_check.py` | `--no-cache` | Force re-generation of embeddings and stage-2 cache |
| `file_check.py`, `compare_to_base.py` | `--status VALUE [VALUE ...]` | Restrict analysis to records whose `status` column value matches one of the supplied strings. Multiple values are OR-combined (e.g. `--status Published "Published EGM"`). For `compare_to_base.py`, the filter is applied to both the query file (Excel only — RIS files have no status) and the base records. Requires the base index to have been built from an Excel file that included a `status` column; if the base pre-dates this feature, rebuild it with `build_base.py` before using this flag. |
| `build_base.py` | `--append` | Add new records to existing base instead of rebuilding |
| `build_base.py` | `--no-cache` | Force re-generation of embeddings |
| `build_base.py` | `--base-dir PATH` | Base index directory (default: `base/ie`) |
| `compare_to_base.py` | `--base-dir PATH` | Base index directory (default: `base/<dataset>`) |

---

## Output

All scripts produce an Excel workbook with two sheets:

**Summary sheet** — run metadata (input file, record counts, threshold, LLM settings), score distribution histogram.

**Pairs sheet** — one row per candidate pair, sorted by composite score descending:

| Column | Description |
|--------|-------------|
| Composite score | Overall likelihood of a link (colour-coded: ≥0.75 red, ≥0.55 orange, ≥0.40 yellow) |
| Author score | Fuzzy author name overlap |
| Title score | TF-IDF cosine similarity |
| Embedding similarity | Semantic vector similarity |
| Year score | Proximity of publication years |
| Country signal | +1 overlap / 0 unknown / −1 disjoint (IE only) |
| LLM score | LLM-assigned probability of a link |
| LLM rationale | Free-text explanation from the LLM |
| LLM key evidence | Specific evidence cited by the LLM |
| Matched authors | Paired names with Jaro-Winkler similarity, e.g. "Smith, J. ↔ Smith, John (0.94)" |
| ID A / ID B | Record identifiers |
| Source project A / B | Value from the `source_project` column, or "NA" if absent (always "NA (data sourced from RIS file)" for RIS input) |
| Status A / B | Value from the `status` column; blank if the Excel file had no `status` column; "NA (data sourced from RIS file)" for RIS input |
| Title A / B | Record titles |
| Authors A / B | Semicolon-delimited author lists |
| Year A / B | Publication years |
| Countries A / B | Countries associated with each record |

---

## Tuning

Adjust `OUTPUT_THRESHOLD` in `.env` to control precision/recall trade-off. Lower values catch more pairs but increase false positives.

Adjust the `WEIGHT_*` variables to change the relative influence of each scoring component. Weights must sum to 1.0 (excluding the country modifier).

**Recommended process:**
1. Run with `--no-llm` on a sample to identify candidates quickly
2. Have a reviewer verify ~50 true positive and ~50 true negative pairs
3. Adjust `OUTPUT_THRESHOLD` and weights based on observed precision/recall
4. Re-run with LLM scoring for final results

---

## Planned integrations

The following integrations are planned for a future phase of development:

**Real-time checking (Laravel)** — an API endpoint (`api_service.py`, built on FastAPI) will allow the web application to query the tool in real time as a user enters a new record, surfacing potential links before the record is saved.

**Database-backed workflow** — scripts (`build_index.py`, `batch_check.py`) for building the FAISS index directly from MySQL and running batch checks across all database records are already implemented and will be integrated into the main workflow once the API service is deployed.

**This work was produced as part of the Evidence Synthesis Hackathon and is licensed under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**
