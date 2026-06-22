"""
config.py
─────────
Central configuration for the Linked Publications Detection tool.
All settings can be overridden via environment variables or a .env file.
Copy .env.example to .env and fill in your values before running any script.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "publications.faiss"
ID_MAP_PATH      = INDEX_DIR / "id_map.json"   # maps FAISS row → publication id
TFIDF_PATH       = INDEX_DIR / "tfidf.json"     # fitted TF-IDF vectoriser

# ── MySQL Database ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "user":     os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", ""),
}

# Table / column names — adjust these to match your actual Laravel schema.
# To find your table names: SHOW TABLES;
# To inspect a table:      DESCRIBE publication_authors;
DB_PUBLICATIONS_TABLE = os.getenv("DB_PUBLICATIONS_TABLE", "publications")
DB_AUTHORS_TABLE      = os.getenv("DB_AUTHORS_TABLE",      "publication_authors")
DB_AUTHORS_FK         = os.getenv("DB_AUTHORS_FK",         "publication_id")
DB_AUTHOR_NAME_COL    = os.getenv("DB_AUTHOR_NAME_COL",    "author")
DB_COUNTRY_COL        = os.getenv("DB_COUNTRY_COL",        "country")
DB_COUNTRY_DELIMITER  = os.getenv("DB_COUNTRY_DELIMITER",  ";")

# ── LLM Provider ──────────────────────────────────────────────────────────────
LLM_PROVIDER      = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" | "openai"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL",   "claude-haiku-4-5-20251001")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# ── Embeddings (OpenAI) ────────────────────────────────────────────────────────
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM          = 1536   # fixed for text-embedding-3-small

# ── Pipeline Retrieval Settings ────────────────────────────────────────────────
ANN_TOP_K               = int(os.getenv("ANN_TOP_K",   100))
LLM_TOP_K               = int(os.getenv("LLM_TOP_K",    20))
OUTPUT_THRESHOLD        = float(os.getenv("OUTPUT_THRESHOLD",        0.40))
LLM_PREFILTER_THRESHOLD = float(os.getenv("LLM_PREFILTER_THRESHOLD", 0.15))
LLM_CONCURRENCY         = int(os.getenv("LLM_CONCURRENCY",    3))
# Minimum seconds to wait after each API call (limits requests/minute to 60 / delay)
LLM_REQUEST_DELAY       = float(os.getenv("LLM_REQUEST_DELAY",  1.0))
# Retry settings for rate-limit (429) and transient server errors (500/503).
# LLM_RETRY_BASE_DELAY is the fallback wait (seconds) when no retry-after header
# is present; actual delay is taken from the API's retry-after header when available.
LLM_MAX_RETRIES         = int(os.getenv("LLM_MAX_RETRIES",      5))
LLM_RETRY_BASE_DELAY    = float(os.getenv("LLM_RETRY_BASE_DELAY", 5.0))

# ── Scoring Weights ────────────────────────────────────────────────────────────
WEIGHT_AUTHOR    = float(os.getenv("WEIGHT_AUTHOR",    0.30))
WEIGHT_TITLE     = float(os.getenv("WEIGHT_TITLE",     0.20))
WEIGHT_LLM       = float(os.getenv("WEIGHT_LLM",       0.35))
WEIGHT_EMBEDDING = float(os.getenv("WEIGHT_EMBEDDING", 0.10))
WEIGHT_YEAR      = float(os.getenv("WEIGHT_YEAR",      0.05))

COUNTRY_OVERLAP_BONUS    = float(os.getenv("COUNTRY_OVERLAP_BONUS",    0.05))
COUNTRY_DISJOINT_PENALTY = float(os.getenv("COUNTRY_DISJOINT_PENALTY", -0.40))

# ── Trial registration number overrides ───────────────────────────────────────
# A shared trial ID overrides the composite to at least this floor value.
# Conflicting trial IDs (both present, none shared) cap the composite at this ceiling.
TRIAL_ID_MATCH_FLOOR   = float(os.getenv("TRIAL_ID_MATCH_FLOOR",   0.85))
TRIAL_ID_CONFLICT_CEIL = float(os.getenv("TRIAL_ID_CONFLICT_CEIL", 0.10))

YEAR_GAP_THRESHOLD = int(os.getenv("YEAR_GAP_THRESHOLD",   8))
YEAR_GAP_PENALTY   = float(os.getenv("YEAR_GAP_PENALTY", 0.15))

AUTHOR_MATCH_THRESHOLD = float(os.getenv("AUTHOR_MATCH_THRESHOLD", 0.88))

# ── Author name frequency weighting ───────────────────────────────────────────
# Controls whether rare names are given more weight when computing author overlap.
# "none"         — unweighted (all matched names count equally)
# "names-dataset" — use the names-dataset Python package (pip install names-dataset)
# "orcid"        — use pre-built ORCID frequency table (index/name_frequencies.json)
NAME_FREQUENCY_SOURCE = os.getenv("NAME_FREQUENCY_SOURCE", "names-dataset")

# Maximum rarity multiplier applied to a matched name.
# A value of 3.0 means a very rare shared name counts 3× as much as a common one.
NAME_RARITY_MAX_WEIGHT = float(os.getenv("NAME_RARITY_MAX_WEIGHT", 3.0))
