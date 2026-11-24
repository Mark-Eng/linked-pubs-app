"""
config.py
─────────
Central configuration for the Linked Publications Detection tool.
All settings can be overridden via environment variables or a .env file.
Copy example.env to .env and fill in your values before running any script.
"""

import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv

load_dotenv()

# ── Env var helpers ───────────────────────────────────────────────────────────
T = TypeVar("T", int, float)


def _env_num(name: str, default: T, cast: Callable[[str], T]) -> T:
    """Read a numeric env var, tolerating quotes and trailing `# comments`.

    python-dotenv only strips an inline comment when the value is unquoted, so
    `KEY="0.88 # note"` arrives as the literal string. Fall back to the default
    (with a warning) rather than crashing at import time.

    Plain-string settings deliberately keep using os.getenv: secrets such as
    API keys and passwords may legitimately contain '#'.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = text.split("#", 1)[0].strip()
    if not text:
        return default
    try:
        return cast(text)
    except ValueError:
        warnings.warn(
            f"{name}={raw!r} is not a valid {cast.__name__}; using {default}.",
            stacklevel=2,
        )
        return default


def _env_int(name: str, default: int) -> int:
    return _env_num(name, default, int)


def _env_float(name: str, default: float) -> float:
    return _env_num(name, default, float)


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var, tolerating quotes and trailing `# comments`."""
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = text.split("#", 1)[0].strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    warnings.warn(
        f"{name}={raw!r} is not a valid boolean; using {default}.",
        stacklevel=2,
    )
    return default


# ── Paths ──────────────────────────────────────────────────────────────────────
# Two different things live in this repo and they must not be conflated.
#
# PACKAGE_DIR is where the *code* lives. Files that ship with the code — the
# LLM prompts and the country controlled vocabulary — are located relative to
# their own module (Path(__file__).parent), not from here, so they follow the
# code wherever it is installed. Nothing below governs them.
PACKAGE_DIR = Path(__file__).parent

# PROJECT_ROOT is where the *data* lives: the base indexes and the data/
# workflow tree. It defaults to the code directory, which is right for a git
# checkout, and is overridable for two reasons: a container usually mounts data
# on a volume separate from its image, and these paths used to be resolved
# against the current working directory, so any process started outside the
# repo root silently read and wrote the wrong places.
PROJECT_ROOT    = Path(os.getenv("LINKED_PUBS_ROOT",      PACKAGE_DIR))
BASE_INDEX_ROOT = Path(os.getenv("LINKED_PUBS_BASE_ROOT", PROJECT_ROOT / "base"))
DATA_ROOT       = Path(os.getenv("LINKED_PUBS_DATA_ROOT", PROJECT_ROOT / "data"))

# Staging area for the ORCID public data dump and the name-frequency table
# built from it (build_name_frequencies.py). Nothing else lives here; the
# base index artifacts are located by base_store.py, not by a fixed path.
ORCID_DIR = Path(os.getenv("ORCID_DIR", PROJECT_ROOT / "index"))
ORCID_FREQUENCIES_PATH = ORCID_DIR / "name_frequencies.json"


def base_index_dir(dataset: str) -> Path:
    """Default base index directory for a dataset ("ie" or "sr")."""
    return BASE_INDEX_ROOT / dataset


def results_dir(dataset: str) -> Path:
    """Where a dataset's comparison output workbooks go."""
    return DATA_ROOT / dataset / "results"


def oneoff_output_dir() -> Path:
    """Where ad-hoc within-file check workbooks go."""
    return DATA_ROOT / "oneoff" / "oneoff_output"

# ── MySQL Database ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     _env_int("DB_PORT", 3306),
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

# Column distinguishing impact evaluations from systematic reviews, and the
# value it holds for each. IE and SR records share one table but must never
# share a base index, so every index build filters on this.
DB_DATASET_COL    = os.getenv("DB_DATASET_COL", "")
DB_DATASET_VALUES = {
    "ie": os.getenv("DB_DATASET_VALUE_IE", "ie"),
    "sr": os.getenv("DB_DATASET_VALUE_SR", "sr"),
}

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
ANN_TOP_K               = _env_int("ANN_TOP_K",   100)
LLM_TOP_K               = _env_int("LLM_TOP_K",    20)
OUTPUT_THRESHOLD        = _env_float("OUTPUT_THRESHOLD",        0.50)
LLM_PREFILTER_THRESHOLD = _env_float("LLM_PREFILTER_THRESHOLD", 0.15)
LLM_CONCURRENCY         = _env_int("LLM_CONCURRENCY",    3)
# Minimum seconds to wait after each API call (limits requests/minute to 60 / delay)
LLM_REQUEST_DELAY       = _env_float("LLM_REQUEST_DELAY",  1.0)
# Retry settings for rate-limit (429) and transient server errors (500/503).
# LLM_RETRY_BASE_DELAY is the fallback wait (seconds) when no retry-after header
# is present; actual delay is taken from the API's retry-after header when available.
# Clamped to >= 1: the retry loops treat this as an attempt count, so 0 would
# skip the API call entirely and leave the caller with no response to unpack.
LLM_MAX_RETRIES         = max(1, _env_int("LLM_MAX_RETRIES", 5))
LLM_RETRY_BASE_DELAY    = _env_float("LLM_RETRY_BASE_DELAY", 5.0)

# ── Scoring Weights ────────────────────────────────────────────────────────────
WEIGHT_AUTHOR    = _env_float("WEIGHT_AUTHOR",    0.30)
WEIGHT_TITLE     = _env_float("WEIGHT_TITLE",     0.15)
WEIGHT_LLM       = _env_float("WEIGHT_LLM",       0.4)
WEIGHT_EMBEDDING = _env_float("WEIGHT_EMBEDDING", 0.10)
WEIGHT_YEAR      = _env_float("WEIGHT_YEAR",      0.05)

COUNTRY_OVERLAP_BONUS    = _env_float("COUNTRY_OVERLAP_BONUS",    0.05)
COUNTRY_DISJOINT_PENALTY = _env_float("COUNTRY_DISJOINT_PENALTY", -0.37)

# Country fill-in occasionally assigns the wrong country to genuinely linked
# pairs, which the LLM (reading only titles/abstracts) still scores near 1.
# When the LLM score reaches the threshold, only FACTOR of the disjoint penalty
# applies. FACTOR = 1.0 disables the rule.
COUNTRY_PENALTY_LLM_RELIEF_THRESHOLD = _env_float("COUNTRY_PENALTY_LLM_RELIEF_THRESHOLD", 0.80)
COUNTRY_PENALTY_LLM_RELIEF_FACTOR    = _env_float("COUNTRY_PENALTY_LLM_RELIEF_FACTOR",    1 / 3)

# ── Trial registration number overrides ───────────────────────────────────────
# A shared trial ID overrides the composite to at least this floor value.
# Conflicting trial IDs (both present, none shared) cap the composite at this ceiling.
TRIAL_ID_MATCH_FLOOR   = _env_float("TRIAL_ID_MATCH_FLOOR",   0.85)
TRIAL_ID_CONFLICT_CEIL = _env_float("TRIAL_ID_CONFLICT_CEIL", 0.10)

YEAR_GAP_THRESHOLD = _env_int("YEAR_GAP_THRESHOLD",   8)
YEAR_GAP_PENALTY   = _env_float("YEAR_GAP_PENALTY", 0.15)

AUTHOR_MATCH_THRESHOLD = _env_float("AUTHOR_MATCH_THRESHOLD", 0.88)

# Match names where one side abbreviates a part to an initial regardless of
# token order, e.g. "Faiela C" ↔ "Candido, Faiela" (see
# author_matching._initial_aware_similarity). Set false to restore the
# pre-2026-10 behaviour.
AUTHOR_INITIAL_MATCHING = _env_bool("AUTHOR_INITIAL_MATCHING", True)

# ── Author name frequency weighting ───────────────────────────────────────────
# Controls whether rare names are given more weight when computing author overlap.
# "none"         — unweighted (all matched names count equally)
# "names-dataset" — use the names-dataset Python package (pip install names-dataset)
# "orcid"        — use pre-built ORCID frequency table (index/name_frequencies.json)
NAME_FREQUENCY_SOURCE = os.getenv("NAME_FREQUENCY_SOURCE", "names-dataset")

# What to do when the requested source cannot be loaded.
#
# False (default): warn and fall back to unweighted matching, so an interactive
# run still produces results. Convenient locally, but it silently changes the
# author component — 30% of the composite — and only a log line records it.
#
# True: raise ConfigurationError instead. Set this in any automated or served
# deployment, where a run that quietly scored differently is worse than a run
# that failed. names-dataset expands to ~1.7 GB resident, so the realistic
# trigger is a memory-capped container, which would otherwise degrade unnoticed.
NAME_FREQUENCY_STRICT = _env_bool("NAME_FREQUENCY_STRICT", False)

# Maximum rarity multiplier applied to a matched name.
# A value of 3.0 means a very rare shared name counts 3× as much as a common one.
NAME_RARITY_MAX_WEIGHT = _env_float("NAME_RARITY_MAX_WEIGHT", 3.0)


# ── Credential checks ─────────────────────────────────────────────────────────

def check_api_keys(*, need_embeddings: bool = False, need_llm: bool = False) -> None:
    """
    Verify the API keys a run will actually use, before it starts.

    Worth checking up front rather than at the first call, because a missing
    key does not always fail loudly: the LLM country-extraction fallback
    catches its own errors and returns no countries, so an absent
    ANTHROPIC_API_KEY would quietly produce a run with the country signal
    disabled for every record — the same class of silent degradation as a
    failed name-frequency load.

    Raises ConfigurationError listing everything missing, so one run surfaces
    all of it rather than one key at a time.
    """
    from errors import ConfigurationError

    provider = LLM_PROVIDER.lower()
    if need_llm and provider not in ("anthropic", "openai"):
        raise ConfigurationError(
            f"LLM_PROVIDER is {LLM_PROVIDER!r} — expected 'anthropic' or 'openai'."
        )

    missing = []
    if need_embeddings and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY (embeddings)")
    if need_llm:
        if provider == "anthropic" and not ANTHROPIC_API_KEY:
            missing.append("ANTHROPIC_API_KEY (LLM scoring, LLM_PROVIDER=anthropic)")
        elif provider == "openai" and not OPENAI_API_KEY and not need_embeddings:
            missing.append("OPENAI_API_KEY (LLM scoring, LLM_PROVIDER=openai)")

    if missing:
        raise ConfigurationError(
            "Missing required API key(s): " + ", ".join(missing)
            + ". Add them to .env or the environment."
        )
