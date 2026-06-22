"""
country_extraction.py
─────────────────────
Extract and normalise country names from free text (titles and abstracts).

Approach: a regex-based gazetteer built from pycountry plus a manually curated
list of common aliases, abbreviations, and demonyms used in development-sector
research.  This is faster and more predictable than NER for this specific task.

The country_signal() function returns:
  +1  if both records mention at least one common country  (weak positive)
   0  if at least one record has no identified country     (no signal)
  -1  if both records have countries but no overlap        (strong negative)
"""

from __future__ import annotations
import re
from functools import lru_cache

import pycountry

# ── Gazetteer: alias → ISO standard name ──────────────────────────────────────
# Maps lowercase aliases → canonical English country name (as used by pycountry)

_ALIASES: dict[str, str] = {
    # Common abbreviations
    "drc":          "Congo, The Democratic Republic of the",
    "rdc":          "Congo, The Democratic Republic of the",
    "droc":         "Congo, The Democratic Republic of the",
    "democratic republic of the congo": "Congo, The Democratic Republic of the",
    "democratic republic of congo": "Congo, The Democratic Republic of the",
    "republic of congo": "Congo",
    "ivory coast":  "Côte d'Ivoire",
    "cote d'ivoire":"Côte d'Ivoire",
    "cote divoire": "Côte d'Ivoire",
    "uk":           "United Kingdom",
    "great britain":"United Kingdom",
    "england":      "United Kingdom",
    "scotland":     "United Kingdom",
    "wales":        "United Kingdom",
    "usa":          "United States",
    "united states of america": "United States",
    "us ":          "United States",   # trailing space avoids matching "us" mid-word
    "uae":          "United Arab Emirates",
    "rok":          "Korea, Republic of",
    "south korea":  "Korea, Republic of",
    "north korea":  "Korea, Democratic People's Republic of",
    "dprk":         "Korea, Democratic People's Republic of",
    "tanzania":     "Tanzania, United Republic of",
    "united republic of tanzania": "Tanzania, United Republic of",
    "vietnam":      "Viet Nam",
    "laos":         "Lao People's Democratic Republic",
    "lao pdr":      "Lao People's Democratic Republic",
    "moldova":      "Moldova, Republic of",
    "russia":       "Russian Federation",
    "iran":         "Iran, Islamic Republic of",
    "syria":        "Syrian Arab Republic",
    "bolivia":      "Bolivia, Plurinational State of",
    "venezuela":    "Venezuela, Bolivarian Republic of",
    "myanmar":      "Myanmar",
    "burma":        "Myanmar",
    "macedonia":    "North Macedonia",
    "kyrgyzstan":   "Kyrgyzstan",
    "gambia":       "Gambia",
    "the gambia":   "Gambia",
    "eswatini":     "Eswatini",
    "swaziland":    "Eswatini",
    "cabo verde":   "Cabo Verde",
    "cape verde":   "Cabo Verde",
    "timor-leste":  "Timor-Leste",
    "east timor":   "Timor-Leste",
    # Common demonyms / adjectival forms used in paper titles
    "kenyan":       "Kenya",
    "ugandan":      "Uganda",
    "tanzanian":    "Tanzania, United Republic of",
    "rwandan":      "Rwanda",
    "ethiopian":    "Ethiopia",
    "nigerian":     "Nigeria",
    "ghanaian":     "Ghana",
    "malawian":     "Malawi",
    "zambian":      "Zambia",
    "zimbabwean":   "Zimbabwe",
    "bangladeshi":  "Bangladesh",
    "indian":       "India",
    "pakistani":    "Pakistan",
    "nepali":       "Nepal",
    "sri lankan":   "Sri Lanka",
    "cambodian":    "Cambodia",
    "indonesian":   "Indonesia",
    "philippine":   "Philippines",
    "philippines":  "Philippines",
    "filipino":     "Philippines",
    "mexican":      "Mexico",
    "peruvian":     "Peru",
    "colombian":    "Colombia",
    "honduran":     "Honduras",
    "guatemalan":   "Guatemala",
    "haitian":      "Haiti",
    "senegalese":   "Senegal",
    "malian":       "Mali",
    "burkinabe":    "Burkina Faso",
    "burkina faso": "Burkina Faso",
    "mozambican":   "Mozambique",
    "south african":"South Africa",
    "liberian":     "Liberia",
    "sierra leonean": "Sierra Leone",
    "sierra leone": "Sierra Leone",
}


@lru_cache(maxsize=1)
def _build_gazetteer() -> dict[str, str]:
    """
    Build a combined gazetteer: {lowercase_alias: canonical_name}.
    Cached so it is only constructed once per process.
    """
    gazetteer: dict[str, str] = {}

    # Add all pycountry entries
    for country in pycountry.countries:
        canonical = country.name
        gazetteer[canonical.lower()] = canonical
        if hasattr(country, "common_name"):
            gazetteer[country.common_name.lower()] = canonical
        if hasattr(country, "official_name"):
            gazetteer[country.official_name.lower()] = canonical

    # Add manual aliases (may overwrite pycountry entries — intentional)
    gazetteer.update(_ALIASES)

    return gazetteer


def _build_pattern(gazetteer: dict[str, str]) -> re.Pattern:
    """Compile a single regex alternation for all gazetteer keys."""
    # Sort by length descending so longer phrases match before shorter ones
    keys = sorted(gazetteer.keys(), key=len, reverse=True)
    escaped = [re.escape(k) for k in keys]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_GAZETTEER: dict[str, str] | None = None
_PATTERN: re.Pattern | None = None


def _ensure_loaded() -> tuple[dict[str, str], re.Pattern]:
    global _GAZETTEER, _PATTERN
    if _GAZETTEER is None:
        _GAZETTEER = _build_gazetteer()
        _PATTERN = _build_pattern(_GAZETTEER)
    return _GAZETTEER, _PATTERN


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_countries(text: str) -> list[str]:
    """
    Extract and normalise country names from a free-text string.
    Returns a deduplicated list of canonical country names.
    """
    if not text:
        return []
    gazetteer, pattern = _ensure_loaded()
    matches = pattern.findall(text)
    canonical = {gazetteer[m.lower()] for m in matches if m.lower() in gazetteer}
    return sorted(canonical)


def country_signal(
    countries_a: list[str],
    countries_b: list[str],
) -> tuple[int, str]:
    """
    Compare two country lists and return a (signal, description) tuple.

    Signal values:
       1  → overlap found         (weak positive indicator)
       0  → one or both empty     (no information)
      -1  → both non-empty, no overlap  (strong negative indicator)
    """
    set_a = set(countries_a)
    set_b = set(countries_b)

    if not set_a or not set_b:
        return 0, "Country information missing for one or both records"

    overlap = set_a & set_b
    if overlap:
        return 1, f"Shared countries: {', '.join(sorted(overlap))}"

    return -1, (
        f"Different countries: {', '.join(sorted(set_a))} vs {', '.join(sorted(set_b))}"
    )
