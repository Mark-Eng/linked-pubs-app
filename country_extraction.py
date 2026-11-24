"""
country_extraction.py
─────────────────────
Extract and normalise country names from free text (titles and abstracts).

Approach: a regex-based gazetteer built from pycountry plus a manually curated
list of common aliases, abbreviations, and demonyms used in development-sector
research. This is faster and more predictable than NER for this specific task.

Every alias in the gazetteer is normalised to the exact strings used in the
base dataset's controlled vocabulary (data/country_names_controlled.txt) —
not to pycountry's own canonical names, which sometimes differ (e.g. the base
data uses "Gambia, The" and "Tanzania" where pycountry uses "Gambia" and
"Tanzania, United Republic of"). Extracted country names are only useful for
comparison if they land on the same strings the base data already uses.

The country_signal() function returns:
  +1  if both records mention at least one common country  (weak positive)
   0  if at least one record has no identified country     (no signal)
  -1  if both records have countries but no overlap        (strong negative)
"""

from __future__ import annotations
import asyncio
import logging
import re
from functools import lru_cache
from pathlib import Path

import pycountry

import config as cfg
from llm_io import extract_first_json_object, retry_after_seconds, strip_control_chars
from progress import progress

logger = logging.getLogger(__name__)

_VOCAB_PATH = Path(__file__).parent / "data" / "country_names_controlled.txt"
_LLM_PROMPT_PATH = Path(__file__).parent / "prompts" / "country_extraction.txt"

# Controlled-vocab entries that are not actual countries and can never be
# produced by text extraction. Excluded from the gazetteer's target set, and
# stripped before country_signal() compares two records — otherwise two base
# records both coded "Not reported" would score as a spurious country match.
NON_COUNTRY_VALUES: frozenset[str] = frozenset({
    "Multi-country",
    "Not applicable",
    "Not applicable (no studies)",
    "Not reported",
})

# Controlled-vocab entries whose ISO alpha-2 code cannot be found via a direct
# pycountry.countries.get(name=...) lookup, because the vocab uses different
# styling ("X, Rep.", "The X", "St. X", "X, FYR", territory suffixes, etc.).
# Verified against pycountry 26.2.16 — every value below is a real, unique
# alpha-2 code with no collisions across the controlled vocabulary.
_VOCAB_TO_ALPHA2: dict[str, str] = {
    "Bahamas, The":              "BS",
    "Bolivia":                   "BO",
    "British Virgin Islands":    "VG",
    "Congo, Dem. Rep.":          "CD",
    "Congo, Rep.":               "CG",
    "Czech Republic":            "CZ",
    "Egypt, Arab Rep.":          "EG",
    "Gambia, The":               "GM",
    "Hong Kong SAR, China":      "HK",
    "Iran, Islamic Rep.":        "IR",
    "Korea, Dem. People's Rep.": "KP",
    "Korea, Rep.":               "KR",
    "Kyrgyz Republic":           "KG",
    "Lao PDR":                   "LA",
    "Macao SAR, China":          "MO",
    "Macedonia, FYR":            "MK",
    "Micronesia, Fed. Sts.":     "FM",
    "Moldova":                   "MD",
    "São Tomé and Principe":     "ST",
    "Slovak Republic":           "SK",
    "St. Kitts and Nevis":       "KN",
    "St. Lucia":                 "LC",
    "St. Martin (French part)":  "MF",
    "St. Vincent and the Grenadines": "VC",
    "Taiwan, China":             "TW",
    "Tanzania":                  "TZ",
    "Turkey":                    "TR",
    "Venezuela, RB":             "VE",
    "Vietnam":                   "VN",
    "Virgin Islands (U.S.)":     "VI",
    "Yemen, Rep.":               "YE",
}

# Controlled-vocab entries with no ISO 3166-1 code at all (disputed
# territory, aggregate region) — given hand-picked text aliases instead of
# going through pycountry.
_VOCAB_MANUAL_ALIASES: dict[str, set[str]] = {
    "Kosovo":              {"kosovo"},
    "West Bank and Gaza":  {"west bank and gaza", "west bank", "gaza", "gaza strip",
                             "palestine", "palestinian territories"},
    "Channel Islands":     {"channel islands", "jersey", "guernsey"},
}

# ── Manual aliases: lowercase alias → pycountry canonical name ────────────────
# These get redirected to the controlled-vocab string the same way pycountry's
# own names do (see _build_gazetteer). Values here must be real pycountry
# country.name values.

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
    "turkiye":      "Türkiye",
    "türkiye":      "Türkiye",
    "czechia":      "Czechia",
    "taiwan":       "Taiwan, Province of China",
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
def _load_controlled_vocab() -> list[str]:
    """Load the base dataset's controlled vocabulary of country strings."""
    lines = _VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _resolve_alpha2(entry: str) -> str | None:
    """Resolve a controlled-vocab entry to its ISO alpha-2 code."""
    if entry in _VOCAB_TO_ALPHA2:
        return _VOCAB_TO_ALPHA2[entry]
    country = pycountry.countries.get(name=entry)
    return country.alpha_2 if country else None


def _build_alpha2_to_vocab() -> dict[str, str]:
    """
    Map ISO alpha-2 code → the exact controlled-vocab string for that
    country. Raises if any non-sentinel, non-manual-alias vocab entry can't
    be resolved, so a future edit to the vocab file can't silently drop
    country-matching coverage.
    """
    vocab = [v for v in _load_controlled_vocab()
             if v not in NON_COUNTRY_VALUES and v not in _VOCAB_MANUAL_ALIASES]

    alpha2_to_vocab: dict[str, str] = {}
    unresolved: list[str] = []
    for entry in vocab:
        code = _resolve_alpha2(entry)
        if code is None:
            unresolved.append(entry)
        else:
            alpha2_to_vocab[code] = entry

    if unresolved:
        raise RuntimeError(
            "country_extraction: could not resolve an ISO alpha-2 code for "
            f"controlled-vocab entries: {unresolved}. Add them to "
            "_VOCAB_TO_ALPHA2 (if they have an ISO code) or "
            "_VOCAB_MANUAL_ALIASES (if they don't)."
        )
    return alpha2_to_vocab


@lru_cache(maxsize=1)
def _build_gazetteer() -> dict[str, str]:
    """
    Build a combined gazetteer: {lowercase_alias: controlled_vocab_string}.
    Cached so it is only constructed once per process.
    """
    alpha2_to_vocab = _build_alpha2_to_vocab()
    gazetteer: dict[str, str] = {}

    # Every controlled-vocab string matches itself.
    for entry in _load_controlled_vocab():
        if entry not in NON_COUNTRY_VALUES:
            gazetteer[entry.lower()] = entry

    # Hand-picked aliases for entries with no ISO code.
    for target, aliases in _VOCAB_MANUAL_ALIASES.items():
        for alias in aliases:
            gazetteer[alias] = target

    # All pycountry names/aliases, redirected to the controlled-vocab string
    # for that country. Countries not covered by the controlled vocabulary
    # are skipped — there's nothing in the base data for them to match.
    for country in pycountry.countries:
        target = alpha2_to_vocab.get(country.alpha_2)
        if target is None:
            continue
        for attr in ("name", "common_name", "official_name"):
            value = getattr(country, attr, None)
            if value:
                gazetteer[value.lower()] = target

    # Manually curated abbreviations/demonyms, redirected the same way.
    for alias, pycountry_name in _ALIASES.items():
        country = pycountry.countries.get(name=pycountry_name)
        target = alpha2_to_vocab.get(country.alpha_2) if country else None
        gazetteer[alias] = target or pycountry_name

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


# ── Funding / acknowledgement filter ──────────────────────────────────────────
# Funding statements name countries that are not study locations ("Funded by
# the Government of Japan"), and the regex pass cannot tell the difference.
# That text is removed before the regex runs. The LLM fallback still receives
# the full text: it can judge context, and a study country mentioned only in a
# funding sentence ("a programme funded by DFID in Malawi") would otherwise be
# lost to it as well.
#
# Bump EXTRACTION_VERSION whenever extraction or comparison output changes, so
# file_check's country and stage-2 caches regenerate instead of serving
# countries extracted under the old rules.
EXTRACTION_VERSION = 2

_ABSTRACT_SECTION_HEADERS = (
    r"(?:background|objectives?|aims?|methods?|design|setting|participants|"
    r"interventions?|findings|results|interpretation|conclusions?|"
    r"trial registration|registration)"
)
# A labelled section ("Funding: ...") runs to the next abstract header or the end.
_NON_STUDY_SECTION = re.compile(
    r"\b(?:funding(?: sources?)?|sources? of funding|financial support|"
    r"acknowledge?ments?|conflicts? of interest|competing interests?|"
    r"declarations? of interests?)\s*:"
    r".*?(?=\b" + _ABSTRACT_SECTION_HEADERS + r"\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
# Unlabelled funding sentences. Deliberately narrow: bare "grant" or "supported"
# would also drop study sentences ("unconditional cash grants in Kenya",
# "households supported by the programme").
_FUNDING_SENTENCE = re.compile(
    r"\b(?:funded by|financed by|sponsored by|financial(?:ly)? support|"
    r"grant (?:no\.?|number|from)|"
    r"(?:work|study|research|project|trial|evaluation) (?:was|is) "
    r"(?:supported|funded) by)",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _strip_non_study_text(text: str) -> str:
    """Remove funding/acknowledgement sections and funding sentences."""
    text = _NON_STUDY_SECTION.sub(" ", text)
    return " ".join(
        s for s in _SENTENCE_BREAK.split(text) if not _FUNDING_SENTENCE.search(s)
    )


# ── Public API ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8192)
def _extract_countries_cached(text: str) -> tuple[str, ...]:
    """
    Memoised extraction. The gazetteer is immutable once built, so the result
    for a given string never changes.

    Caching matters because of the fallback in scoring._resolve_countries():
    a record with no country field is re-extracted on every pair it appears
    in, which for one query record against 100 candidates means 100 identical
    regex passes over the same title and abstract.

    Returns a tuple so the cached value cannot be mutated by a caller.
    """
    gazetteer, pattern = _ensure_loaded()
    matches = pattern.findall(_strip_non_study_text(text))
    canonical = {gazetteer[m.lower()] for m in matches if m.lower() in gazetteer}
    return tuple(sorted(canonical))


def extract_countries(text: str) -> list[str]:
    """
    Extract and normalise country names from a free-text string.
    Returns a deduplicated list of country names, using the exact strings
    from the base dataset's controlled vocabulary. Countries named only in
    funding or acknowledgement text are ignored.
    """
    if not text:
        return []
    return list(_extract_countries_cached(text))


def normalise_countries(countries: list[str]) -> list[str]:
    """
    Map stored country strings onto the controlled vocabulary.

    Extraction already produces vocabulary strings, but country lists also
    arrive from input fields, older caches, and base records, where a variant
    spelling ("Iran, Islamic Republic of" for "Iran, Islamic Rep.") would make
    the same country look like a mismatch. Unrecognised strings are kept as-is
    rather than dropped, so they still take part in the comparison.
    """
    gazetteer, _ = _ensure_loaded()
    vocab = set(_load_controlled_vocab())
    normalised = []
    for country in countries:
        country = country.strip()
        normalised.append(country if country in vocab else gazetteer.get(country.lower(), country))
    return normalised


# ── LLM fallback extraction ──────────────────────────────────────────────────
# Used only for text where the regex gazetteer above finds nothing — e.g. a
# country referred to by a spelling/demonym not in the gazetteer. The prompt
# forces the model to pick only from the controlled vocabulary, so its output
# needs no further normalisation to match the base dataset (only validation).

_sanitise = strip_control_chars


@lru_cache(maxsize=1)
def _load_llm_prompt_template() -> str:
    return _LLM_PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _llm_vocab_listing() -> str:
    vocab = [v for v in _load_controlled_vocab() if v not in NON_COUNTRY_VALUES]
    return "\n".join(f"- {v}" for v in vocab)


def _build_llm_prompt(text: str) -> str:
    return (
        _load_llm_prompt_template()
        .replace("{country_list}", _llm_vocab_listing())
        .replace("{text}", _sanitise(text)[:3000])   # hard cap for token safety
    )


def _validate_llm_countries(raw: list) -> list[str]:
    """
    Keep only values that resolve to the controlled vocabulary. The prompt
    already instructs the model to use exact vocab strings; this is a safety
    net against minor case/spacing drift or an occasional hallucinated value
    (falls back to the full gazetteer, so a stray demonym/alias still lands
    on the right controlled-vocab string instead of being dropped).
    """
    gazetteer, _ = _ensure_loaded()
    vocab_lower = {v.lower(): v for v in _load_controlled_vocab() if v not in NON_COUNTRY_VALUES}
    validated = []
    for item in raw:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        target = vocab_lower.get(key) or gazetteer.get(key)
        if target:
            validated.append(target)
        else:
            logger.warning("LLM country extraction returned a value outside the controlled vocabulary, dropped: %r", item)
    return sorted(set(validated))


def _parse_llm_response(text: str) -> list[str]:
    """
    Parse the {"countries": [...]} object out of an LLM response. The model
    is instructed to return nothing else, but occasionally wraps the JSON in
    markdown fences or adds stray reasoning text before/after it. The raw
    response is never logged above DEBUG: it may contain the model's
    reasoning and shouldn't spam the terminal during normal runs.
    """
    data = extract_first_json_object(text)
    if not isinstance(data, dict):
        logger.warning("LLM country response contained no JSON object — treating as no countries identified")
        logger.debug("Raw LLM country response: %s", text)
        return []
    raw = data.get("countries", [])
    if not isinstance(raw, list):
        return []
    return _validate_llm_countries(raw)




async def _extract_llm_anthropic(prompt: str, semaphore: asyncio.Semaphore) -> list[str]:
    import anthropic as _anthropic
    # max_retries=0: disable SDK's own retry so our backoff is the only mechanism
    client = _anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY, max_retries=0)
    delay = cfg.LLM_RETRY_BASE_DELAY

    for attempt in range(1, cfg.LLM_MAX_RETRIES + 1):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=cfg.ANTHROPIC_MODEL,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                )
                if cfg.LLM_REQUEST_DELAY > 0:
                    await asyncio.sleep(cfg.LLM_REQUEST_DELAY)
                return _parse_llm_response(response.content[0].text)
            except _anthropic.RateLimitError as exc:
                if attempt >= cfg.LLM_MAX_RETRIES:
                    logger.error("Anthropic country-extraction error after %d attempts: %s", attempt, exc)
                    return []
                wait = retry_after_seconds(exc) or delay
                logger.warning("Rate limit (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                delay *= 2
            except _anthropic.InternalServerError as exc:
                if attempt >= cfg.LLM_MAX_RETRIES:
                    logger.error("Anthropic country-extraction error after %d attempts: %s", attempt, exc)
                    return []
                logger.warning("Server error (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, delay)
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error("Anthropic country-extraction error: %s", exc)
                return []
    return []


async def _extract_llm_openai(prompt: str, semaphore: asyncio.Semaphore) -> list[str]:
    from openai import AsyncOpenAI, RateLimitError as _OAIRateLimit, InternalServerError as _OAIServerError
    # max_retries=0: disable SDK's own retry so our backoff is the only mechanism
    client = AsyncOpenAI(api_key=cfg.OPENAI_API_KEY, max_retries=0)
    delay = cfg.LLM_RETRY_BASE_DELAY

    for attempt in range(1, cfg.LLM_MAX_RETRIES + 1):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=cfg.OPENAI_CHAT_MODEL,
                    max_tokens=300,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You are a research librarian assistant. Return only valid JSON."},
                        {"role": "user",   "content": prompt},
                    ],
                )
                if cfg.LLM_REQUEST_DELAY > 0:
                    await asyncio.sleep(cfg.LLM_REQUEST_DELAY)
                return _parse_llm_response(response.choices[0].message.content or "")
            except _OAIRateLimit as exc:
                if attempt >= cfg.LLM_MAX_RETRIES:
                    logger.error("OpenAI country-extraction error after %d attempts: %s", attempt, exc)
                    return []
                wait = retry_after_seconds(exc) or delay
                logger.warning("Rate limit (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                delay *= 2
            except _OAIServerError as exc:
                if attempt >= cfg.LLM_MAX_RETRIES:
                    logger.error("OpenAI country-extraction error after %d attempts: %s", attempt, exc)
                    return []
                logger.warning("Server error (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, delay)
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error("OpenAI country-extraction error: %s", exc)
                return []
    return []


async def extract_countries_llm_batch_async(
    texts: list[str],
    concurrency: int | None = None,
    show_progress: bool = True,
) -> list[list[str]]:
    """
    Resolve countries for each text via the LLM, in the same order as input.
    Intended as a fallback for texts where extract_countries() found nothing
    — call sites should not run this over every record unconditionally.
    """
    if concurrency is None:
        concurrency = cfg.LLM_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)
    provider = cfg.LLM_PROVIDER.lower()

    bar = None
    if show_progress:
        bar = progress(total=len(texts), desc="Country ID: LLM check")

    async def _one(text: str) -> list[str]:
        if not text:
            if bar is not None:
                bar.update(1)
            return []
        prompt = _build_llm_prompt(text)
        if provider == "anthropic":
            result = await _extract_llm_anthropic(prompt, semaphore)
        elif provider == "openai":
            result = await _extract_llm_openai(prompt, semaphore)
        else:
            logger.error("Unknown LLM_PROVIDER: %r — skipping LLM country extraction", provider)
            result = []
        if bar is not None:
            bar.update(1)
        return result

    try:
        return list(await asyncio.gather(*[_one(t) for t in texts]))
    finally:
        if bar is not None:
            bar.close()


def extract_countries_llm_batch(
    texts: list[str],
    concurrency: int | None = None,
    show_progress: bool = True,
) -> list[list[str]]:
    """Synchronous wrapper around extract_countries_llm_batch_async."""
    return asyncio.run(extract_countries_llm_batch_async(texts, concurrency, show_progress))


def resolve_countries_with_fallback(texts: list[str]) -> list[list[str]]:
    """
    Extract countries for each text with the fast regex gazetteer, falling
    back to an LLM call — constrained to the controlled vocabulary — only
    for texts where the regex pass found nothing.
    """
    results = [extract_countries(t) for t in texts]
    fallback_idx = [i for i, r in enumerate(results) if not r and texts[i]]
    if not fallback_idx:
        return results

    model = cfg.ANTHROPIC_MODEL if cfg.LLM_PROVIDER.lower() == "anthropic" else cfg.OPENAI_CHAT_MODEL
    logger.info(
        "Sending %d/%d records to LLM for country identification (%s, %s)…",
        len(fallback_idx), len(texts), cfg.LLM_PROVIDER, model,
    )
    llm_results = extract_countries_llm_batch([texts[i] for i in fallback_idx])
    for i, res in zip(fallback_idx, llm_results):
        results[i] = res

    identified = sum(1 for i in fallback_idx if results[i])
    logger.info(
        "LLM country identification: %d/%d additional countries identified",
        identified, len(fallback_idx),
    )
    return results


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

    Both lists are normalised to the controlled vocabulary first, so spelling
    variants of one country compare as a match.
    """
    set_a = set(normalise_countries(countries_a)) - NON_COUNTRY_VALUES
    set_b = set(normalise_countries(countries_b)) - NON_COUNTRY_VALUES

    if not set_a or not set_b:
        return 0, "Country information missing for one or both records"

    overlap = set_a & set_b
    if overlap:
        return 1, f"Shared countries: {', '.join(sorted(overlap))}"

    return -1, (
        f"Different countries: {', '.join(sorted(set_a))} vs {', '.join(sorted(set_b))}"
    )
