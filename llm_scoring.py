"""
llm_scoring.py
──────────────
LLM-based semantic similarity scoring for candidate publication pairs.

Supports both Anthropic (Claude) and OpenAI (GPT) backends, selected via
cfg.LLM_PROVIDER. Calls are made concurrently using asyncio to keep latency
low when scoring many pairs at once.

Rate limiting
─────────────
Two config settings work together to stay within API rate limits:

  LLM_CONCURRENCY        — maximum number of API calls in-flight at once
  LLM_REQUEST_DELAY      — minimum seconds between releasing the semaphore
                           and making the next call (effectively sets a
                           requests-per-minute ceiling of 60 / delay)

Rate-limit (429) and transient server errors (500/503) are retried
automatically with exponential backoff (up to LLM_MAX_RETRIES attempts).
The wait time is taken from the API's retry-after header when present;
otherwise it starts at LLM_RETRY_BASE_DELAY seconds and doubles each attempt.

Each call returns a ScoringResult with:
  score        — float [0, 1] probability of being linked
  rationale    — plain-text explanation for human reviewers
  key_evidence — list of specific observations
  confidence   — "high" | "medium" | "low"
  error        — error message if the call failed after all retries
"""

from __future__ import annotations
import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import config as cfg
from llm_io import extract_first_json_object, retry_after_seconds, strip_control_chars

logger = logging.getLogger(__name__)

# ── Load prompt template ───────────────────────────────────────────────────────
_PROMPT_PATH = Path(__file__).parent / "prompts" / "semantic_similarity.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


_sanitise = strip_control_chars


def _build_prompt(rec_a: dict, rec_b: dict) -> str:
    """Fill the prompt template with data from two records."""
    abstract_a = _sanitise(rec_a.get("abstract") or "[Abstract not available]")
    abstract_b = _sanitise(rec_b.get("abstract") or "[Abstract not available]")

    def _fmt_authors(rec: dict) -> str:
        authors = rec.get("authors") or []
        return "; ".join(_sanitise(a) for a in authors) if authors else "[Not available]"

    return (
        _PROMPT_TEMPLATE
        .replace("{title_a}",    _sanitise(rec_a.get("title") or ""))
        .replace("{authors_a}",  _fmt_authors(rec_a))
        .replace("{abstract_a}", abstract_a[:2000])   # hard cap for token safety
        .replace("{title_b}",    _sanitise(rec_b.get("title") or ""))
        .replace("{authors_b}",  _fmt_authors(rec_b))
        .replace("{abstract_b}", abstract_b[:2000])
    )


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ScoringResult:
    score: float | None          = None
    rationale: str               = ""
    key_evidence: list[str]      = field(default_factory=list)
    confidence: str              = "low"
    error: str | None            = None
    raw_response: str            = ""


def _coerce_score(raw: object) -> float | None:
    """
    A usable score in [0, 1], or None if the model didn't give one.

    Clamped rather than trusted. The score is multiplied by WEIGHT_LLM and
    added to the composite, which is only clipped at the very end — so an
    out-of-range value (a model answering 0-100, say) would swamp every other
    signal and push the pair to the ceiling.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if score != score:                        # NaN
        return None
    if not 0.0 <= score <= 1.0:
        logger.warning("LLM returned score %r outside [0, 1] — clamping", raw)
        score = min(1.0, max(0.0, score))
    return score


def _parse_llm_response(text: str) -> ScoringResult:
    """
    Parse the scoring JSON out of an LLM response.

    Every field is validated rather than trusted. The previous version called
    data.get(...) on whatever json.loads returned and coerced the score with a
    bare float(), so a response that was a JSON array raised AttributeError and
    a non-numeric score raised TypeError — neither of which it caught, so both
    escaped asyncio.gather and failed the whole batch instead of degrading the
    one pair.
    """
    data = extract_first_json_object(text)
    if data is None:
        logger.warning("LLM response contained no JSON object — %s", text[:200])
        return ScoringResult(error="Parse error: no JSON object in response",
                             raw_response=text)

    score = _coerce_score(data.get("score"))
    if score is None:
        logger.warning("LLM response had no usable score: %r", data.get("score"))
        return ScoringResult(error=f"Parse error: unusable score {data.get('score')!r}",
                             raw_response=text)

    evidence = data.get("key_evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]

    confidence = data.get("confidence", "low")

    return ScoringResult(
        score=score,
        rationale=str(data.get("rationale") or ""),
        key_evidence=[str(item) for item in evidence],
        confidence=str(confidence) if confidence else "low",
        raw_response=text,
    )




# ── Anthropic backend ──────────────────────────────────────────────────────────

async def _score_anthropic(prompt: str, semaphore: asyncio.Semaphore) -> ScoringResult:
    import anthropic as _anthropic
    # max_retries=0: disable SDK's own retry so our backoff is the only mechanism
    client = _anthropic.AsyncAnthropic(api_key=cfg.ANTHROPIC_API_KEY, max_retries=0)
    delay = cfg.LLM_RETRY_BASE_DELAY

    for attempt in range(1, cfg.LLM_MAX_RETRIES + 1):
        async with semaphore:
            try:
                response = await client.messages.create(
                    model=cfg.ANTHROPIC_MODEL,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                if cfg.LLM_REQUEST_DELAY > 0:
                    await asyncio.sleep(cfg.LLM_REQUEST_DELAY)
                return _parse_llm_response(response.content[0].text)
            except _anthropic.RateLimitError as exc:
                if attempt >= cfg.LLM_MAX_RETRIES:
                    logger.error("Anthropic API error after %d attempts: %s", attempt, exc)
                    return ScoringResult(error=str(exc))
                wait = retry_after_seconds(exc) or delay
                logger.warning("Rate limit (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                delay *= 2
            except _anthropic.InternalServerError as exc:
                if attempt >= cfg.LLM_MAX_RETRIES:
                    logger.error("Anthropic API error after %d attempts: %s", attempt, exc)
                    return ScoringResult(error=str(exc))
                logger.warning("Server error (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, delay)
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error("Anthropic API error: %s", exc)
                return ScoringResult(error=str(exc))


# ── OpenAI backend ─────────────────────────────────────────────────────────────

async def _score_openai(prompt: str, semaphore: asyncio.Semaphore) -> ScoringResult:
    from openai import AsyncOpenAI, RateLimitError as _OAIRateLimit, InternalServerError as _OAIServerError
    # max_retries=0: disable SDK's own retry so our backoff is the only mechanism
    client = AsyncOpenAI(api_key=cfg.OPENAI_API_KEY, max_retries=0)
    delay = cfg.LLM_RETRY_BASE_DELAY

    for attempt in range(1, cfg.LLM_MAX_RETRIES + 1):
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=cfg.OPENAI_CHAT_MODEL,
                    max_tokens=512,
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
                    logger.error("OpenAI API error after %d attempts: %s", attempt, exc)
                    return ScoringResult(error=str(exc))
                wait = retry_after_seconds(exc) or delay
                logger.warning("Rate limit (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, wait)
                await asyncio.sleep(wait)
                delay *= 2
            except _OAIServerError as exc:
                if attempt >= cfg.LLM_MAX_RETRIES:
                    logger.error("OpenAI API error after %d attempts: %s", attempt, exc)
                    return ScoringResult(error=str(exc))
                logger.warning("Server error (attempt %d/%d) — retrying in %.1fs", attempt, cfg.LLM_MAX_RETRIES, delay)
                await asyncio.sleep(delay)
                delay *= 2
            except Exception as exc:
                logger.error("OpenAI API error: %s", exc)
                return ScoringResult(error=str(exc))


# ── Public API ─────────────────────────────────────────────────────────────────

async def score_pairs_async(
    pairs: list[tuple[dict, dict]],
    concurrency: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ScoringResult]:
    """
    Score a list of (record_a, record_b) pairs concurrently.
    Returns results in the same order as the input pairs.
    progress_callback(completed, total) is called after each pair finishes.
    """
    if concurrency is None:
        concurrency = cfg.LLM_CONCURRENCY

    semaphore = asyncio.Semaphore(concurrency)
    provider = cfg.LLM_PROVIDER.lower()
    total = len(pairs)
    completed = 0

    async def _score_one(rec_a: dict, rec_b: dict) -> ScoringResult:
        prompt = _build_prompt(rec_a, rec_b)
        if provider == "anthropic":
            return await _score_anthropic(prompt, semaphore)
        elif provider == "openai":
            return await _score_openai(prompt, semaphore)
        else:
            return ScoringResult(error=f"Unknown LLM_PROVIDER: {provider!r}")

    from progress import progress
    bar = progress(total=total, desc="Stage 3: LLM scoring")

    async def _score_and_update(rec_a: dict, rec_b: dict) -> ScoringResult:
        nonlocal completed
        result = await _score_one(rec_a, rec_b)
        completed += 1
        bar.update(1)
        if progress_callback is not None:
            progress_callback(completed, total)
        return result

    tasks = [_score_and_update(a, b) for a, b in pairs]
    results = await asyncio.gather(*tasks)
    bar.close()
    return results


def score_pairs(
    pairs: list[tuple[dict, dict]],
    concurrency: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ScoringResult]:
    """Synchronous wrapper around score_pairs_async."""
    return asyncio.run(score_pairs_async(pairs, concurrency, progress_callback))


def score_single_pair(rec_a: dict, rec_b: dict) -> ScoringResult:
    """Convenience function to score a single pair synchronously."""
    results = score_pairs([(rec_a, rec_b)])
    return results[0]
