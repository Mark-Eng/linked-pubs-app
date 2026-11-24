"""
llm_io.py
─────────
Shared plumbing for talking to the embedding and LLM APIs.

These three helpers were previously duplicated across embeddings.py,
llm_scoring.py, and country_extraction.py — byte-identical copies of
retry_after_seconds, and two identical copies of the control-character
stripper. The JSON extractor existed only in country_extraction, so
llm_scoring kept using a weaker markdown-fence regex and repeatedly leaked
model reasoning into its output (commits 475ee55, 2b348ed). One
implementation means a fix lands everywhere.
"""

from __future__ import annotations

import json
import re

# Control characters that break JSON serialisation. Tabs (\x09) and newlines
# (\x0a) are deliberately kept: they occur legitimately in abstracts.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_JSON_DECODER = json.JSONDecoder()


def strip_control_chars(text: str) -> str:
    """Remove control characters that would break JSON serialisation."""
    return _CONTROL_CHARS_RE.sub("", text)


def extract_first_json_object(text: str) -> dict | None:
    """
    Find and parse the first complete JSON object in text, ignoring anything
    before or after it.

    Uses raw_decode, which stops at the end of the first valid value, rather
    than taking a first-'{'-to-last-'}' span: that span breaks as soon as
    trailing reasoning prose contains a brace of its own, because it then
    overshoots the real object's close and json.loads reports the prose as
    "Extra data" after an otherwise-valid parse.

    Returns None when no JSON object can be parsed. A non-object value (a bare
    list or number) also yields None — callers want an object.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = _JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        return obj if isinstance(obj, dict) else None
    return None


def retry_after_seconds(exc: Exception) -> float | None:
    """
    The API's suggested wait from a rate-limit response, if it provided one.

    Returns None when the exception carries no usable header, in which case
    the caller falls back to its own backoff. The attribute walk is guarded
    narrowly: an exception without .response raises AttributeError, a
    non-numeric header value raises ValueError or TypeError, and none of
    those should mask a real failure.
    """
    try:
        value = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
