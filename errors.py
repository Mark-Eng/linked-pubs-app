"""
errors.py
─────────
Exception types raised deliberately by this tool, and the helper that turns
them into a clean CLI exit.

Why these exist
───────────────
The scripts began as CLIs, so their validation called sys.exit() directly —
including inside `load_records()` and the Excel/RIS parsers, which are library
functions. That is fine for a terminal but wrong anywhere else: a web worker
reusing the parsers as import adapters would be *terminated* by a spreadsheet
missing its `title` column, instead of returning a validation message to the
user who uploaded it.

The rule now is: **library code raises, entry points exit.** Every module
raises one of the types below; each CLI's `main()` wraps its body in
`cli_entry()`, which logs the message and exits 1.

Choosing a type
───────────────
The distinction is who can fix the problem, because that decides what a web
interface should do with it:

  InputError          The submitted data is wrong. Show the message to the
                      user who submitted it; they can correct and retry.
  ConfigurationError  The deployment is wrong (missing dependency, missing
                      credential, unset setting). Alert an operator; retrying
                      the same request will not help.
  BaseIndexError      The base index is missing, incomplete, or unreadable.
                      Usually means a build job needs to run, or that
                      `git lfs pull` was skipped.

All three inherit from LinkedPubsError, so a caller that does not care can
catch just that.
"""

from __future__ import annotations

import logging
from collections.abc import Callable


class LinkedPubsError(Exception):
    """Base class for every error this tool raises on purpose."""


class InputError(LinkedPubsError):
    """Input data is malformed, empty, or of an unsupported type."""


class ConfigurationError(LinkedPubsError):
    """A dependency, credential, or setting the tool needs is missing."""


class BaseIndexError(LinkedPubsError):
    """A base index is absent, incomplete, or unreadable."""


def cli_entry(run: Callable[[], None], logger: logging.Logger) -> None:
    """
    Run a CLI body, converting an expected error into a logged exit code 1.

    Keeps `sys.exit` out of the library layer while preserving CLI behaviour:
    a clear one-line message and a non-zero status. Unexpected exceptions are
    deliberately left to propagate, so genuine bugs still produce a traceback
    rather than being flattened into "something went wrong".
    """
    try:
        run()
    except LinkedPubsError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
