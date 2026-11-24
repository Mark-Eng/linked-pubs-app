"""
progress.py
───────────
One place that decides how progress is displayed.

tqdm writes to stderr, so it never corrupted piped output, but with its
default settings it emits a carriage-return-separated redraw for every update.
In a terminal that animates a bar; in a log file or a captured worker stream it
is thousands of lines of noise.

Passing disable=None makes tqdm suppress itself when the output stream is not a
terminal, which is exactly the wanted behaviour in both cases. Routing every
call site through here also gives the eventual web layer a single place to
swap terminal bars for job-record updates, rather than editing a dozen loops.
"""

from __future__ import annotations

from typing import Any


def progress(iterable: Any = None, **kwargs: Any):
    """
    tqdm, but silent when output is not a terminal.

    An explicit `disable=` argument still wins, so a caller that must show or
    hide a bar regardless can say so.
    """
    from tqdm import tqdm

    kwargs.setdefault("disable", None)
    if iterable is None:
        return tqdm(**kwargs)
    return tqdm(iterable, **kwargs)
