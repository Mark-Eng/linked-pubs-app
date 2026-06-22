"""
trial_registration.py
─────────────────────
Extraction and comparison of study registration numbers.

Shared registration IDs are strong indicators of linked publications:
  - A shared ID almost certainly means two records report the same study.
  - Conflicting IDs (both records have one, but different) suggests they
    describe different studies and are unlikely to be linked.
  - If one or both records have no ID, no signal is produced.

Supported registries
────────────────────
  PROSPERO  Systematic review register (NIHR/York)  CRD42021124658
  NCT       ClinicalTrials.gov (US)                 NCT00123456
  ISRCTN    ISRCTN Registry (UK/international)      ISRCTN12345678
  ACTRN     Australia/New Zealand                   ACTRN12606000123456
  DRKS      German Clinical Trials Register         DRKS00000001
  ChiCTR    Chinese Clinical Trial Registry         ChiCTR-TRC-00000001
  CTRI      Clinical Trials Registry India          CTRI/2009/091/000021
  IRCT      Iranian Registry                        IRCT2012062210108N1
  PACTR     Pan African                             PACTR201609001654304
  EudraCT   EU Clinical Trials Register             2004-000274-02
  NTR       Netherlands Trial Register              NTR1234
  JPRN      Japan Primary Registries Network        JPRN-UMIN000001
  KCT       Korea Clinical Trial Registry           KCT0000001
  UMIN      University Hospital Medical Network     UMIN000001
  RPCEC     Cuban Public Registry                   RPCEC00000001
  SLCTR     Sri Lanka Clinical Trials Registry      SLCTR/2009/001
  TCTR      Thai Clinical Trials Registry           TCTR20140101001
  LBCTR     Lebanese Clinical Trials Registry       LBCTR2020040501
"""

from __future__ import annotations
import re


# ── Regex patterns ─────────────────────────────────────────────────────────────

_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        # PROSPERO systematic review registry (NIHR/University of York).
        # Format: CRD4 + year (4d) + sequential counter (6-7d). Word boundary
        # guards against matching within longer numeric strings.
        r'\bCRD4\d{8,}\b',
        r'\bNCT\d{8}\b',
        r'\bISRCTN\d{8}\b',
        r'\bACTRN\d{14}\b',
        r'\bDRKS\d{8}\b',
        # ChiCTR uses hyphens or underscores as separators
        r'\bChiCTR[-_][A-Z]{2,6}[-_]\d+\b',
        # CTRI uses slashes: CTRI/YYYY/MM/number
        r'\bCTRI/\d{4}/\d{2,3}/\d+',
        # IRCT: long numeric string followed by a letter and digit suffix
        r'\bIRCT\d{14,}[A-Z]\d+\b',
        r'\bPACTR\d{15}\b',
        # EudraCT: YYYY-NNNNNN-NN; year restricted to 2004+ (registry launch year)
        # to avoid matching arbitrary date-like strings
        r'\b(?:200[4-9]|20[1-9]\d)-\d{6}-\d{2}\b',
        r'\bNTR\d{4,6}\b',
        r'\bJPRN-[A-Z0-9-]+\b',
        r'\bKCT\d{7}\b',
        r'\bUMIN\d{6,}\b',
        r'\bRPCEC\d{8}\b',
        r'\bSLCTR/\d{4}/\d+',
        r'\bTCTR\d{11}\b',
        r'\bLBCTR\d{10}\b',
    ]
]

# Characters that may appear as trailing punctuation after a TRN in running text
_TRAILING_PUNCT = re.compile(r'[^\w/]+$')


def _normalise(raw: str) -> str:
    """Strip trailing punctuation and uppercase for consistent comparison."""
    return _TRAILING_PUNCT.sub("", raw).upper()


def extract_trial_ids(text: str) -> frozenset[str]:
    """
    Extract and normalise all trial registration numbers from a text string.

    Searches the text (typically an abstract) against all known registry
    patterns.  Returns a frozenset of normalised ID strings (uppercase,
    trailing punctuation removed).  Returns an empty frozenset if none found.
    """
    if not text:
        return frozenset()
    found: set[str] = set()
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            found.add(_normalise(match.group()))
    return frozenset(found)


def trial_id_signal(
    ids_a: frozenset[str],
    ids_b: frozenset[str],
) -> tuple[int, str]:
    """
    Compare two sets of trial registration IDs and return a directional signal.

    Returns
    -------
    (+1, description)  — at least one ID is shared → strong positive signal
    (-1, description)  — both have IDs but none match → strong negative signal
    ( 0, "")           — one or both records have no IDs → no signal
    """
    if not ids_a or not ids_b:
        return 0, ""

    shared = ids_a & ids_b
    if shared:
        return 1, "Shared trial ID(s): " + ", ".join(sorted(shared))

    return -1, (
        "Conflicting trial IDs: "
        + ", ".join(sorted(ids_a))
        + " vs "
        + ", ".join(sorted(ids_b))
    )
