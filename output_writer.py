"""
output_writer.py
────────────────
Shared Excel output logic used by both batch_check.py and file_check.py.

Creates a two-sheet workbook:
  Sheet 1 — Summary     : run metadata, score distribution, top findings
  Sheet 2 — Pairs       : one row per candidate pair, all scores + LLM rationale

Colour coding (composite score):
  ≥ 0.75  — red    (very likely linked)
  ≥ 0.55  — orange (likely linked)
  ≥ 0.40  — yellow (possibly linked)
"""

from __future__ import annotations
import datetime
import re
from pathlib import Path
from typing import Any

_ILLEGAL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitise(val: Any) -> Any:
    """Strip Excel-illegal control characters from strings."""
    if isinstance(val, str):
        return _ILLEGAL_CHARS_RE.sub("", val)
    return val

from openpyxl import Workbook
from openpyxl.styles import (
    Alignment, Font, PatternFill, Border, Side
)
from openpyxl.utils import get_column_letter

from scoring import PairScore


# ── Colour constants ───────────────────────────────────────────────────────────
_RED      = "FFFF9999"
_ORANGE   = "FFFFC266"
_YELLOW   = "FFFFFF99"
_HEADER   = "FF595959"  # charcoal grey header (score/neutral columns)
_HEADER_A = "FF4472C4"  # blue header (A columns)
_HEADER_B = "FF548235"  # forest green header (B columns)
_FILL_A   = "FFDCE6F1"  # light blue data cells (A columns)
_FILL_B   = "FFE2EFDA"  # light green data cells (B columns)
_WHITE    = "FFFFFFFF"


def _score_fill(score: float) -> PatternFill | None:
    if score >= 0.75:
        return PatternFill("solid", fgColor=_RED)
    if score >= 0.55:
        return PatternFill("solid", fgColor=_ORANGE)
    if score >= 0.40:
        return PatternFill("solid", fgColor=_YELLOW)
    return None


def _header_fill() -> PatternFill:
    return PatternFill("solid", fgColor=_HEADER)


def _col_group(field_key: str) -> str | None:
    """Return 'a', 'b', or None based on the field key suffix."""
    if field_key.endswith("_a"):
        return "a"
    if field_key.endswith("_b"):
        return "b"
    return None


def _thin_border() -> Border:
    side = Side(style="thin")
    return Border(left=side, right=side, top=side, bottom=side)


def _set_column_widths(ws, widths: dict[str, int]):
    for col_letter, width in widths.items():
        ws.column_dimensions[col_letter].width = width


# ── Pairs sheet ────────────────────────────────────────────────────────────────

_PAIR_COLUMNS = [
    ("Composite Score",    "composite",           10),
    ("LLM Score",          "llm_score",            9),
    ("Author Score",       "author_score",          9),
    ("Title Score",        "title_score",           9),
    ("Embedding Sim",      "embedding_sim",          9),
    ("Year Score",         "year_score",             9),
    ("Country Signal",     "country_signal_value",   9),
    ("Country Info",       "country_description",   30),
    ("Trial ID Signal",    "trial_id_signal_value",  9),
    ("Trial ID Info",      "trial_id_description",  40),
    ("ID A",               "id_a",                  8),
    ("Source project A",   "source_project_a",      25),
    ("Status A",           "status_a",              18),
    ("Title A",            "title_a",               45),
    ("Authors A",          "authors_a",             30),
    ("Year A",             "year_a",                 7),
    ("Countries A",        "countries_a",           20),
    ("ID B",               "id_b",                  8),
    ("Source project B",   "source_project_b",      25),
    ("Status B",           "status_b",              18),
    ("Title B",            "title_b",               45),
    ("Authors B",          "authors_b",             30),
    ("Year B",             "year_b",                 7),
    ("Countries B",        "countries_b",           20),
    ("LLM Rationale",      "llm_rationale",         60),
    ("LLM Key Evidence",   "llm_key_evidence",      40),
    ("LLM Confidence",     "llm_confidence",        12),
    ("Matched Authors",    "matched_authors",        35),
]


def _pair_row(ps: PairScore, records: dict) -> dict[str, Any]:
    rec_a = records.get(ps.id_a, {})
    rec_b = records.get(ps.id_b, {})

    def fmt_authors(authors):
        return "; ".join(authors) if authors else ""

    def fmt_list(lst):
        return "; ".join(str(x) for x in lst) if lst else ""

    def fmt_matched(pairs):
        return "; ".join(f"{a} ↔ {b} ({s:.2f})" for a, b, s in pairs) if pairs else ""

    return {
        "composite":           ps.composite,
        "llm_score":           ps.llm_score,
        "author_score":        ps.author_score,
        "title_score":         ps.title_score,
        "embedding_sim":       ps.embedding_sim,
        "year_score":          ps.year_score,
        "country_signal_value": ps.country_signal_value,
        "country_description": ps.country_description,
        "trial_id_signal_value": ps.trial_id_signal_value,
        "trial_id_description": ps.trial_id_description,
        "id_a":                ps.id_a,
        "source_project_a":    rec_a.get("source_project") or "NA",
        "status_a":            rec_a.get("status") or "",
        "title_a":             rec_a.get("title", ""),
        "authors_a":           fmt_authors(rec_a.get("authors", [])),
        "year_a":              rec_a.get("year"),
        "countries_a":         fmt_list(ps.countries_a),
        "id_b":                ps.id_b,
        "source_project_b":    rec_b.get("source_project") or "NA",
        "status_b":            rec_b.get("status") or "",
        "title_b":             rec_b.get("title", ""),
        "authors_b":           fmt_authors(rec_b.get("authors", [])),
        "year_b":              rec_b.get("year"),
        "countries_b":         fmt_list(ps.countries_b),
        "llm_rationale":       ps.llm_rationale,
        "llm_key_evidence":    fmt_list(ps.llm_key_evidence),
        "llm_confidence":      ps.llm_confidence,
        "matched_authors":     fmt_matched(ps.matched_authors),
    }


def write_excel(
    pair_scores: list[PairScore],
    records: dict,          # id → record dict
    output_path: str | Path,
    run_metadata: dict | None = None,
):
    """
    Write scored pairs to an Excel workbook.

    Parameters
    ----------
    pair_scores   : list of PairScore (pre-sorted by composite desc)
    records       : mapping of id → record dict (for title/author lookup)
    output_path   : path for the output .xlsx file
    run_metadata  : optional dict of key→value to include in summary sheet
    """
    wb = Workbook()

    # ── Sheet 1: Summary ───────────────────────────────────────────────────────
    ws_sum = wb.active
    ws_sum.title = "Summary"

    def write_kv(row, key, val):
        ws_sum.cell(row=row, column=1, value=key).font = Font(bold=True)
        ws_sum.cell(row=row, column=2, value=val)

    r = 1
    ws_sum.cell(r, 1, "Linked Publications Detection — Run Summary").font = Font(bold=True, size=14)
    r += 2

    write_kv(r, "Run date/time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    r += 1
    write_kv(r, "Total pairs above threshold", len(pair_scores))
    r += 1

    if run_metadata:
        for k, v in run_metadata.items():
            write_kv(r, k, v)
            r += 1
    r += 1

    # Score distribution
    ws_sum.cell(r, 1, "Score Distribution").font = Font(bold=True)
    r += 1
    brackets = [
        ("Very likely linked (≥ 0.75)", 0.75),
        ("Likely linked (0.55 – 0.74)",  0.55),
        ("Possibly linked (0.40 – 0.54)", 0.40),
    ]
    for label, threshold in brackets:
        n = sum(1 for ps in pair_scores if ps.composite >= threshold)
        write_kv(r, label, n)
        r += 1

    ws_sum.column_dimensions["A"].width = 40
    ws_sum.column_dimensions["B"].width = 25

    # ── Sheet 2: Pairs ─────────────────────────────────────────────────────────
    ws_pairs = wb.create_sheet("Pairs")

    # Header row
    header_font = Font(bold=True, color="FFFFFFFF")
    _group_header_fill = {
        "a": PatternFill("solid", fgColor=_HEADER_A),
        "b": PatternFill("solid", fgColor=_HEADER_B),
    }
    for col_idx, (label, field_key, _) in enumerate(_PAIR_COLUMNS, start=1):
        cell = ws_pairs.cell(row=1, column=col_idx, value=label)
        cell.fill = _group_header_fill.get(_col_group(field_key)) or _header_fill()
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    ws_pairs.row_dimensions[1].height = 30
    ws_pairs.freeze_panes = "A2"

    _group_data_fill = {
        "a": PatternFill("solid", fgColor=_FILL_A),
        "b": PatternFill("solid", fgColor=_FILL_B),
    }

    # Data rows
    for row_idx, ps in enumerate(pair_scores, start=2):
        row_data = _pair_row(ps, records)
        score_fill = _score_fill(ps.composite)
        for col_idx, (_, field_key, _) in enumerate(_PAIR_COLUMNS, start=1):
            val = _sanitise(row_data.get(field_key))
            cell = ws_pairs.cell(row=row_idx, column=col_idx, value=val)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            group = _col_group(field_key)
            if group:
                cell.fill = _group_data_fill[group]
            elif col_idx == 1 and score_fill:
                cell.fill = score_fill
            if isinstance(val, float):
                cell.number_format = "0.000"

    # Column widths
    for col_idx, (_, _, width) in enumerate(_PAIR_COLUMNS, start=1):
        ws_pairs.column_dimensions[get_column_letter(col_idx)].width = width

    wb.save(output_path)
    print(f"Output saved to: {output_path}")
