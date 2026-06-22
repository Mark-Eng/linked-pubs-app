"""
app.py
──────
Streamlit web app for the Linked Publications Checker.
Upload a RIS file, configure scoring parameters, and download an Excel
report of candidate linked-publication pairs.
"""

from __future__ import annotations

import datetime
import os
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
load_dotenv(override=True)   # override=True so .env changes are picked up without restarting

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Linked Publications Checker",
    page_icon="🔗",
    layout="centered",
)

# ── API keys ──────────────────────────────────────────────────────────────────
# Read from Streamlit secrets (Community Cloud) with .env / env-var fallback.
def _secret(key: str) -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, "")

OPENAI_KEY    = _secret("OPENAI_API_KEY")
ANTHROPIC_KEY = _secret("ANTHROPIC_API_KEY")

# ── Default values ────────────────────────────────────────────────────────────
_WEIGHTS_DEFAULT = {
    "Title":     0.20,
    "Author":    0.30,
    "Embedding": 0.10,
    "Year":      0.05,
    "LLM":       0.35,
}
_COUNTRY_BONUS_DEFAULT   =  0.05
_COUNTRY_PENALTY_DEFAULT = -0.40

_PROMPT_PATH     = Path(__file__).parent / "prompts" / "semantic_similarity.txt"
_DEFAULT_PROMPT  = _PROMPT_PATH.read_text(encoding="utf-8")

_LLM_OPTIONS = {
    "Claude Haiku 4.5":   "anthropic",
    "OpenAI GPT 4o-mini": "openai",
}
BASIC_PARAMS = ["Title", "Author", "Embedding", "Year"]
COMING_SOON  = ["Disease", "Species"]

# ── Session state ─────────────────────────────────────────────────────────────
# job_state is a plain dict passed directly to the background thread so it can
# write results without needing a Streamlit execution context (st.session_state
# writes from non-Streamlit threads are silently dropped).
def _fresh_job_state() -> dict:
    return {"status": "idle", "result": None, "error": "", "log": []}

if "job_state" not in st.session_state:
    st.session_state.job_state = _fresh_job_state()
if "result_filename" not in st.session_state:
    st.session_state.result_filename = "linked_pairs.xlsx"
if "_thread" not in st.session_state:
    st.session_state._thread = None


# ── Background runner ─────────────────────────────────────────────────────────
def _run_in_background(tmp_path: str, app_config: dict, state: dict):
    import traceback

    def _log(msg: str):
        state["log"].append(msg)

    try:
        from app_runner import run_pipeline
        state["result"] = run_pipeline(tmp_path, app_config, _log)
        state["status"] = "done"
    except BaseException as exc:
        # BaseException catches SystemExit raised by sys.exit() calls inside
        # the pipeline (file_check.py uses sys.exit on parse errors).
        state["error"] = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        state["status"] = "error"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Form (always rendered; progress/results appended below the run button)
# ─────────────────────────────────────────────────────────────────────────────
_js      = st.session_state.job_state   # shorthand; same dict object throughout
_running = (_js["status"] == "running")

st.title("Linked Publications Checker")
st.caption(
    "Upload a RIS file and configure scoring parameters. "
    "The tool compares every pair of references and flags those likely to be "
    "linked publications (e.g. a working paper and its journal counterpart)."
)

# ── 1. Upload ──────────────────────────────────────────────────────────────────
st.subheader("1. Upload References")
uploaded_file = st.file_uploader(
    "Select a RIS file",
    type=["ris"],
    help="Placeholder text",
    label_visibility="collapsed",
)
if uploaded_file:
    st.caption(f"File: **{uploaded_file.name}**")
    try:
        import io
        import rispy
        _entries = rispy.load(io.TextIOWrapper(io.BytesIO(uploaded_file.getvalue()), encoding="utf-8-sig"))
        _n = sum(1 for e in _entries if e.get("title") or e.get("primary_title"))
        st.caption(f"References parsed: **{_n}**")
    except Exception:
        pass

st.divider()

# ── 2. Basic Parameters ───────────────────────────────────────────────────────
st.subheader("2. Basic Parameters")
st.caption(
    "Choose which signals to include and set their weights. "
    "The enabled weights (plus the LLM weight if LLM scoring is on) must sum to **1.00**."
)

# Header row
_c1, _c2, _c3 = st.columns([0.5, 3, 1.5])
_c2.markdown("<small>**Parameter**</small>", unsafe_allow_html=True)
_c3.markdown("<small>**Weight**</small>", unsafe_allow_html=True)

param_enabled: dict[str, bool] = {}
param_weight:  dict[str, float] = {}

for param in BASIC_PARAMS:
    col1, col2, col3 = st.columns([0.5, 3, 1.5])
    with col1:
        enabled = st.checkbox(
            f"Enable {param}",
            value=True,
            key=f"en_{param}",
            label_visibility="collapsed",
        )
    with col2:
        label_style = "color:#333;font-weight:500" if enabled else "color:#aaa"
        st.markdown(f"<span style='{label_style}'>{param}</span>", unsafe_allow_html=True)
    with col3:
        w = st.number_input(
            f"Weight for {param}",
            min_value=0.0,
            max_value=1.0,
            value=_WEIGHTS_DEFAULT[param],
            step=0.01,
            format="%.2f",
            disabled=not enabled,
            key=f"w_{param}",
            label_visibility="collapsed",
            help="Placeholder text",
        )
    param_enabled[param] = enabled
    param_weight[param]  = w if enabled else 0.0

st.divider()

# ── 3. LLM Scoring ────────────────────────────────────────────────────────────
st.subheader("3. LLM Scoring")
use_llm = st.checkbox(
    "Use LLM scoring",
    value=True,
    key="use_llm",
    help="Placeholder text",
)

llm_weight   = 0.0
llm_provider = "anthropic"
llm_min: float | None = None
llm_max: float | None = None
llm_prompt: str | None = None

if use_llm:
    _ll1, _ll2 = st.columns(2)
    with _ll1:
        llm_label = st.selectbox(
            "LLM provider",
            list(_LLM_OPTIONS.keys()),
            key="llm_provider",
            help="Placeholder text",
        )
        llm_provider = _LLM_OPTIONS[llm_label]
    with _ll2:
        llm_weight = st.number_input(
            "LLM weight",
            min_value=0.0,
            max_value=1.0,
            value=_WEIGHTS_DEFAULT["LLM"],
            step=0.01,
            format="%.2f",
            key="w_llm",
            help=(
                "The share of the composite score assigned to the LLM judgment "
                "for pairs that are sent to the LLM. Pairs that do not meet the "
                "score-range filter (or the author/title pre-filter) are not sent "
                "to the LLM — for those pairs this weight is redistributed "
                "proportionally among the other enabled components."
            ),
        )

    llm_range = st.slider(
        "Score range for LLM",
        min_value=0.0,
        max_value=1.0,
        value=(0.15, 1.0),
        step=0.05,
        key="llm_range",
        help="Placeholder text",
    )
    llm_min, llm_max = float(llm_range[0]), float(llm_range[1])

    use_default_prompt = st.checkbox(
        "Use default prompt",
        value=True,
        key="use_default_prompt",
        help="Placeholder text",
    )
    if not use_default_prompt:
        llm_prompt = st.text_area(
            "LLM prompt",
            value=_DEFAULT_PROMPT,
            height=300,
            key="custom_prompt",
            help="Placeholder text",
        )

    # Key-missing warning
    if llm_provider == "anthropic" and not ANTHROPIC_KEY:
        st.warning("⚠ No Anthropic API key found. Add ANTHROPIC_API_KEY to your Streamlit secrets or .env file.")
    elif llm_provider == "openai" and not OPENAI_KEY:
        st.warning("⚠ No OpenAI API key found. Add OPENAI_API_KEY to your Streamlit secrets or .env file.")

# ── Weight total (Basic + LLM) ────────────────────────────────────────────────
basic_total  = sum(param_weight[p] for p in BASIC_PARAMS)
weight_total = basic_total + llm_weight
weight_valid = abs(weight_total - 1.0) < 0.001

if use_llm:
    breakdown = f"Basic: **{basic_total:.2f}** + LLM: **{llm_weight:.2f}** = **{weight_total:.2f}**"
else:
    breakdown = f"Basic: **{basic_total:.2f}** = **{weight_total:.2f}**"

if weight_valid:
    st.success(f"{breakdown} ✓")
else:
    st.error(f"{breakdown} — weights must sum to exactly 1.00 before running.")

st.divider()

# ── 4. Asymmetric Parameters ──────────────────────────────────────────────────
st.subheader("4. Asymmetric Parameters")
st.caption(
    "These signals adjust the composite score up or down based on whether a "
    "characteristic matches between two references, rather than contributing a "
    "fixed weighted component."
)

# Country
_ac1, _ac2, _ac3 = st.columns([0.5, 2.5, 2.5])
with _ac1:
    use_country = st.checkbox(
        "Use country signal",
        value=True,
        key="use_country",
        label_visibility="collapsed",
        help="Placeholder text",
    )
with _ac2:
    country_style = "color:#333;font-weight:500" if use_country else "color:#aaa"
    st.markdown(f"<span style='{country_style}'>Country</span>", unsafe_allow_html=True)
with _ac3:
    modifier_choice = st.selectbox(
        "Country modifier",
        ["Use default modifier", "Use custom modifier"],
        key="country_modifier_choice",
        label_visibility="collapsed",
        disabled=not use_country,
        help="Placeholder text",
    )

use_custom_country = use_country and (modifier_choice == "Use custom modifier")

if use_custom_country:
    _cm1, _cm2, _cm3 = st.columns([0.5, 2.5, 2.5])
    with _cm2:
        country_bonus = st.number_input(
            "Same-country bonus",
            min_value=0.0,
            max_value=1.0,
            value=_COUNTRY_BONUS_DEFAULT,
            step=0.05,
            format="%.2f",
            key="country_bonus",
            help="Placeholder text",
        )
    with _cm3:
        country_penalty = st.number_input(
            "Different-country penalty",
            min_value=-1.0,
            max_value=0.0,
            value=_COUNTRY_PENALTY_DEFAULT,
            step=0.05,
            format="%.2f",
            key="country_penalty",
            help="Placeholder text",
        )
else:
    country_bonus   = _COUNTRY_BONUS_DEFAULT   if use_country else 0.0
    country_penalty = _COUNTRY_PENALTY_DEFAULT if use_country else 0.0

# Coming-soon rows
for cs in COMING_SOON:
    _sc1, _sc2, _sc3 = st.columns([0.5, 2.5, 2.5])
    with _sc1:
        st.checkbox(f"Enable {cs}", value=False, disabled=True, key=f"cs_{cs}", label_visibility="collapsed")
    with _sc2:
        st.markdown(f"<span style='color:#bbb'>{cs} &nbsp;<small><em>(coming soon)</em></small></span>", unsafe_allow_html=True)

st.divider()

# ── 5. Output Settings ─────────────────────────────────────────────────────────
st.subheader("5. Output Settings")
output_threshold = st.number_input(
    "Minimum composite score to include in output",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.05,
    format="%.2f",
    key="output_threshold",
    help="Placeholder text",
)

st.divider()

# ── Run button ────────────────────────────────────────────────────────────────
run_ready = (uploaded_file is not None) and weight_valid and not _running

if not _running:
    if uploaded_file is None and not weight_valid:
        st.caption("Upload a RIS file and fix the weight total to run.")
    elif uploaded_file is None:
        st.caption("Upload a RIS file to run.")
    elif not weight_valid:
        st.caption("Fix the weight total to run.")

if st.button(
    "Check References",
    type="primary",
    disabled=not run_ready,
):
    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(
        suffix=Path(uploaded_file.name).suffix or ".ris",
        delete=False,
    ) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    app_config = {
        "WEIGHT_TITLE":             param_weight["Title"],
        "WEIGHT_AUTHOR":            param_weight["Author"],
        "WEIGHT_EMBEDDING":         param_weight["Embedding"],
        "WEIGHT_YEAR":              param_weight["Year"],
        "WEIGHT_LLM":               llm_weight,
        "use_country":              use_country,
        "COUNTRY_OVERLAP_BONUS":    country_bonus,
        "COUNTRY_DISJOINT_PENALTY": country_penalty,
        "run_llm":                  use_llm,
        "LLM_PROVIDER":             llm_provider,
        "llm_min":                  llm_min,
        "llm_max":                  llm_max,
        "llm_prompt":               llm_prompt,
        "OUTPUT_THRESHOLD":         output_threshold,
        "OPENAI_API_KEY":           OPENAI_KEY,
        "ANTHROPIC_API_KEY":        ANTHROPIC_KEY,
    }

    date_str = datetime.date.today().strftime("%Y%m%d")
    stem     = Path(uploaded_file.name).stem
    st.session_state.result_filename = f"linked_pairs_{stem}_{date_str}.xlsx"

    state = _fresh_job_state()
    state["status"] = "running"
    st.session_state.job_state = state

    t = threading.Thread(
        target=_run_in_background,
        args=(tmp_path, app_config, state),
        daemon=True,
    )
    st.session_state._thread = t
    t.start()
    st.rerun()

def _render_log(messages: list[str]) -> None:
    """Render log messages, converting PROGRESS:done:total lines into progress bars."""
    last_progress: dict[str, tuple[int, int]] = {}  # label → (done, total)
    text_msgs = []
    for msg in messages:
        if msg.startswith("PROGRESS:"):
            _, done_s, total_s = msg.split(":")
            last_progress["bar"] = (int(done_s), int(total_s))
        else:
            text_msgs.append(msg)

    for msg in text_msgs:
        st.write(f"• {msg}")
    if last_progress:
        done, total = last_progress["bar"]
        frac = done / total if total else 0.0
        st.progress(frac, text=f"{done} / {total}")


# ── Progress / results (always at the bottom, below the run button) ───────────
if _js["status"] == "running":
    st.divider()
    thread: threading.Thread | None = st.session_state._thread
    with st.status("Running pipeline…", expanded=True):
        _render_log(_js["log"])
    if thread and thread.is_alive():
        time.sleep(1)
        st.rerun()
    else:
        if _js["status"] == "running":
            _js["status"] = "error"
            _js["error"] = "Pipeline thread ended without reporting a result."
        st.rerun()

elif _js["status"] == "done":
    st.divider()
    with st.status("Pipeline complete", state="complete", expanded=False):
        _render_log(_js["log"])
    st.success("Run complete. Download your results below.")
    st.download_button(
        label="📥 Download results (Excel)",
        data=_js["result"],
        file_name=st.session_state.result_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

elif _js["status"] == "error":
    st.divider()
    st.error(f"**Pipeline error:**\n\n```\n{_js['error']}\n```")
    if st.button("Dismiss error"):
        st.session_state.job_state = _fresh_job_state()
        st.rerun()
