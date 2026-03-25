"""
app.py — Med-Verify: Clinical Logic Engine
===========================================
Ophthalmology & Indian Clinical Standards — 100% on-premise inference.

UI Layout
---------
  Sidebar  : 🔬 Med-Verify brand · System Health · Clinical Specialty
             selector · Protocol Ingest (drag-and-drop)
  Main     : Source-aware Logic Badge · Verified Clinical Answer (left, 60%)
             · Evidence Vault — NLI chips, confidence bars, Direct PDF Clips
             (right, 40%)

V-RAG Pipeline
--------------
  1. ChromaDataStore     → Dynamic-K clinical semantic retrieval
  2. PairwiseAuditor     → Cross-protocol clinical contradiction detection
  3. ClinicalSynthesizer → Ollama llama3.1:8b answer (VRAM offloaded after)
  4. ClinicalAuditor     → DeBERTa-v3-large NLI verification (VRAM offloaded)
  5. PairwiseAuditor     → Dosage safety check on best chunk
  [Pipeline closes — text renders]
  6. EntityExtractor     → Primary disease entity extracted from query
  7. ImageSearch (async spinner) → DuckDuckGo clinical web image (silent fail)
  8. Visualizer          → PyMuPDF Direct PDF Clip for Evidence Vault

PHI Safety: No cloud APIs. All inference runs locally on RTX 4050.
"""

import os
import gc
import sys
import yaml
import tempfile
import datetime
import torch
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path) as _fh:
    _cfg = yaml.safe_load(_fh)
MODEL_NAME = _cfg.get("model_name", "llama3.1:8b")

# ─── Imports ──────────────────────────────────────────────────────────────────
from src.retrieval.vector_store   import ChromaDataStore
from src.audit.nli_judge          import PairwiseAuditor, check_contraindication_hardstop
from src.generation.llm_interface import ClinicalSynthesizer
from src.verifier                 import ClinicalAuditor
from src.ingestion.pdf_parser     import extract_text_with_metadata
from src.ingestion.semantic_chunker import HybridHierarchicalChunker
from src.utils.image_search       import get_clinical_images
from src.utils.entity_extractor   import extract_disease_entity
from src.utils.diagram_generator  import generate_clinical_diagram

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Med-Verify | Clinical Logic Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stHeader"] { background-color: #0d1117 !important; }

/* ── Logic Badges ───────────────────────────────────────────────────────── */
.badge-entailed {
    background: #1a4731; border: 2px solid #238636; color: #3fb950;
    border-radius: 8px; padding: 0.85rem 1.5rem; font-size: 1.2rem;
    font-weight: 700; letter-spacing: 0.04em; text-align: center;
    margin-bottom: 1.4rem; display: block;
}
.badge-contradiction {
    background: #3d1515; border: 2px solid #da3633; color: #f85149;
    border-radius: 8px; padding: 0.85rem 1.5rem; font-size: 1.2rem;
    font-weight: 700; letter-spacing: 0.04em; text-align: center;
    margin-bottom: 1.4rem; display: block;
}
.badge-neutral {
    background: #2d2200; border: 2px solid #9e6a03; color: #d29922;
    border-radius: 8px; padding: 0.85rem 1.5rem; font-size: 1.2rem;
    font-weight: 700; letter-spacing: 0.04em; text-align: center;
    margin-bottom: 1.4rem; display: block;
}

/* ── NLI Chips ──────────────────────────────────────────────────────────── */
.chip-entailed {
    background:#1a4731; color:#3fb950; border:1px solid #238636;
    padding:3px 11px; border-radius:20px; font-size:0.78rem;
    font-weight:600; display:inline-block;
}
.chip-contradiction {
    background:#3d1515; color:#f85149; border:1px solid #da3633;
    padding:3px 11px; border-radius:20px; font-size:0.78rem;
    font-weight:600; display:inline-block;
}
.chip-neutral {
    background:#2d2200; color:#d29922; border:1px solid #9e6a03;
    padding:3px 11px; border-radius:20px; font-size:0.78rem;
    font-weight:600; display:inline-block;
}

/* ── Dosage Conflict Badge ──────────────────────────────────────────────── */
.dosage-conflict {
    background:#3d2000; border:1px solid #f0883e; color:#f0883e;
    border-radius:6px; padding:0.55rem 1rem; font-size:0.83rem;
    font-weight:600; margin:0.5rem 0; display:block;
}
.dosage-ok {
    background:#1a3020; border:1px solid #3fb950; color:#3fb950;
    border-radius:6px; padding:0.55rem 1rem; font-size:0.83rem;
    font-weight:600; margin:0.5rem 0; display:block;
}

/* ── Section Labels ─────────────────────────────────────────────────────── */
.section-label {
    color:#8b949e; font-size:0.73rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase;
    margin-bottom:0.55rem; display:block;
}
.section-label-blue {
    color:#58a6ff; font-size:0.73rem; font-weight:700;
    letter-spacing:0.12em; text-transform:uppercase;
    margin-bottom:0.55rem; display:block;
}

/* ── Health Card ────────────────────────────────────────────────────────── */
.health-card {
    background:#0d1117; border:1px solid #30363d; border-radius:8px;
    padding:0.85rem 1rem; margin-bottom:0.8rem; font-size:0.82rem;
}
.health-card table { width:100%; border-collapse:collapse; }
.health-card td    { padding:4px 0; }
.dot-on  { color:#3fb950; }
.dot-off { color:#f85149; }

/* ── Conf Label ─────────────────────────────────────────────────────────── */
.conf-label {
    font-size:0.75rem; color:#8b949e; text-transform:uppercase;
    letter-spacing:0.08em; margin-bottom:1px;
}

/* ── File Uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background:#0d1117 !important; border:2px dashed #30363d !important;
    border-radius:8px !important;
}

/* ── Text Area ──────────────────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea {
    background-color:#161b22 !important; color:#c9d1d9 !important;
    border:1px solid #30363d !important; border-radius:8px !important;
    font-size:0.95rem !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stButton"] > button {
    background-color:#238636 !important; color:#ffffff !important;
    border:none !important; border-radius:6px !important;
    font-weight:600 !important; font-size:0.95rem !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stButton"] > button:hover {
    background-color:#2ea043 !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color:#161b22 !important; border:1px solid #30363d !important;
    border-radius:8px !important;
}

/* ── Misc ───────────────────────────────────────────────────────────────── */
hr { border-color:#30363d !important; }
[data-testid="stProgress"] > div > div { background-color:#30363d !important; }
::-webkit-scrollbar       { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#30363d; border-radius:3px; }

/* ── Critical Safety Alert ──────────────────────────────────────────────── */
.critical-alert {
    background: #2d0a0a;
    border: 2px solid #da3633;
    border-left: 6px solid #f85149;
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin: 0.8rem 0 1rem;
    animation: pulse-red 2s infinite;
}
.critical-alert-title {
    color: #f85149;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: 0.03em;
}
.critical-alert-body {
    color: #ffa198;
    font-size: 0.88rem;
    margin-top: 0.4rem;
    line-height: 1.6;
}
@keyframes pulse-red {
    0%   { border-left-color: #f85149; box-shadow: 0 0 0 0 rgba(248,81,73,0.4); }
    50%  { border-left-color: #ff7b72; box-shadow: 0 0 0 6px rgba(248,81,73,0); }
    100% { border-left-color: #f85149; box-shadow: 0 0 0 0 rgba(248,81,73,0); }
}

/* ── Correct Protocol Override ───────────────────────────────────────────── */
.protocol-override {
    background: #0d2818;
    border: 1px solid #238636;
    border-left: 5px solid #3fb950;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.7;
}

/* ── Demo Mode Banner ───────────────────────────────────────────────────── */
.demo-banner {
    background: #1a1a2e;
    border: 1px dashed #58a6ff;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: #58a6ff;
    margin-bottom: 0.8rem;
}

/* ── Clinical Anatomy Diagram ────────────────────────────────────────────── */
.diagram-container {
    background: #0d1117;
    border: 1px solid #30363d;
    border-top: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 1rem 1.2rem 0.8rem;
    margin-top: 1.4rem;
}
.diagram-title {
    color: #58a6ff;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.65rem;
    display: block;
}
.diagram-caption {
    color: #8b949e;
    font-size: 0.72rem;
    margin-top: 0.45rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _vram_stats():
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0) / 1024 ** 2
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        return used, total
    return 0.0, 0.0


def _protocol_label(filename: str) -> str:
    """Map a PDF filename to a short authority label for display."""
    fn = filename.lower()
    if "icmr"  in fn: return "ICMR"
    if "aiims" in fn: return "AIIMS"
    if "aios"  in fn: return "AIOS"
    if "aao"   in fn: return "AAO PPP 2025"
    if "nlem"  in fn: return "NLEM 2022"
    return os.path.splitext(filename)[0].upper()


def _source_badge(top_verdict: dict, verdicts: list) -> str:
    """
    Source-aware Logic Badge.
    When ENTAILED, appends the specific authority name:
      "✅ LOGICALLY ENTAILED BY ICMR"
    """
    css_map = {
        "ENTAILED":      "badge-entailed",
        "CONTRADICTION": "badge-contradiction",
        "NEUTRAL":       "badge-neutral",
    }
    css = css_map.get(top_verdict["status"], "badge-neutral")

    if top_verdict["status"] == "ENTAILED" and verdicts:
        # Find the highest-confidence entailed verdict
        best = max(
            (v for v in verdicts if v["status"] == "ENTAILED"),
            key=lambda v: v["confidence"],
            default=verdicts[0],
        )
        source = _protocol_label(best["metadata"].get("filename", ""))
        label  = f"LOGICALLY ENTAILED BY {source}"
        emoji  = "✅"
    elif top_verdict["status"] == "CONTRADICTION":
        label = "CONTRADICTION DETECTED"
        emoji = "🔴"
    else:
        label = "UNVERIFIED — NEUTRAL"
        emoji = "⚠️"

    return (
        f'<div class="{css}">'
        f'{emoji}&nbsp;&nbsp;{label}'
        f'<span style="font-size:0.85rem;font-weight:400;margin-left:1.4rem;">'
        f'Verification Confidence:&nbsp;<b>{top_verdict["confidence"]}%</b>'
        f'</span></div>'
    )


def _nli_chip(status: str, emoji: str) -> str:
    css = {
        "ENTAILED":      "chip-entailed",
        "CONTRADICTION": "chip-contradiction",
        "NEUTRAL":       "chip-neutral",
    }.get(status, "chip-neutral")
    return f'<span class="{css}">{emoji}&nbsp;{status}</span>'


def _conf_bar(label: str, value: float, color: str) -> None:
    hex_map = {"green": "#238636", "orange": "#9e6a03", "red": "#da3633"}
    hx = hex_map.get(color, "#58a6ff")
    st.markdown(f'<div class="conf-label">{label}</div>', unsafe_allow_html=True)
    st.progress(min(int(value), 100))
    st.markdown(
        f'<div style="font-size:0.82rem;color:{hx};font-weight:700;'
        f'text-align:right;margin-top:-0.45rem;">{value:.1f}%</div>',
        unsafe_allow_html=True,
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown(
        '<h2 style="color:#58a6ff;font-weight:700;letter-spacing:0.03em;'
        'margin-bottom:0.1rem;">🔬 Med-Verify</h2>'
        '<p style="color:#8b949e;font-size:0.78rem;margin-top:0;">'
        'Clinical Logic Engine · Ophthalmology Edition</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── System Health ─────────────────────────────────────────────────────────
    st.markdown('<span class="section-label">⚙ System Health</span>',
                unsafe_allow_html=True)
    used_mb, total_mb = _vram_stats()
    vram_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0.0
    gpu_ok   = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_properties(0).name if gpu_ok else "CPU only"

    st.markdown(
        f"""
        <div class="health-card">
          <table>
            <tr>
              <td style="color:#8b949e;">NLI Engine</td>
              <td style="text-align:right;">
                <span class="dot-on">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">&nbsp;DeBERTa-v3-large</span>
              </td>
            </tr>
            <tr>
              <td style="color:#8b949e;">LLM Engine</td>
              <td style="text-align:right;">
                <span class="dot-on">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">&nbsp;Ollama {MODEL_NAME}</span>
              </td>
            </tr>
            <tr>
              <td style="color:#8b949e;">Knowledge Base</td>
              <td style="text-align:right;">
                <span class="dot-on">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">&nbsp;ICMR · AIIMS · AAO</span>
              </td>
            </tr>
            <tr>
              <td style="color:#8b949e;">GPU</td>
              <td style="text-align:right;">
                <span class="{'dot-on' if gpu_ok else 'dot-off'}">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">&nbsp;{gpu_name}</span>
              </td>
            </tr>
            <tr>
              <td style="color:#8b949e;">VRAM&nbsp;(PyTorch)</td>
              <td style="text-align:right;color:#c9d1d9;">
                {used_mb:.0f}&nbsp;/&nbsp;{total_mb:.0f}&nbsp;MB
              </td>
            </tr>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total_mb > 0:
        st.progress(min(int(vram_pct), 100), text=f"VRAM {vram_pct:.1f}%")

    # ── Clinical Specialty Selector ───────────────────────────────────────────
    st.markdown('<hr style="margin:0.6rem 0;">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">🏥 Clinical Specialty</span>',
                unsafe_allow_html=True)
    specialty = st.selectbox(
        "Clinical Specialty",
        options=[
            "General Ophthalmology",
            "Glaucoma",
            "Cataract & Refractive",
            "Cornea & External Disease",
            "Retina & Vitreous",
            "Neuro-Ophthalmology",
            "Paediatric Ophthalmology",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.session_state["specialty"] = specialty

    # ── Active Collection ──────────────────────────────────────────────────────
    active_doc = st.session_state.get("active_doc", "ICMR · AIIMS · AAO · NLEM (53 protocols)")
    st.markdown(
        f'<div style="font-size:0.78rem;color:#8b949e;margin:0.4rem 0 0.8rem;">'
        f'📂 Active:&nbsp;<span style="color:#58a6ff;">{active_doc}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── Protocol Ingestion ────────────────────────────────────────────────────
    st.markdown('<span class="section-label">📥 Ingest New Protocol</span>',
                unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader(
        "Drop a clinical protocol / STG PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )
    year_str = st.text_input(
        "Publication year (optional)",
        placeholder="e.g. 2025",
        label_visibility="visible",
    )

    if uploaded_pdf is not None and st.button("⚡ Ingest Protocol"):
        with st.spinner(f"Ingesting `{uploaded_pdf.name}`…"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    tmp_path = tmp.name

                year = (
                    int(year_str.strip()) if year_str.strip().isdigit() else None
                )
                chunker = HybridHierarchicalChunker()
                store   = ChromaDataStore()
                blocks  = extract_text_with_metadata(tmp_path, doc_year=year)
                chunks  = chunker.chunk(blocks)
                n       = store.ingest(chunks)

                os.unlink(tmp_path)
                st.session_state["active_doc"] = uploaded_pdf.name
                st.success(f"✅ {n} chunks ingested from `{uploaded_pdf.name}`")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.markdown('<hr style="margin:0.8rem 0;">', unsafe_allow_html=True)

    # ── History Controls ───────────────────────────────────────────────────────
    history_count = len(st.session_state.get("query_history", []))
    if history_count:
        st.markdown(
            f'<div style="font-size:0.78rem;color:#8b949e;margin-bottom:0.4rem;">'
            f'🕐 <b>{history_count}</b> quer{"y" if history_count == 1 else "ies"} in session</div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state["query_history"] = []
            st.rerun()

    st.markdown('<hr style="margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.72rem;color:#8b949e;line-height:1.6;">'
        '🔒 <b>100% Local Processing</b><br>'
        'No data leaves this machine.<br>'
        'PHI-safe · Air-gapped inference.<br>'
        'Optimised for RTX 4050 / 6 GB VRAM.</div>',
        unsafe_allow_html=True,
    )


# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#e6edf3;font-weight:700;font-size:1.75rem;'
    'margin-bottom:0.1rem;">🔬 Med-Verify: Clinical Logic Engine</h1>'
    '<p style="color:#8b949e;margin-top:0;font-size:0.88rem;">'
    'NLI-Verified by&nbsp;<b>DeBERTa-v3-large</b>&nbsp;·&nbsp;'
    'Generated by&nbsp;<b>Llama 3.1 8B</b>&nbsp;·&nbsp;'
    'Sources:&nbsp;<b>ICMR · AIIMS RISHIKESH · AAO PPP 2025 · NLEM 2022</b></p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ─── Demo Mode ────────────────────────────────────────────────────────────────
_DEMO_QUERY = (
    "Patient presents with a corneal ulcer showing feathery margins and "
    "satellite lesions. A junior doctor has prescribed Prednisolone 1% eye "
    "drops four times daily. Should I continue this prescription? "
    "What is the correct treatment?"
)

col_demo, _ = st.columns([1, 3])
with col_demo:
    if st.button("🎭 Demo: Safety Hard-Stop", use_container_width=True,
                 help="Simulates a dangerous Steroid-in-Fungal-Keratitis prescription"):
        st.session_state["_demo_query"]  = _DEMO_QUERY
        st.session_state["_auto_submit"] = True
        st.rerun()

# Pick up prefilled query and auto-submit flag set by Demo button
_auto_submit = st.session_state.pop("_auto_submit", False)
_prefill     = st.session_state.get("_demo_query", "")


# ─── Query Form ───────────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=False):

    # ── View-Mode Toggle ───────────────────────────────────────────────────────
    view_mode = st.radio(
        "Response depth",
        options=["Short", "Brief"],
        index=1,
        horizontal=True,
        help=(
            "**Short** — 3-sentence summary: diagnosis + 1st-line drug + 1 warning.\n\n"
            "**Brief** — Full clinical protocol: etiology · dosage · "
            "contraindications · follow-up."
        ),
    )

    query = st.text_area(
        "Clinical Query",
        value=_prefill,
        placeholder=(
            "e.g. What is the first-line treatment for open-angle glaucoma "
            "with IOP > 30 mmHg per ICMR guidelines?"
        ),
        height=100,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🔍  Verify & Analyse", use_container_width=True)

# Merge auto-submit from Demo Mode button
if _auto_submit and _prefill:
    submitted = True
    query     = _prefill
    st.session_state.pop("_demo_query", None)     # consume so next run is clean

    st.markdown(
        '<div class="demo-banner">🎭 <b>Demo Mode active</b> — '
        'Running the Fungal Keratitis / Steroid safety hard-stop scenario.</div>',
        unsafe_allow_html=True,
    )


# ─── Validation ───────────────────────────────────────────────────────────────
if submitted and not query.strip():
    st.warning("⚠️ Please enter a clinical query before submitting.")
    st.stop()


# ─── V-RAG Pipeline ───────────────────────────────────────────────────────────
if submitted and query.strip():

    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []

    active_specialty = st.session_state.get("specialty", "General Ophthalmology")

    with st.status("🔄 Running Clinical V-RAG Pipeline…", expanded=True) as pipe_status:

        # Step 1 — Dynamic-K Semantic Retrieval
        st.write(f"📡 Dynamic-K retrieval ({active_specialty}) from protocol vector store…")
        store   = ChromaDataStore()
        results = store.search(query, specialty=active_specialty)

        if not results:
            pipe_status.update(label="❌ No protocols found", state="error", expanded=True)
            st.error(
                "No clinical context found. "
                "Run `python scripts/ingest_all.py --pdf-dir data/input_pdfs/` first."
            )
            st.stop()

        # Step 2 — Cross-Protocol Clinical Contradiction Check
        st.write("🔬 Cross-protocol clinical safety contradiction scan…")
        pairwise = PairwiseAuditor()
        conflict = pairwise.detect_logic_flips(results)
        pairwise.unload()

        # Step 3 — Answer Generation (VRAM offloaded after)
        st.write(f"🤖 Synthesising clinical answer via Ollama ({MODEL_NAME}) [{view_mode} mode]…")
        synthesizer = ClinicalSynthesizer()
        answer_raw  = synthesizer.generate_answer(query, results, conflict, view_mode=view_mode)

        answer_body = answer_raw
        if conflict.get("ConflictFound") and answer_raw.lstrip().startswith("⚠️"):
            parts = answer_raw.split("\n\n", 1)
            answer_body = parts[1] if len(parts) > 1 else answer_raw

        # Step 4 — NLI Answer Verification (VRAM offloaded after)
        st.write("🧠 NLI verification with DeBERTa-v3-large…")
        verifier    = ClinicalAuditor()
        verdicts    = verifier.verify_answer(answer_body, results)
        top_verdict = verifier.aggregate_verdict(verdicts)
        verifier.unload()

        # Step 5 — Dosage Safety Check on highest-confidence chunk
        dosage_report = None
        if verdicts:
            best_chunk   = max(verdicts, key=lambda v: v["confidence"])
            _tmp_auditor = PairwiseAuditor()
            dosage_report = _tmp_auditor.detect_dosage_contradiction(
                best_chunk["chunk_text"], answer_body
            )
            _tmp_auditor.unload()

        # Step 6 — Hard-Stop Contraindication Check (zero-latency rule engine)
        st.write("🛡 Running contraindication hard-stop safety check…")
        hard_stop_rule = check_contraindication_hardstop(query, answer_body)
        if hard_stop_rule:
            # Force the NLI verdict to CONTRADICTION regardless of model score
            top_verdict = {
                "status":     "CONTRADICTION",
                "confidence": 99.0,
                "emoji":      "🔴",
                "color":      "red",
                "label":      "CONTRADICTION DETECTED",
            }

        pipe_status.update(label="✅ Pipeline Complete", state="complete", expanded=False)

    # Entity extracted from query (used later in Evidence Vault image fetch)
    primary_condition = extract_disease_entity(query, results)

    # ── Critical Hard-Stop Safety Alert ───────────────────────────────────────
    if hard_stop_rule:
        st.markdown(
            f'<div class="critical-alert">'
            f'<div class="critical-alert-title">'
            f'⚠️ CRITICAL SAFETY ALERT: Potential Clinical Contradiction Detected'
            f'</div>'
            f'<div class="critical-alert-body">'
            f'<b>Rule triggered:</b> {hard_stop_rule["name"]}<br>'
            f'<b>Risk:</b> {hard_stop_rule["reason"]}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Dosage Safety Banner ───────────────────────────────────────────────────
    if dosage_report:
        if dosage_report["DosageConflict"]:
            st.markdown(
                f'<div class="dosage-conflict">⚠️ DOSAGE SAFETY ALERT — '
                f'AI suggested dosage(s) not found in source PDF: '
                f'<b>{", ".join(dosage_report["HallucDosages"])}</b>. '
                f'Verified from PDF: {", ".join(dosage_report["VerifiedDosages"]) or "none"}'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif dosage_report["VerifiedDosages"]:
            st.markdown(
                f'<div class="dosage-ok">✅ DOSAGE VERIFIED — '
                f'All dosage values confirmed in source protocol: '
                f'<b>{", ".join(dosage_report["VerifiedDosages"])}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Cross-Protocol Conflict Banner ────────────────────────────────────────
    if conflict.get("ConflictFound"):
        p_meta = conflict.get("PrioritizedChunk", {}).get("metadata", {})
        r_meta = conflict.get("RejectedChunk",    {}).get("metadata", {})
        p_auth = conflict.get("AuthorityA", "?")
        st.warning(
            f"⚠️ **CLINICAL PROTOCOL CONFLICT** — Contradiction detected between "
            f"`{r_meta.get('filename', 'Unknown')}` and "
            f"`{p_meta.get('filename', 'Unknown')}`. "
            f"Deferring to higher-authority source: "
            f"**{p_meta.get('Protocol_Name', _protocol_label(p_meta.get('filename','')))}** "
            f"(Authority Rank: {p_auth})."
        )

    st.markdown("---")

    # ── Dual-Pane Layout ───────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    # ── LEFT PANE (60%) — Verified Clinical Answer ─────────────────────────────
    with left_col:
        st.markdown(
            '<span class="section-label">📋 Verified Clinical Answer</span>',
            unsafe_allow_html=True,
        )

        if hard_stop_rule:
            # ── Correct Protocol Override (shown above blurred answer) ─────────
            st.markdown(
                '<span class="section-label" style="color:#3fb950;">'
                '✅ Correct Protocol (ICMR/AIIMS)</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="protocol-override">'
                f'{hard_stop_rule["correct_treatment"].replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # ── Blurred AI Answer with Reveal Checkbox ────────────────────────
            st.markdown(
                '<span class="section-label" style="color:#f85149;">'
                '🔴 AI Response (flagged — may contain dangerous advice)</span>',
                unsafe_allow_html=True,
            )
            reveal = st.checkbox(
                "⚠️ I understand the clinical risk — reveal AI response",
                value=False,
                key="reveal_blurred",
            )
            if reveal:
                with st.container(border=True):
                    st.markdown(answer_body)
            else:
                blurred_html = answer_body.replace("\n", "<br>")
                st.markdown(
                    f'<div style="filter:blur(6px);user-select:none;'
                    f'background:#161b22;border:1px solid #30363d;'
                    f'border-radius:8px;padding:1rem;font-size:0.9rem;'
                    f'color:#c9d1d9;line-height:1.7;">'
                    f'{blurred_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.container(border=True):
                st.markdown(answer_body)

        # ── Citations ──────────────────────────────────────────────────────────
        ts = (verdicts[0]["timestamp"] if verdicts
              else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        st.markdown(
            '<div style="margin-top:1rem;">'
            '<span style="color:#58a6ff;font-size:0.73rem;font-weight:700;'
            'letter-spacing:0.12em;text-transform:uppercase;">📎 Sources</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        for i, chunk in enumerate(results, 1):
            m      = chunk.get("metadata", {})
            fname  = m.get("filename", "Unknown")
            pg     = m.get("Page_Number", m.get("page", "?"))
            proto  = m.get("Protocol_Name") or _protocol_label(fname)
            year   = m.get("doc_year", m.get("statute_year", "—"))
            st.markdown(
                f'<div style="background:#161b22;border:1px solid #30363d;'
                f'border-left:3px solid #58a6ff;border-radius:6px;'
                f'padding:0.55rem 0.85rem;margin:0.35rem 0;font-size:0.82rem;">'
                f'<span style="color:#58a6ff;font-weight:700;">[{i}]</span>'
                f'&nbsp;<span style="color:#e6edf3;font-weight:600;">{fname}</span>'
                f'<span style="color:#8b949e;"> &nbsp;·&nbsp; </span>'
                f'<span style="color:#c9d1d9;">Page&nbsp;<b>{pg}</b></span>'
                f'<span style="color:#8b949e;"> &nbsp;·&nbsp; </span>'
                f'<span style="color:#8b949e;">{proto} · {year}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="font-size:0.72rem;color:#6e7681;margin-top:0.4rem;">'
            f'⏱ Verified at {ts} · Specialty: {active_specialty}</div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT PANE (40%) — Evidence Vault ─────────────────────────────────────
    with right_col:
        st.subheader("Visual Evidence")

        # ── NLI Confidence Score / Badge ───────────────────────────────────────
        _conf_bar("Overall NLI Confidence", top_verdict["confidence"], top_verdict["color"])
        st.divider()

        # ── Web-Retrieved Clinical Images (up to 3) ────────────────────────────
        # Fetched after pipeline closes — answer already visible in left pane.
        # SILENT FAIL: if no images found, section is entirely absent.
        with st.spinner("🔍 Fetching clinical reference images…"):
            img_urls = get_clinical_images(primary_condition, count=3)

        if img_urls:
            st.markdown(
                '<span class="section-label" style="color:#58a6ff;">'
                '🌐 Clinical Reference Images</span>',
                unsafe_allow_html=True,
            )
            img_cols = st.columns(len(img_urls))
            for col, url in zip(img_cols, img_urls):
                with col:
                    st.image(url, use_container_width=True)
            st.caption(f"Web Reference: {primary_condition} — Source: Clinical Web Search")
            st.divider()

    # ── Clinical Anatomy Diagram (full-width, below dual pane) ────────────────
    # Generated locally via matplotlib — split-pane anatomy matched to condition.
    # SILENT FAIL: section is absent when generation returns None.
    with st.spinner("🔬 Generating clinical anatomy diagram…"):
        diagram_bytes = generate_clinical_diagram(
            condition_entity=primary_condition,
            query=query,
            answer_text=answer_body,
        )

    if diagram_bytes:
        st.markdown(
            '<div class="diagram-container">'
            '<span class="diagram-title">🔬 Clinical Anatomy Reference — '
            'Visual Explanation</span>',
            unsafe_allow_html=True,
        )
        st.image(
            diagram_bytes,
            caption=(
                f"Split-pane anatomy diagram for: {primary_condition}  ·  "
                f"Panel A = cross-section / pathway  ·  Panel B = contextual anatomy  ·  "
                f"Labels sync with clinical text above"
            ),
            use_container_width=True,
        )
        st.markdown(
            '<div class="diagram-caption">'
            '📌 <b>Visual Sync:</b> Every anatomical term labelled in the diagram '
            'corresponds to a clinical value or structure described in the verified '
            'answer. Use the panels as a reference key while reading the protocol above.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── Save current result to session history ─────────────────────────────────
    _ts = verdicts[0]["timestamp"] if verdicts else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["query_history"].insert(0, {
        "query":          query,
        "answer_body":    answer_body,
        "top_verdict":    top_verdict,
        "citations":      [
            {
                "filename": c.get("metadata", {}).get("filename", "?"),
                "page":     c.get("metadata", {}).get("Page_Number",
                            c.get("metadata", {}).get("page", "?")),
                "proto":    c.get("metadata", {}).get("Protocol_Name")
                            or _protocol_label(c.get("metadata", {}).get("filename", "?")),
                "year":     c.get("metadata", {}).get("doc_year",
                            c.get("metadata", {}).get("statute_year", "—")),
            }
            for c in results
        ],
        "hard_stop_name":     hard_stop_rule["name"]     if hard_stop_rule else None,
        "hard_stop_severity": hard_stop_rule["severity"] if hard_stop_rule else None,
        "img_urls":       img_urls,
        "timestamp":      _ts,
        "view_mode":      view_mode,
        "specialty":      active_specialty,
    })
    # Keep at most 10 entries to avoid unbounded session growth
    st.session_state["query_history"] = st.session_state["query_history"][:10]

    # ── Previous Queries ───────────────────────────────────────────────────────
    past = st.session_state["query_history"][1:]   # [0] is the one just rendered
    if past:
        st.markdown("---")
        st.markdown(
            '<span style="color:#8b949e;font-size:0.73rem;font-weight:700;'
            'letter-spacing:0.12em;text-transform:uppercase;">🕐 Previous Queries</span>',
            unsafe_allow_html=True,
        )

        _verdict_color = {
            "ENTAILED":      "#238636",
            "NEUTRAL":       "#9e6a03",
            "CONTRADICTION": "#da3633",
        }
        _verdict_bg = {
            "ENTAILED":      "#1a4731",
            "NEUTRAL":       "#2d2200",
            "CONTRADICTION": "#3d1515",
        }

        for idx, entry in enumerate(past):
            v        = entry["top_verdict"]
            status   = v.get("status", "NEUTRAL")
            emoji    = v.get("emoji", "⚠️")
            conf     = v.get("confidence", 0)
            q_short  = entry["query"][:80] + ("…" if len(entry["query"]) > 80 else "")
            bg       = _verdict_bg.get(status,  "#161b22")
            col      = _verdict_color.get(status, "#8b949e")
            hs_badge = (
                f'&nbsp;<span style="background:#2d0a0a;color:#f85149;'
                f'border:1px solid #da3633;border-radius:10px;'
                f'padding:1px 7px;font-size:0.7rem;font-weight:700;">'
                f'⚠️ HARD-STOP</span>'
                if entry.get("hard_stop_name") else ""
            )

            with st.expander(
                f"{emoji}  {q_short}    [{entry['timestamp']}]",
                expanded=False,
            ):
                # Mini verdict badge
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {col};'
                    f'border-radius:6px;padding:0.5rem 1rem;margin-bottom:0.6rem;'
                    f'font-size:0.85rem;font-weight:700;color:{col};">'
                    f'{emoji}&nbsp;&nbsp;{status}&nbsp;&nbsp;'
                    f'<span style="font-weight:400;font-size:0.78rem;">'
                    f'Confidence: {conf}%</span>{hs_badge}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Hard-stop flag
                if entry.get("hard_stop_name"):
                    st.markdown(
                        f'<div style="background:#2d0a0a;border-left:3px solid #f85149;'
                        f'border-radius:4px;padding:0.4rem 0.8rem;font-size:0.8rem;'
                        f'color:#ffa198;margin-bottom:0.5rem;">'
                        f'🛑 <b>Hard-Stop Rule:</b> {entry["hard_stop_name"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Answer
                st.markdown(
                    f'<span style="color:#8b949e;font-size:0.72rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.1em;">Answer</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(entry["answer_body"])

                # Citations
                if entry["citations"]:
                    st.markdown(
                        '<span style="color:#58a6ff;font-size:0.72rem;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.1em;">📎 Sources</span>',
                        unsafe_allow_html=True,
                    )
                    for i, cit in enumerate(entry["citations"], 1):
                        st.markdown(
                            f'<div style="background:#161b22;border:1px solid #30363d;'
                            f'border-left:3px solid #58a6ff;border-radius:4px;'
                            f'padding:0.35rem 0.7rem;margin:0.25rem 0;font-size:0.79rem;">'
                            f'<b style="color:#58a6ff;">[{i}]</b>'
                            f'&nbsp;<span style="color:#e6edf3;">{cit["filename"]}</span>'
                            f'&nbsp;<span style="color:#8b949e;">·</span>'
                            f'&nbsp;<span style="color:#c9d1d9;">Page <b>{cit["page"]}</b></span>'
                            f'&nbsp;<span style="color:#8b949e;">· {cit["proto"]} · {cit["year"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Images (if any were saved)
                if entry.get("img_urls"):
                    st.markdown(
                        '<span style="color:#58a6ff;font-size:0.72rem;font-weight:700;'
                        'text-transform:uppercase;letter-spacing:0.1em;">🌐 Images</span>',
                        unsafe_allow_html=True,
                    )
                    img_c = st.columns(len(entry["img_urls"]))
                    for col_w, url in zip(img_c, entry["img_urls"]):
                        with col_w:
                            st.image(url, use_container_width=True)

                st.markdown(
                    f'<div style="font-size:0.71rem;color:#6e7681;margin-top:0.4rem;">'
                    f'⏱ {entry["timestamp"]} · {entry["specialty"]} · {entry["view_mode"]} mode'
                    f'</div>',
                    unsafe_allow_html=True,
                )
