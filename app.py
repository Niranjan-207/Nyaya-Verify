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
  5. Visualizer          → PyMuPDF Direct PDF Clip for Evidence Vault

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
from src.audit.nli_judge          import PairwiseAuditor
from src.generation.llm_interface import ClinicalSynthesizer
from src.verifier                 import ClinicalAuditor
from src.ingestion.pdf_parser     import extract_text_with_metadata
from src.ingestion.semantic_chunker import HybridHierarchicalChunker
from src.utils.visualizer         import extract_pdf_clip

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

/* ── Citation Footer ────────────────────────────────────────────────────── */
.citation-footer {
    background:#0d1117; border:1px solid #30363d;
    border-top:2px solid #58a6ff; border-radius:0 0 8px 8px;
    padding:0.75rem 1.1rem; font-size:0.8rem; color:#8b949e;
    margin-top:0.8rem; line-height:1.8;
}

/* ── Direct Clip ────────────────────────────────────────────────────────── */
.direct-clip {
    background:#0d1117; border-left:3px solid #58a6ff;
    padding:0.75rem 0.9rem; border-radius:0 6px 6px 0;
    font-size:0.82rem; color:#8b949e; font-style:italic;
    margin:0.6rem 0; max-height:130px; overflow-y:auto; line-height:1.55;
}

/* ── Test-Case Pills ────────────────────────────────────────────────────── */
.testcase-pill {
    display:inline-block; background:#161b22; border:1px solid #30363d;
    border-radius:20px; padding:4px 14px; font-size:0.8rem; color:#8b949e;
    cursor:pointer; margin:3px;
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


# ─── Hackathon Test Cases ──────────────────────────────────────────────────────
st.markdown(
    '<span class="section-label">⚡ Quick Verify — Ophthalmology Test Cases</span>',
    unsafe_allow_html=True,
)
tc1, tc2, tc3 = st.columns(3)
_TC = {
    "IOP 32 mmHg": "What is the treatment protocol for a patient with Intraocular Pressure (IOP) of 32 mmHg?",
    "Sugar for eye surgery": "What is the acceptable blood sugar level (FBS and RBS) for a patient undergoing eye surgery?",
    "Post-op steroid schedule": "What is the recommended post-operative steroid eye drop schedule after cataract surgery?",
}
with tc1:
    if st.button("👁 IOP 32 mmHg", use_container_width=True):
        st.session_state["prefill_query"] = _TC["IOP 32 mmHg"]
        st.rerun()
with tc2:
    if st.button("🩸 Sugar for eye surgery", use_container_width=True):
        st.session_state["prefill_query"] = _TC["Sugar for eye surgery"]
        st.rerun()
with tc3:
    if st.button("💊 Post-op steroid schedule", use_container_width=True):
        st.session_state["prefill_query"] = _TC["Post-op steroid schedule"]
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# ─── Query Form ───────────────────────────────────────────────────────────────
_prefill = st.session_state.pop("prefill_query", "")

with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Clinical Query",
        value=_prefill,
        placeholder=(
            "e.g. What is the first-line treatment for open-angle glaucoma "
            "with IOP > 30 mmHg per ICMR guidelines?"
        ),
        height=110,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("🔍  Verify & Analyse", use_container_width=True)


# ─── Validation ───────────────────────────────────────────────────────────────
if submitted and not query.strip():
    st.warning("⚠️ Please enter a clinical query before submitting.")
    st.stop()


# ─── V-RAG Pipeline ───────────────────────────────────────────────────────────
if submitted and query.strip():

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
        st.write(f"🤖 Synthesising clinical answer via Ollama ({MODEL_NAME})…")
        synthesizer = ClinicalSynthesizer()
        answer_raw  = synthesizer.generate_answer(query, results, conflict)

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

        # Step 5 — Dosage Safety Check on best chunk
        dosage_report = None
        if verdicts:
            best_chunk = max(verdicts, key=lambda v: v["confidence"])
            dosage_report = pairwise.__class__(
            ).detect_dosage_contradiction.__func__(
                pairwise.__class__(), best_chunk["chunk_text"], answer_body
            ) if False else None  # Re-instantiate cheaply below
            _tmp_auditor = PairwiseAuditor()
            dosage_report = _tmp_auditor.detect_dosage_contradiction(
                best_chunk["chunk_text"], answer_body
            )
            _tmp_auditor.unload()

        pipe_status.update(label="✅ Pipeline Complete", state="complete", expanded=False)

    # ── Source-Aware Logic Badge ───────────────────────────────────────────────
    st.markdown(_source_badge(top_verdict, verdicts), unsafe_allow_html=True)

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
        with st.container(border=True):
            st.markdown(answer_body)

        # ── Citation Footer ────────────────────────────────────────────────────
        ts = (verdicts[0]["timestamp"] if verdicts
              else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cit_rows = []
        for i, chunk in enumerate(results, 1):
            m = chunk.get("metadata", {})
            proto = m.get("Protocol_Name") or _protocol_label(m.get("filename", "?"))
            cit_rows.append(
                f"<b>[{i}]</b>&nbsp;{proto}&nbsp;·&nbsp;"
                f"p.{m.get('Page_Number', m.get('page', '?'))}&nbsp;·&nbsp;"
                f"{m.get('doc_year', m.get('statute_year', '?'))}"
            )
        st.markdown(
            '<div class="citation-footer">'
            '<span style="color:#58a6ff;font-weight:700;">📎 Source Metadata</span>'
            '<br>' + "&nbsp;&nbsp;│&nbsp;&nbsp;".join(cit_rows) +
            f'<br><span style="color:#6e7681;font-size:0.75rem;">'
            f'⏱ Verified at {ts}&nbsp;·&nbsp;Specialty: {active_specialty}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT PANE (40%) — Evidence Vault ─────────────────────────────────────
    with right_col:
        st.markdown(
            '<span class="section-label-blue">🔐 Evidence Vault</span>',
            unsafe_allow_html=True,
        )

        # Overall Confidence Meter
        _conf_bar("Overall NLI Confidence", top_verdict["confidence"], top_verdict["color"])
        st.markdown("---")

        # Per-Chunk Evidence Cards
        for i, v in enumerate(verdicts, 1):
            m     = v["metadata"]
            clip  = v["chunk_text"][:420].replace("\n", " ")
            fname = m.get("filename", "Unknown")
            pg    = m.get("Page_Number", m.get("page", "?"))
            proto = m.get("Protocol_Name") or _protocol_label(fname)

            with st.expander(
                f"Evidence #{i} — {proto}  ·  p.{pg}",
                expanded=(i == 1),
            ):
                # NLI Status Chip
                st.markdown(
                    f'{_nli_chip(v["status"], v["emoji"])}'
                    f'&nbsp;&nbsp;<span style="font-size:0.8rem;color:#8b949e;">'
                    f'Confidence:&nbsp;<b style="color:#c9d1d9;">{v["confidence"]}%</b></span>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # 3-way NLI probability breakdown
                color_map = {"ENTAILED": "green", "NEUTRAL": "orange", "CONTRADICTION": "red"}
                for lbl, val in sorted(v["breakdown"].items(), key=lambda x: x[1], reverse=True):
                    _conf_bar(lbl, val, color_map.get(lbl, "green"))

                # ── Direct PDF Clip (Visual Evidence) ─────────────────────────
                try:
                    pg_int     = int(pg) if str(pg).isdigit() else 1
                    clip_bytes = extract_pdf_clip(fname, pg_int, v["chunk_text"][:150])
                    if clip_bytes:
                        st.markdown(
                            '<span class="section-label" style="margin-top:0.6rem;">'
                            '🖼 Direct PDF Clip</span>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            clip_bytes,
                            caption=f"{proto}  ·  p.{pg}",
                            use_container_width=True,
                        )
                    else:
                        # Fallback: text clip
                        st.markdown(
                            f'<div class="direct-clip">"{clip}…"</div>',
                            unsafe_allow_html=True,
                        )
                except Exception:
                    st.markdown(
                        f'<div class="direct-clip">"{clip}…"</div>',
                        unsafe_allow_html=True,
                    )

                # Source Metadata footer
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#6e7681;margin-top:0.5rem;'
                    f'border-top:1px solid #30363d;padding-top:0.4rem;">'
                    f'📄&nbsp;{fname}&nbsp;·&nbsp;Page&nbsp;{pg}&nbsp;·&nbsp;'
                    f'Protocol:&nbsp;{proto}&nbsp;·&nbsp;'
                    f'Published:&nbsp;{m.get("doc_year", m.get("statute_year","?"))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
