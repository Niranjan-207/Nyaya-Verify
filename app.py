"""
app.py — Medify  |  Clinical Intelligence Dashboard
=====================================================
UI/UX: Dark-themed dual-pane clinical dashboard.
  Left  (60%): Verified Clinical Answer + Citation Footer
  Right (40%): Evidence Vault — NLI status chips, confidence bars,
               direct text clips, per-source metadata

Pipeline:
  1. ChromaDataStore  → dynamic-K semantic retrieval
  2. PairwiseAuditor  → cross-chunk contradiction detection (unloaded after)
  3. ClinicalSynthesizer → Ollama llama3.1:8b answer generation
  4. ClinicalAuditor  → DeBERTa-v3-large answer NLI verification (unloaded after)
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
from src.retrieval.vector_store import ChromaDataStore
from src.audit.nli_judge import PairwiseAuditor
from src.generation.llm_interface import ClinicalSynthesizer
from src.verifier import ClinicalAuditor
from src.ingestion.pdf_parser import extract_text_with_metadata
from src.ingestion.semantic_chunker import HybridHierarchicalChunker

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medify | Clinical Intelligence Dashboard",
    page_icon="🏥",
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
[data-testid="stHeader"] {
    background-color: #0d1117 !important;
}

/* ── Truth Badge ────────────────────────────────────────────────────────── */
.badge-entailed {
    background: #1a4731;
    border: 2px solid #238636;
    color: #3fb950;
    border-radius: 8px;
    padding: 0.85rem 1.5rem;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-align: center;
    margin-bottom: 1.4rem;
    display: block;
}
.badge-contradiction {
    background: #3d1515;
    border: 2px solid #da3633;
    color: #f85149;
    border-radius: 8px;
    padding: 0.85rem 1.5rem;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-align: center;
    margin-bottom: 1.4rem;
    display: block;
}
.badge-neutral {
    background: #2d2200;
    border: 2px solid #9e6a03;
    color: #d29922;
    border-radius: 8px;
    padding: 0.85rem 1.5rem;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-align: center;
    margin-bottom: 1.4rem;
    display: block;
}

/* ── NLI Status Chips ───────────────────────────────────────────────────── */
.chip-entailed {
    background: #1a4731; color: #3fb950;
    border: 1px solid #238636;
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    display: inline-block;
}
.chip-contradiction {
    background: #3d1515; color: #f85149;
    border: 1px solid #da3633;
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    display: inline-block;
}
.chip-neutral {
    background: #2d2200; color: #d29922;
    border: 1px solid #9e6a03;
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    display: inline-block;
}

/* ── Citation Footer ────────────────────────────────────────────────────── */
.citation-footer {
    background: #0d1117;
    border: 1px solid #30363d;
    border-top: 2px solid #58a6ff;
    border-radius: 0 0 8px 8px;
    padding: 0.75rem 1.1rem;
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 0.8rem;
    line-height: 1.8;
}

/* ── Direct Clip ────────────────────────────────────────────────────────── */
.direct-clip {
    background: #0d1117;
    border-left: 3px solid #58a6ff;
    padding: 0.75rem 0.9rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #8b949e;
    font-style: italic;
    margin: 0.6rem 0;
    max-height: 130px;
    overflow-y: auto;
    line-height: 1.55;
}

/* ── Section Labels ─────────────────────────────────────────────────────── */
.section-label {
    color: #8b949e;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    display: block;
}
.section-label-blue {
    color: #58a6ff;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    display: block;
}

/* ── Health Card ────────────────────────────────────────────────────────── */
.health-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.8rem;
    font-size: 0.82rem;
}
.health-card table { width: 100%; border-collapse: collapse; }
.health-card td    { padding: 4px 0; }
.dot-on  { color: #3fb950; }
.dot-off { color: #f85149; }

/* ── Conf Label ─────────────────────────────────────────────────────────── */
.conf-label {
    font-size: 0.75rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1px;
}

/* ── File Uploader ──────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 2px dashed #30363d !important;
    border-radius: 8px !important;
}

/* ── Text Area ──────────────────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea {
    background-color: #161b22 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}

/* ── Buttons ────────────────────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stButton"] > button {
    background-color: #238636 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stButton"] > button:hover {
    background-color: #2ea043 !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Horizontal Rule ────────────────────────────────────────────────────── */
hr { border-color: #30363d !important; }

/* ── Progress Bar Track ─────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background-color: #30363d !important;
}

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _vram_stats():
    """Returns (used_mb, total_mb). Falls back to (0, 0) when no CUDA."""
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated(0) / 1024 ** 2
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        return used, total
    return 0.0, 0.0


def _truth_badge(verdict: dict) -> str:
    css = {
        "ENTAILED":      "badge-entailed",
        "CONTRADICTION": "badge-contradiction",
        "NEUTRAL":       "badge-neutral",
    }.get(verdict["status"], "badge-neutral")
    return (
        f'<div class="{css}">'
        f'{verdict["emoji"]}&nbsp;&nbsp;{verdict["label"]}'
        f'<span style="font-size:0.88rem;font-weight:400;margin-left:1.2rem;">'
        f'Verification Confidence: <b>{verdict["confidence"]}%</b></span>'
        f'</div>'
    )


def _nli_chip(status: str, emoji: str) -> str:
    css = {
        "ENTAILED":      "chip-entailed",
        "CONTRADICTION": "chip-contradiction",
        "NEUTRAL":       "chip-neutral",
    }.get(status, "chip-neutral")
    return f'<span class="{css}">{emoji}&nbsp;{status}</span>'


def _conf_bar(label: str, value: float, color: str) -> None:
    """Renders a labelled percentage progress bar."""
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
        'margin-bottom:0.1rem;">🏥 Medify</h2>'
        '<p style="color:#8b949e;font-size:0.78rem;margin-top:0;">'
        'Clinical Intelligence Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── System Health ──────────────────────────────────────────────────────────
    st.markdown(
        '<span class="section-label">⚙ System Health</span>',
        unsafe_allow_html=True,
    )
    used_mb, total_mb = _vram_stats()
    vram_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0.0
    gpu_ok   = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_properties(0).name if gpu_ok else "CPU only"
    )

    st.markdown(
        f"""
        <div class="health-card">
          <table>
            <tr>
              <td style="color:#8b949e;">NLI Engine</td>
              <td style="text-align:right;">
                <span class="dot-on">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">
                &nbsp;DeBERTa-v3-large</span>
              </td>
            </tr>
            <tr>
              <td style="color:#8b949e;">LLM Engine</td>
              <td style="text-align:right;">
                <span class="dot-on">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">
                &nbsp;Ollama {MODEL_NAME}</span>
              </td>
            </tr>
            <tr>
              <td style="color:#8b949e;">Vector Store</td>
              <td style="text-align:right;">
                <span class="dot-on">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">
                &nbsp;medify_protocols</span>
              </td>
            </tr>
            <tr>
              <td style="color:#8b949e;">GPU</td>
              <td style="text-align:right;">
                <span class="{'dot-on' if gpu_ok else 'dot-off'}">●</span>
                <span style="color:#c9d1d9;font-size:0.8rem;">
                &nbsp;{gpu_name}</span>
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

    # ── Active Collection ──────────────────────────────────────────────────────
    active_doc = st.session_state.get("active_doc", "medify_protocols (53 documents)")
    st.markdown(
        f'<div style="font-size:0.78rem;color:#8b949e;margin:0.5rem 0 0.8rem;">'
        f'📂 Active:&nbsp;<span style="color:#58a6ff;">{active_doc}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="margin:0.6rem 0;">', unsafe_allow_html=True)

    # ── PDF Ingestion ──────────────────────────────────────────────────────────
    st.markdown(
        '<span class="section-label">📥 Ingest New Protocol</span>',
        unsafe_allow_html=True,
    )
    uploaded_pdf = st.file_uploader(
        "Drop a clinical protocol PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )
    year_str = st.text_input(
        "Publication year (optional)",
        placeholder="e.g. 2024",
        label_visibility="visible",
    )

    if uploaded_pdf is not None and st.button("⚡ Ingest Document"):
        with st.spinner(f"Ingesting `{uploaded_pdf.name}`…"):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    tmp_path = tmp.name

                year = (
                    int(year_str.strip())
                    if year_str.strip().isdigit()
                    else None
                )
                chunker = HybridHierarchicalChunker()
                store   = ChromaDataStore()

                blocks = extract_text_with_metadata(tmp_path, doc_year=year)
                chunks = chunker.chunk(blocks)
                n      = store.ingest(chunks)

                os.unlink(tmp_path)
                st.session_state["active_doc"] = uploaded_pdf.name
                st.success(
                    f"✅ {n} chunks ingested from `{uploaded_pdf.name}`"
                )
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.markdown('<hr style="margin:0.8rem 0;">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.72rem;color:#8b949e;line-height:1.6;">'
        '🔒 <b>100% Local Processing</b><br>'
        'No data leaves this machine.<br>'
        'PHI-safe · Air-gapped inference.</div>',
        unsafe_allow_html=True,
    )


# ─── Main Header ──────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#e6edf3;font-weight:700;font-size:1.75rem;'
    'margin-bottom:0.15rem;">Clinical Intelligence Query Interface</h1>'
    '<p style="color:#8b949e;margin-top:0;font-size:0.88rem;">'
    'NLI Verified by&nbsp;<b>DeBERTa-v3-large</b>&nbsp;·&nbsp;'
    'Generated by&nbsp;<b>Llama 3.1 8B</b>&nbsp;·&nbsp;'
    'Sourced from&nbsp;<b>Medical Protocol Database</b></p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ─── Query Form ───────────────────────────────────────────────────────────────
with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "Clinical Query",
        placeholder=(
            "e.g. What is the first-line treatment and recommended dosage "
            "of Metformin for Type 2 Diabetes in adults?"
        ),
        height=110,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "🔍  Verify & Analyse", use_container_width=True
    )


# ─── Validation ───────────────────────────────────────────────────────────────
if submitted and not query.strip():
    st.warning("⚠️ Please enter a clinical query before submitting.")
    st.stop()


# ─── V-RAG Pipeline ───────────────────────────────────────────────────────────
if submitted and query.strip():

    with st.status("🔄 Running V-RAG Pipeline…", expanded=True) as pipe_status:

        # Step 1 — Semantic Retrieval
        st.write("📡 Retrieving semantic fragments from Vector Store…")
        store   = ChromaDataStore()
        results = store.search(query)

        if not results:
            pipe_status.update(
                label="❌ No context found", state="error", expanded=True
            )
            st.error(
                "Context undefined. "
                "Run `python scripts/ingest_all.py` before querying."
            )
            st.stop()

        # Step 2 — Pairwise Chunk Contradiction Check
        st.write("🔬 Pairwise NLI logic-flip detection across chunks…")
        pairwise    = PairwiseAuditor()
        conflict    = pairwise.detect_logic_flips(results)
        pairwise.unload()

        # Step 3 — Answer Generation
        st.write(f"🤖 Synthesising answer via Ollama ({MODEL_NAME})…")
        synthesizer = ClinicalSynthesizer()
        answer_raw  = synthesizer.generate_answer(query, results, conflict)

        # Strip the embedded conflict header so the answer pane is clean
        answer_body = answer_raw
        if conflict.get("ConflictFound") and answer_raw.lstrip().startswith("⚠️"):
            parts = answer_raw.split("\n\n", 1)
            answer_body = parts[1] if len(parts) > 1 else answer_raw

        # Step 4 — NLI Answer Verification
        st.write("🧠 Verifying answer with DeBERTa-v3-large NLI…")
        verifier    = ClinicalAuditor()
        verdicts    = verifier.verify_answer(answer_body, results)
        top_verdict = verifier.aggregate_verdict(verdicts)
        verifier.unload()

        pipe_status.update(
            label="✅ Pipeline Complete", state="complete", expanded=False
        )

    # ── Truth Badge ────────────────────────────────────────────────────────────
    st.markdown(_truth_badge(top_verdict), unsafe_allow_html=True)

    # ── Logic-Flip Banner (only when cross-source contradiction found) ─────────
    if conflict.get("ConflictFound"):
        p_meta = conflict.get("PrioritizedChunk", {}).get("metadata", {})
        r_meta = conflict.get("RejectedChunk",    {}).get("metadata", {})
        st.warning(
            f"⚠️ **CLINICAL CONFLICT DETECTED** — Contradiction between "
            f"`{r_meta.get('filename', 'Unknown')}` and "
            f"`{p_meta.get('filename', 'Unknown')}`. "
            f"Prioritising **{p_meta.get('doc_year', p_meta.get('statute_year', '?'))}** protocol."
        )

    st.markdown("---")

    # ── Dual-Pane Layout ───────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    # ── LEFT PANE (60%) — Verified Legal Answer ────────────────────────────────
    with left_col:
        st.markdown(
            '<span class="section-label">📋 Verified Clinical Answer</span>',
            unsafe_allow_html=True,
        )

        # Render the LLM answer as Markdown (handles bold dosages, headers, etc.)
        with st.container(border=True):
            st.markdown(answer_body)

        # ── Citation Footer ────────────────────────────────────────────────────
        ts       = verdicts[0]["timestamp"] if verdicts else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cit_rows = []
        for i, chunk in enumerate(results, 1):
            m = chunk.get("metadata", {})
            cit_rows.append(
                f"<b>[{i}]</b>&nbsp;{m.get('filename','?')}&nbsp;·&nbsp;"
                f"Page&nbsp;{m.get('page','?')}&nbsp;·&nbsp;"
                f"Year&nbsp;{m.get('doc_year', m.get('statute_year','?'))}"
            )
        st.markdown(
            '<div class="citation-footer">'
            '<span style="color:#58a6ff;font-weight:700;">📎 Source Metadata</span>'
            '<br>' + "&nbsp;&nbsp;│&nbsp;&nbsp;".join(cit_rows) +
            f'<br><span style="color:#6e7681;font-size:0.75rem;">'
            f'⏱ Verified at {ts}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT PANE (40%) — Evidence Vault ─────────────────────────────────────
    with right_col:
        st.markdown(
            '<span class="section-label-blue">🔐 Evidence Vault</span>',
            unsafe_allow_html=True,
        )

        # ── Overall Confidence Meter ───────────────────────────────────────────
        _conf_bar(
            "Overall NLI Confidence",
            top_verdict["confidence"],
            top_verdict["color"],
        )
        st.markdown("---")

        # ── Per-Chunk Evidence Cards ───────────────────────────────────────────
        for i, v in enumerate(verdicts, 1):
            m    = v["metadata"]
            clip = v["chunk_text"][:420].replace("\n", " ")
            fname = m.get("filename", "Unknown")
            pg    = m.get("page", "?")

            with st.expander(
                f"Evidence #{i} — {fname}  ·  p.{pg}",
                expanded=(i == 1),
            ):
                # NLI Status Chip + confidence inline
                st.markdown(
                    f'{_nli_chip(v["status"], v["emoji"])}'
                    f'&nbsp;&nbsp;<span style="font-size:0.8rem;color:#8b949e;">'
                    f'Confidence:&nbsp;<b style="color:#c9d1d9;">'
                    f'{v["confidence"]}%</b></span>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # 3-way probability breakdown
                color_map = {
                    "ENTAILED":      "green",
                    "NEUTRAL":       "orange",
                    "CONTRADICTION": "red",
                }
                for lbl, val in sorted(
                    v["breakdown"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    _conf_bar(lbl, val, color_map.get(lbl, "green"))

                # Direct Clip
                st.markdown(
                    f'<div class="direct-clip">"{clip}…"</div>',
                    unsafe_allow_html=True,
                )

                # Source Metadata footer
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#6e7681;'
                    f'margin-top:0.5rem;border-top:1px solid #30363d;'
                    f'padding-top:0.4rem;">'
                    f'📄&nbsp;{fname}&nbsp;·&nbsp;'
                    f'Page&nbsp;{pg}&nbsp;·&nbsp;'
                    f'Published:&nbsp;{m.get("doc_year", m.get("statute_year","?"))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
