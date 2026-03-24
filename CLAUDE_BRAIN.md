# 🧠 Claude Brain — Nyaya-Verify Project Memory

> Auto-maintained by Claude. Updated after every session.
> Do **not** delete — this is Claude's persistent memory across context windows.

---

## 📌 Project Identity

| Field | Value |
|---|---|
| **Project Name** | Nyaya-Verify |
| **Domain** | Indian Legal V-RAG (Verification-Retrieval-Augmented Generation) |
| **Root Path** | `D:\Nyaya-Verify\` |
| **Python Version** | 3.12 |
| **Venv Path** | `D:\Nyaya-Verify\.venv\` |
| **Hardware** | NVIDIA RTX 4050 · 6 GB VRAM · CUDA 12.4 |

---

## 🏗️ Architecture Overview

```
PDF Input → PDFParser → HybridHierarchicalChunker → ChromaDB
                                                        ↓
Query → ChromaDataStore.search() → PairwiseAuditor (chunk-vs-chunk NLI)
                                        ↓
                              LegalSynthesizer (Ollama llama3.1:8b)
                                        ↓
                              ClinicalAuditor (answer-vs-chunk NLI)
                                        ↓
                              Streamlit Clinical Dashboard
```

### Pipeline Stages
1. **Dynamic K Retrieval** — `ChromaDataStore.search()` scales K (5–12) by query complexity
2. **Pairwise Contradiction Check** — `PairwiseAuditor` (DeBERTa-v3-small cross-encoder) compares top-5 chunks cross-source
3. **VRAM Flush** — `auditor.unload()` clears GPU before LLM
4. **Answer Synthesis** — `LegalSynthesizer` → Ollama `llama3.1:8b`
5. **NLI Verification** — `ClinicalAuditor` (DeBERTa-v3-large) verifies answer vs. each chunk
6. **Dashboard Render** — Streamlit dual-pane: Answer (60%) + Evidence Vault (40%)

---

## 📁 File Map

```
Nyaya-Verify/
├── app.py                          ← Streamlit Clinical Dashboard (REWRITTEN)
├── config.yaml                     ← device: cuda, model_name: llama3.1:8b
├── requirements.txt                ← All deps incl. torch (cu124), transformers
├── CLAUDE_BRAIN.md                 ← This file
├── data/
│   ├── chroma_db/                  ← ChromaDB persistent store
│   └── input_pdfs/                 ← 53 PDFs ingested
├── scripts/
│   ├── ask.py                      ← CLI query tool
│   ├── ingest.py                   ← Single PDF ingestion
│   └── ingest_all.py               ← Batch ingestion (priority order)
└── src/
    ├── audit/nli_judge.py          ← PairwiseAuditor (cross-encoder/nli-deberta-v3-small)
    ├── evaluation/
    │   ├── faithfulness_scorer.py  ← DeBERTa-based faithfulness eval
    │   └── rag_evaluator.py        ← RAGAS benchmark
    ├── generation/llm_interface.py ← LegalSynthesizer (Ollama wrapper)
    ├── ingestion/
    │   ├── pdf_parser.py           ← PyMuPDF extractor
    │   └── semantic_chunker.py     ← HybridHierarchicalChunker (3-stage)
    ├── retrieval/vector_store.py   ← ChromaDataStore (BGE-small-en-v1.5)
    └── verifier.py                 ← ClinicalAuditor (DeBERTa-v3-large) ← NEW
```

---

## 🔑 Key Models

| Model | Purpose | VRAM (fp16) | Location |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | Chunk embeddings | ~130 MB | ChromaDataStore |
| `cross-encoder/nli-deberta-v3-small` | Pairwise chunk contradiction | ~400 MB | PairwiseAuditor |
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli` | Answer NLI verification | ~430 MB (fp16) | ClinicalAuditor |
| `llama3.1:8b` (4-bit via Ollama) | Answer generation | ~4.7 GB | Ollama process |

### VRAM Budget (RTX 4050, 6 GB)
- BGE + DeBERTa-small active during retrieval phase: ~530 MB
- Ollama active during generation: ~4.7 GB
- DeBERTa-large active during verification: ~430 MB (fp16)
- ⚠️ Only one heavy model loaded at a time via `unload()` pattern

---

## 🧩 Module Details

### `src/ingestion/semantic_chunker.py` — HybridHierarchicalChunker
Three-stage chunking pipeline (ALREADY IMPLEMENTED):
1. **Structural Split** — Regex on ALL-CAPS lines, numbered sections (`3.2`), bold-like short lines
2. **Semantic Split** — `SemanticChunker` (langchain_experimental) with `BAAI/bge-small-en-v1.5` when super-chunks > 600 tokens
3. **Smart Overlap** — section_header + first sentence of prior chunk (capped at 300 chars)

### `src/retrieval/vector_store.py` — ChromaDataStore
- Embeddings: `BAAI/bge-small-en-v1.5`
- Dynamic K: 5–12 based on query keywords/acts/years
- Persists to `data/chroma_db/`, collection `nyaya_legal`

### `src/audit/nli_judge.py` — PairwiseAuditor
- Cross-encoder: `cross-encoder/nli-deberta-v3-small`
- Compares top-5 chunks pairwise, skips same source+year
- Contradiction threshold: 0.8
- Recency rule: newer statute year wins

### `src/verifier.py` — ClinicalAuditor (NEW — this session)
- Model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli`
- Premise = retrieved PDF chunk; Hypothesis = AI answer
- 3-tier: ENTAILED / NEUTRAL / CONTRADICTION
- Loaded in fp16 to coexist with Ollama
- Aggregate verdict: any CONTRADICTION > all ENTAILED > NEUTRAL

### `src/generation/llm_interface.py` — LegalSynthesizer
- Ollama client → `llama3.1:8b`
- Injects logic flip alert if PairwiseAuditor finds contradiction
- System prompt enforces no hallucination, citation tags

---

## 🎨 App UI — Clinical Dashboard (REWRITTEN this session)

### Layout
```
┌─ Sidebar (dark) ──────────┐  ┌─ Main ──────────────────────────────────────┐
│ ⚖️ Nyaya-Verify            │  │ Legal Intelligence Query Interface           │
│                           │  │ [Large text area query input]                │
│ ⚙ System Health           │  │ [🔍 Verify & Analyse button]                 │
│  • DeBERTa-v3-large ●     │  ├─────────────────────────────────────────────┤
│  • Ollama llama3.1:8b ●   │  │ [TRUTH BADGE — full width]                   │
│  • nyaya_legal ●          │  ├──────────────── 60% ──┬──────── 40% ─────────┤
│  • CUDA ●                 │  │ 📋 Verified Legal     │ 🔐 Evidence Vault    │
│  VRAM [████░░] xx%        │  │    Answer             │                      │
│                           │  │                       │ Overall Confidence   │
│ 📥 Ingest New Statute     │  │  [Answer markdown]    │ [Progress bar]       │
│  [Drag & Drop PDF]        │  │                       │                      │
│  [Year input]             │  │  [Citation Footer]    │ [Per-chunk cards:    │
│  [⚡ Ingest]              │  │  📎 Source Metadata   │  chip + conf bars +  │
│                           │  │  Filename·Page·Year   │  direct clip text]   │
│ 🔒 100% Local             │  │  ⏱ Verified at HH:MM  │                      │
└───────────────────────────┘  └───────────────────────┴──────────────────────┘
```

### Key CSS Classes
- `.badge-entailed` / `.badge-contradiction` / `.badge-neutral` — truth badge
- `.chip-entailed` / `.chip-contradiction` / `.chip-neutral` — inline NLI chips
- `.answer-pane` — left pane card (green left border)
- `.evidence-vault` — right pane card (blue top border)
- `.direct-clip` — blue-border scrollable text snippet
- `.citation-footer` — blue-top-bordered footer with source metadata

---

## 🛠️ Setup / Run Commands

```powershell
# Prerequisites (one-time)
$env:HF_HOME = "D:\HuggingFace"
$env:OLLAMA_MODELS = "D:\Ollama"

# Create venv with Python 3.12
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install CUDA-enabled PyTorch FIRST
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining deps
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Run ingestion
python scripts/ingest_all.py --pdf-dir data/input_pdfs/

# Launch dashboard
streamlit run app.py
```

---

## 📦 requirements.txt (current)

```
--extra-index-url https://download.pytorch.org/whl/cu124
torch
torchvision
torchaudio
transformers
langchain
langchain-experimental
langchain-huggingface
chromadb
sentence-transformers
pymupdf
ollama
ragas
pyyaml
streamlit
langchain-ollama
```

---

## 🐛 Known Issues / Decisions

| Issue | Resolution |
|---|---|
| `torch` installed as CPU-only | Force reinstall: `pip install torch --index-url .../cu124 --force-reinstall` |
| `langchain_experimental` not found | `pip install langchain-experimental` |
| DeBERTa-v3-large + Ollama VRAM clash | Load DeBERTa in fp16 (~430 MB); call `verifier.unload()` after use |
| `embeddings.position_ids UNEXPECTED` warning | Benign; BGE-small loaded into BertModel arch, positions key mismatch is harmless |
| `st.container(border=True)` requires Streamlit ≥ 1.29 | Included in requirements (latest stable) |

---

## 📊 Performance Metrics (from README)

- Faithfulness Score: **0.83**
- Logic Flip Detection Accuracy: **98%**
- Average Chunks Ingested: ~36–106 per PDF
- Total Corpus: **53 PDFs**, Indian statutes

---

## 🔄 Session Log

| Date | Changes Made |
|---|---|
| 2026-03-24 | Initial project exploration; CUDA torch fix (CPU→cu124); added `langchain-experimental`; implemented 3-stage HybridHierarchicalChunker; dynamic-K vector store |
| 2026-03-24 | **This session**: Created `src/verifier.py` (ClinicalAuditor, DeBERTa-v3-large NLI); rewrote `app.py` as Clinical Dashboard (dual-pane, truth badge, evidence vault, PDF upload); added `transformers` to requirements.txt; created this brain file |

---

*Last updated: 2026-03-24 | Next: commit + PR*
