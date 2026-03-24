"""
src/retrieval/vector_store.py — ChromaDataStore
================================================
Semantic retrieval with Dynamic K-Scaling for clinical complexity.

Clinical K-Scaling Rules
------------------------
  Base K = 5
  +2  per comorbidity term detected   (glaucoma+diabetes, cataract+hypertension …)
  +2  if ≥2 distinct protocols named  (icmr, aiims, aios, aao, nlem)
  +2  if query is a comparison        (versus, compare, difference, vs)
  +1  per additional clinical flag    (dosage, iop, mmhg, contraindication…)
  MAX = 14

Metadata Fields (normalised on ingest)
---------------------------------------
  filename        : str   — original PDF filename
  page            : int   — 1-indexed page number
  Page_Number     : int   — alias for page (for frontend compatibility)
  doc_year        : str   — publication year (canonical, str for ChromaDB)
  Protocol_Name   : str   — inferred from filename (ICMR, AIIMS, AAO …)
  Last_Updated_Date : str — same as doc_year until we have richer metadata
"""

import os
import re
from typing import List, Dict, Any, Optional

import chromadb
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Clinical K-scaling vocabulary
# ---------------------------------------------------------------------------
_COMORBIDITY_TERMS = {
    "comorbid", "comorbidity", "diabetes", "diabetic", "glaucoma",
    "hypertension", "hypertensive", "cataract", "retinopathy",
    "neuropathy", "cardiovascular", "renal", "renal failure",
    "combined", "simultaneous", "both", "and",
}
_MULTI_PROTOCOL_TERMS = {
    "icmr", "aiims", "aios", "aao", "nlem", "protocol", "guideline",
    "standard", "stg", "ppp", "compare", "versus", "vs", "difference",
}
_CLINICAL_DETAIL_TERMS = {
    "dosage", "dose", "mg", "ml", "iop", "mmhg", "contraindication",
    "side effect", "adverse", "schedule", "frequency", "taper",
    "post-op", "post op", "intraocular", "topical", "systemic",
}

# ---------------------------------------------------------------------------
# Protocol label inference
# ---------------------------------------------------------------------------
_PROTOCOL_PATTERNS: List[tuple] = [
    ("icmr",  "ICMR"),
    ("aiims", "AIIMS"),
    ("aios",  "AIOS"),
    ("aao",   "AAO PPP 2025"),
    ("nlem",  "NLEM 2022"),
]


def _infer_protocol_name(filename: str) -> str:
    """Map a PDF filename to a human-readable protocol label."""
    fn = filename.lower()
    for key, label in _PROTOCOL_PATTERNS:
        if key in fn:
            return label
    # Fall back to capitalised filename stem
    return os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# ChromaDataStore
# ---------------------------------------------------------------------------
class ChromaDataStore:
    """
    Persistent ChromaDB vector store with BGE-small-en-v1.5 embeddings.

    Parameters
    ----------
    persist_directory : str
        Path to the ChromaDB storage folder.
    collection_name : str
        ChromaDB collection identifier.
    model_name : str
        SentenceTransformer model for embedding generation.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str   = "nyaya_legal",
        model_name: str        = "BAAI/bge-small-en-v1.5",
    ) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self.client     = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] ChromaDataStore embedding device: {device}")
        self.model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Metadata normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_metadata(m: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all required metadata keys are present and correctly typed.
        ChromaDB requires all metadata values to be str / int / float / bool.
        """
        # Canonical year key — resolve legacy alias
        year = m.get("doc_year") or m.get("statute_year") or "Unknown"
        m["doc_year"]          = str(year)
        m["Last_Updated_Date"] = str(year)

        # Protocol label
        m.setdefault("Protocol_Name", _infer_protocol_name(m.get("filename", "")))

        # Page aliases — store both for backward compatibility
        page = m.get("page", 0)
        m["page"]        = int(page) if str(page).isdigit() else 0
        m["Page_Number"] = m["page"]

        # Drop None values — ChromaDB cannot serialise them
        return {k: v for k, v in m.items() if v is not None}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    def ingest(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embed and upsert chunks into ChromaDB.

        Returns
        -------
        int : number of chunks stored.
        """
        if not chunks:
            return 0

        texts     = [c["text"] for c in chunks]
        metadatas = [self._normalise_metadata(dict(c["metadata"])) for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        ids = [
            f'{m["filename"]}_{m["page"]}_{i}'
            for i, m in enumerate(metadatas)
        ]

        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        return len(texts)

    # ------------------------------------------------------------------
    # Dynamic K-scaling
    # ------------------------------------------------------------------
    @staticmethod
    def _dynamic_k(query: str) -> int:
        """
        Compute retrieval depth K based on clinical complexity signals.

        Scoring
        -------
        Base K = 5
        +2  for each comorbidity keyword found
        +2  if ≥2 protocol names mentioned
        +2  for explicit comparison language
        +1  for each clinical detail keyword
        MAX = 14
        """
        q = query.lower()
        words = set(re.findall(r'\b\w+\b', q))

        k = 5

        # Comorbidity boost — "glaucoma and diabetes", "diabetic cataract"
        comorbidity_hits = len(_COMORBIDITY_TERMS & words)
        if comorbidity_hits >= 2:
            k += 2

        # Multi-protocol boost — query mentions ≥2 authority names
        protocol_hits = sum(1 for p in ("icmr", "aiims", "aios", "aao", "nlem") if p in q)
        if protocol_hits >= 2:
            k += 2

        # Comparison boost
        if any(t in q for t in ("versus", "compare", "difference", " vs ")):
            k += 2

        # Clinical detail boost
        detail_hits = sum(1 for t in _CLINICAL_DETAIL_TERMS if t in q)
        k += min(detail_hits, 3)   # cap at +3

        k = min(k, 14)
        print(f"[*] Clinical dynamic K selected: {k}  (query: {query[:60]}…)")
        return k

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        specialty: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search with dynamic K.

        Parameters
        ----------
        query : str
            The clinical question.
        k : int, optional
            Override the auto-computed K.
        specialty : str, optional
            If supplied (e.g. "Glaucoma"), boost K by 2 to ensure
            specialty-specific chunks are retrieved.

        Returns
        -------
        list of dicts with keys "text" and "metadata".
        """
        effective_k = k if k is not None else self._dynamic_k(query)

        if specialty:
            effective_k = min(effective_k + 2, 14)

        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=effective_k,
        )

        chunks: List[Dict[str, Any]] = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                chunks.append({"text": doc, "metadata": meta})

        return chunks
