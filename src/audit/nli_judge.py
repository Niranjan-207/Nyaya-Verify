"""
src/audit/nli_judge.py — PairwiseAuditor
=========================================
Clinical Safety Logic Engine.

Replaced temporal conflict logic (old law vs new law) with
Clinical Safety Logic: cross-protocol guideline adherence vs contradiction.

Resolution Priority
-------------------
  1. Authority Rank  — ICMR(5) > AIIMS(4) > AIOS(3) = AAO(3) > NLEM(2) > Generic(1)
  2. Recency tiebreaker — higher doc_year wins when ranks are equal

Dosage Contradiction Detection
-------------------------------
  detect_dosage_contradiction(premise, hypothesis) checks whether numeric
  dosage values in the AI hypothesis exist in the source chunk (premise).
  Flags as CONTRADICTION if hypothesis introduces dosages absent from the PDF.

NLI Label Index (cross-encoder/nli-deberta-v3-small)
------------------------------------------------------
  0 → CONTRADICTION  |  1 → ENTAILED  |  2 → NEUTRAL
"""

import re
import gc
import torch
from typing import List, Dict, Any, Set
from sentence_transformers.cross_encoder import CrossEncoder

# ---------------------------------------------------------------------------
# Clinical authority ranking
# ---------------------------------------------------------------------------
_AUTHORITY_RANK: Dict[str, int] = {
    "icmr":  5,
    "aiims": 4,
    "aios":  3,
    "aao":   3,
    "nlem":  2,
}

# Dosage pattern — matches clinical numeric values with units
_DOSAGE_RE = re.compile(
    r'\b\d+(?:\.\d+)?\s*'
    r'(?:mg|ml|mcg|μg|ug|g|%|mmhg|mm\s*hg|iu|unit|units|'
    r'drop|drops|tablet|tab|cap|capsule|vial|patch|sachet)\b',
    re.IGNORECASE,
)


def _authority_score(filename: str) -> int:
    """Return the clinical authority rank for a given filename."""
    fn = filename.lower()
    for key, rank in sorted(_AUTHORITY_RANK.items(), key=lambda x: -x[1]):
        if key in fn:
            return rank
    return 1


def _doc_year_int(meta: Dict[str, Any]) -> int:
    """Parse doc_year (or statute_year alias) to int for comparison."""
    raw = meta.get("doc_year") or meta.get("statute_year") or 0
    try:
        return int(raw)
    except (ValueError, TypeError):
        return 0


def _extract_dosages(text: str) -> Set[str]:
    """
    Extract all normalised dosage strings from text.
    Returns a set of lowercase tokens like {"500mg", "1%", "32 mmhg"}.
    """
    return {m.group().lower().replace(" ", "") for m in _DOSAGE_RE.finditer(text)}


# ---------------------------------------------------------------------------
# PairwiseAuditor
# ---------------------------------------------------------------------------
class PairwiseAuditor:
    """
    Cross-chunk Clinical Contradiction Detector.

    Uses DeBERTa-v3-small NLI CrossEncoder to detect logical contradictions
    between retrieved chunks from *different* clinical protocols/sources.

    Conflict Resolution
    -------------------
    When contradiction_prob > 0.80 the two chunks are resolved by:
      1. Authority rank (ICMR > AIIMS > AIOS/AAO > NLEM > generic)
      2. Publication recency (doc_year tiebreaker)

    Payload keys
    ------------
    ConflictFound         : bool
    PrioritizedChunk      : dict  (winning chunk with higher authority)
    RejectedChunk         : dict  (lower-authority or older chunk)
    ContradictionScore    : float (0–1)
    AuthorityA / B        : int   (scores of the two chunks)
    ProtocolA / B         : str   (filenames for display)
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self.model = CrossEncoder(model_name)

    # ------------------------------------------------------------------
    # Public: cross-chunk contradiction
    # ------------------------------------------------------------------
    def detect_logic_flips(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieved chunks for clinical contradictions.
        Only compares chunks from *different* source protocols to avoid
        trivial intra-document comparisons and O(N²) latency spikes.

        Constrained to top-5 chunks.
        """
        chunks = chunks[:5]
        pairs, chunk_pairs = [], []

        for idx1, c1 in enumerate(chunks):
            for c2 in chunks[idx1 + 1:]:
                m1 = c1.get("metadata", {})
                m2 = c2.get("metadata", {})

                # Skip if same protocol file AND same year — intra-document
                same_file = m1.get("filename") == m2.get("filename")
                same_year = _doc_year_int(m1) == _doc_year_int(m2)
                if same_file and same_year:
                    continue

                pairs.append((c1["text"], c2["text"]))
                chunk_pairs.append((c1, c2))

        if not pairs:
            return {"ConflictFound": False}

        scores = self.model.predict(pairs)
        probs  = torch.nn.functional.softmax(
            torch.tensor(scores), dim=1
        ).numpy()

        for i, prob_mapping in enumerate(probs):
            # Index 0 = contradiction (see module docstring)
            contradiction_prob = float(prob_mapping[0])
            if contradiction_prob > 0.80:
                c1, c2 = chunk_pairs[i]
                m1, m2 = c1.get("metadata", {}), c2.get("metadata", {})

                auth1 = _authority_score(m1.get("filename", ""))
                auth2 = _authority_score(m2.get("filename", ""))
                year1 = _doc_year_int(m1)
                year2 = _doc_year_int(m2)

                # Resolution: authority first, recency as tiebreaker
                if auth1 > auth2 or (auth1 == auth2 and year1 >= year2):
                    prioritized, rejected = c1, c2
                else:
                    prioritized, rejected = c2, c1

                return {
                    "ConflictFound":     True,
                    "PrioritizedChunk":  prioritized,
                    "RejectedChunk":     rejected,
                    "ContradictionScore": contradiction_prob,
                    "AuthorityA":        auth1,
                    "AuthorityB":        auth2,
                    "ProtocolA":         m1.get("filename", "Unknown"),
                    "ProtocolB":         m2.get("filename", "Unknown"),
                }

        return {"ConflictFound": False}

    # ------------------------------------------------------------------
    # Public: dosage-specific contradiction
    # ------------------------------------------------------------------
    def detect_dosage_contradiction(
        self, premise: str, hypothesis: str
    ) -> Dict[str, Any]:
        """
        Clinical Dosage Safety Check.

        Extracts all dosage values from the AI hypothesis and checks
        each against the source PDF chunk (premise).

        Returns
        -------
        {
            "DosageConflict":    bool,
            "HallucDosages":     list[str]  — dosages in answer NOT in PDF,
            "VerifiedDosages":   list[str]  — dosages confirmed in PDF,
            "PremiseDosages":    list[str]  — all dosages found in the PDF chunk,
        }
        """
        premise_doses    = _extract_dosages(premise)
        hypothesis_doses = _extract_dosages(hypothesis)

        verified   = hypothesis_doses & premise_doses
        halluc     = hypothesis_doses - premise_doses

        return {
            "DosageConflict":  len(halluc) > 0,
            "HallucDosages":   sorted(halluc),
            "VerifiedDosages": sorted(verified),
            "PremiseDosages":  sorted(premise_doses),
        }

    # ------------------------------------------------------------------
    # VRAM release
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """
        Wipes the CrossEncoder and rebalances VRAM for Ollama / ClinicalAuditor.
        """
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
