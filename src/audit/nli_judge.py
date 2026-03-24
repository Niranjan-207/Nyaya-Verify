"""
src/audit/nli_judge.py — PairwiseAuditor + Hard-Stop Safety Engine
===================================================================
Clinical Safety Logic Engine.

Two independent safety layers
------------------------------
  Layer 1 — PairwiseAuditor (NLI-based)
    Cross-chunk contradiction detection using DeBERTa-v3-small CrossEncoder.
    Authority ranking: ICMR(5) > AIIMS(4) > AIOS(3) = AAO(3) > NLEM(2) > Generic(1)

  Layer 2 — check_contraindication_hardstop() (rule-based, zero-latency)
    Deterministic keyword-collision engine. Runs AFTER the LLM generates its
    answer. If a dangerous symptom + dangerous drug co-occur in the query or
    AI response, the status is FORCED to CONTRADICTION regardless of NLI score.

    Current rules
    -------------
    FUNGAL_KERATITIS_STEROID
      Trigger  : "feathery margins" | "satellite lesions" | "fungal keratitis"
                 | "fungal ulcer" | "aspergillus" | "fusarium"
               + "steroid" | "prednisolone" | "dexamethasone" | "betamethasone"
                 | "corticosteroid"
      Severity : CRITICAL
      Reason   : Steroids suppress immunity → fungal proliferation → perforation.
                 Use Natamycin 5% / Voriconazole 1%.

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
from typing import List, Dict, Any, Set, Optional
from sentence_transformers.cross_encoder import CrossEncoder


# ---------------------------------------------------------------------------
# Hard-Stop Contraindication Rules
# ---------------------------------------------------------------------------
# Each rule fires when ANY symptom_keyword AND ANY drug_keyword appear in the
# combined query + AI-response text (case-insensitive substring match).
# Add new rules to this list — no code changes elsewhere needed.
# ---------------------------------------------------------------------------
CONTRAINDICATION_RULES: List[Dict[str, Any]] = [
    {
        "id":       "FUNGAL_KERATITIS_STEROID",
        "name":     "Steroid Contraindicated in Fungal Keratitis",
        "severity": "CRITICAL",
        "symptom_keywords": [
            "feathery margins", "feathery margin",
            "satellite lesions", "satellite lesion",
            "fungal keratitis", "fungal ulcer",
            "fungal infection of cornea", "mycotic keratitis",
            "aspergillus", "fusarium", "candida keratitis",
        ],
        "drug_keywords": [
            "steroid", "steroids", "prednisolone", "dexamethasone",
            "betamethasone", "triamcinolone", "methylprednisolone",
            "hydrocortisone", "corticosteroid", "corticosteroids",
            "cortisone", "fluorometholone",
        ],
        "reason": (
            "Steroids (prednisolone / dexamethasone) are ABSOLUTELY "
            "CONTRAINDICATED in fungal keratitis. They suppress the local "
            "immune response, allowing rapid fungal proliferation and risking "
            "corneal perforation and endophthalmitis."
        ),
        "correct_treatment": (
            "**Fungal Keratitis — Correct Protocol (ICMR/AIIMS)**\n\n"
            "- **First-line:** Natamycin 5% eye drops — 1 drop every **1 hour** "
            "(waking hours) × 3–4 weeks\n"
            "- **Alternative / deep stromal:** Voriconazole 1% eye drops or "
            "oral Voriconazole **200 mg BD**\n"
            "- **Cycloplegic:** Atropine 1% TDS for pain and to prevent synechiae\n"
            "- **Do NOT use:** Prednisolone, Dexamethasone, or any corticosteroid\n"
            "- **Refer** to cornea specialist if no improvement in 48–72 hours "
            "or if perforation is imminent\n"
            "- **Culture** corneal scrapings before starting antifungals"
        ),
    },
    {
        "id":       "VIRAL_KERATITIS_ANTIFUNGAL",
        "name":     "Antifungal Contraindicated in Viral (Herpetic) Keratitis",
        "severity": "HIGH",
        "symptom_keywords": [
            "dendritic ulcer", "dendritic lesion", "herpetic keratitis",
            "herpes simplex keratitis", "hsv keratitis", "geographic ulcer",
        ],
        "drug_keywords": [
            "natamycin", "voriconazole", "antifungal", "fluconazole",
            "itraconazole", "amphotericin",
        ],
        "reason": (
            "Antifungals are ineffective and waste critical time in viral "
            "(herpetic) keratitis. The correct treatment is antiviral therapy "
            "with Acyclovir / Ganciclovir."
        ),
        "correct_treatment": (
            "**Herpetic Keratitis — Correct Protocol (AIIMS/AAO)**\n\n"
            "- **First-line:** Acyclovir 3% ointment **5× daily** × 14 days\n"
            "- **Alternative:** Ganciclovir 0.15% gel **5× daily**\n"
            "- **Do NOT use:** Antifungals (Natamycin, Voriconazole)\n"
            "- **Mild topical steroids** may be added for stromal disease "
            "*only under specialist supervision*\n"
            "- **Cycloplegic** for iritis component"
        ),
    },
]


def check_contraindication_hardstop(
    query: str,
    ai_response: str,
) -> Optional[Dict[str, Any]]:
    """
    Zero-latency keyword-collision safety check.

    Scans the combined query + AI response for dangerous drug-symptom pairs.
    Returns the triggered rule dict if a hard-stop is warranted, else None.

    Parameters
    ----------
    query : str
        The clinician's original question.
    ai_response : str
        The LLM-generated answer text.

    Returns
    -------
    dict | None
        The matching rule from CONTRAINDICATION_RULES, or None if safe.
    """
    combined = (query + " " + ai_response).lower()

    for rule in CONTRAINDICATION_RULES:
        symptom_hit = any(kw in combined for kw in rule["symptom_keywords"])
        drug_hit    = any(kw in combined for kw in rule["drug_keywords"])
        if symptom_hit and drug_hit:
            return rule

    return None

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
