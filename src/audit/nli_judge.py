"""
src/audit/nli_judge.py — PairwiseAuditor + Hard-Stop Safety Engine
===================================================================
Clinical Safety Logic Engine.

Three independent safety layers
--------------------------------
  Layer 1 — PairwiseAuditor (NLI-based)
    Cross-chunk contradiction detection using DeBERTa-v3-small CrossEncoder.
    Authority ranking: ICMR(5) > AIIMS(4) > AIOS(3) = AAO(3) > NLEM(2) > Generic(1)

  Layer 2 — check_contraindication_hardstop() (rule-based, zero-latency)
    Deterministic keyword-collision engine. Fires when a dangerous symptom +
    dangerous drug co-occur in the combined query + AI response.
    Status is FORCED to CONTRADICTION regardless of NLI score.

    Current rules
    -------------
    FUNGAL_KERATITIS_STEROID
      Trigger  : feathery margins / satellite lesions / fungal keratitis …
               + steroid / prednisolone / dexamethasone …
      Severity : CRITICAL

    VIRAL_KERATITIS_ANTIFUNGAL
      Trigger  : dendritic ulcer / herpetic keratitis …
               + natamycin / voriconazole / antifungal …
      Severity : HIGH

  Layer 3 — check_pediatric_dose_hardstop() (weight-arithmetic, zero-latency)
    Fires when the query mentions a patient weight (kg) AND the AI response
    proposes a flat mg dose that breaches a weight-scaled safety threshold.

    Thresholds
    ----------
    dose > weight_kg × 20  → severity CRITICAL  → forces CONTRADICTION
    dose > weight_kg × 15  → severity WARNING   → amber banner, NLI unchanged

    Also enforces the Inference Prohibition: if the query is paediatric but
    the system prompt has been followed correctly, the LLM will already have
    printed the standard safety warning. This layer catches any case where it
    did not.

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
from typing import List, Dict, Any, Set, Optional, Tuple
from sentence_transformers.cross_encoder import CrossEncoder


# ---------------------------------------------------------------------------
# Pediatric Safety — Regex patterns
# ---------------------------------------------------------------------------
# Matches flat weight values: "12 kg", "12kg", "12 kilograms"
# NOT "mg/kg" — that is a rate, not a body weight.
_WEIGHT_RE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*(?:kg|kilograms?|kgs?)\b',
    re.IGNORECASE,
)

# Matches flat mg dose values: "500mg", "500 mg", "amoxicillin 500 mg"
# Negative lookahead (?!\s*/\s*kg) excludes mg/kg rate expressions.
_FLAT_MG_RE = re.compile(
    r'\b(\d+(?:\.\d+)?)\s*mg\b(?!\s*/\s*kg)',
    re.IGNORECASE,
)

# Keywords that signal the query is about a paediatric patient.
_PAEDIATRIC_KW = (
    "child", "children", "infant", "baby", "babies",
    "neonate", "neonatal", "newborn", "toddler",
    "paediatric", "pediatric", "paeds", "peds",
    "preschool", "school-age", "under 12", "under 5",
)

# Threshold multipliers (mg per kg body weight)
_WARN_MG_PER_KG  = 15.0   # soft warning
_CRIT_MG_PER_KG  = 20.0   # hard CONTRADICTION override


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
# Pediatric Weight-Based Dose Safety (Layer 3)
# ---------------------------------------------------------------------------

def extract_pediatric_context(query: str, ai_response: str) -> Dict[str, Any]:
    """
    Extract weight (kg) and proposed flat dose (mg) from the clinical text.

    Weight is taken from the QUERY (clinician supplies it).
    Dose is taken from the AI RESPONSE (what was prescribed).
    When multiple values are found the MOST CONSERVATIVE choices are used:
      • Smallest weight  → lowest safe dose ceiling → strictest check.
      • Largest flat mg  → worst-case prescribed dose → strictest check.

    Returns
    -------
    {
        "weight_kg"          : float | None,
        "dose_mg"            : float | None,
        "is_paediatric"      : bool,
        "all_weights"        : list[float],
        "all_doses_in_response" : list[float],
    }
    """
    # Weight from query only (AI response might mention dosing thresholds)
    query_weights = [float(m.group(1)) for m in _WEIGHT_RE.finditer(query)]

    # Flat mg dose from AI response only (not from the PDF context blocks)
    resp_doses = [float(m.group(1)) for m in _FLAT_MG_RE.finditer(ai_response)]

    lower_query = query.lower()
    is_paed = (
        bool(query_weights)
        or any(kw in lower_query for kw in _PAEDIATRIC_KW)
    )

    return {
        "weight_kg":              min(query_weights) if query_weights else None,
        "dose_mg":                max(resp_doses)    if resp_doses    else None,
        "is_paediatric":          is_paed,
        "all_weights":            query_weights,
        "all_doses_in_response":  resp_doses,
    }


def check_pediatric_dose_hardstop(
    query: str,
    ai_response: str,
) -> Optional[Dict[str, Any]]:
    """
    Pediatric Weight-Based Dose Safety Override (Layer 3).

    Fires when:
      • The query contains a patient weight in kg, AND
      • The AI response proposes a flat mg dose that exceeds the
        weight-scaled safety threshold.

    Severity mapping
    ----------------
    dose_mg > weight_kg × 20  → CRITICAL  (forces NLI → CONTRADICTION)
    dose_mg > weight_kg × 15  → WARNING   (amber banner, NLI unchanged)

    Returns a rule-shaped dict (same schema as CONTRAINDICATION_RULES) if
    triggered, or None if the dose is safe / no weight found.
    """
    ctx = extract_pediatric_context(query, ai_response)

    weight = ctx["weight_kg"]
    dose   = ctx["dose_mg"]

    # Both must be present to evaluate
    if weight is None or dose is None:
        return None

    warn_ceiling = weight * _WARN_MG_PER_KG   # e.g. 15 kg × 15 = 225 mg
    crit_ceiling = weight * _CRIT_MG_PER_KG   # e.g. 15 kg × 20 = 300 mg

    if dose <= warn_ceiling:
        return None   # dose is within safe range

    severity = "CRITICAL" if dose > crit_ceiling else "WARNING"
    exceeded = "CRITICAL safety ceiling" if severity == "CRITICAL" else "standard safety limit"
    threshold_used = crit_ceiling if severity == "CRITICAL" else warn_ceiling
    mg_per_kg_used = _CRIT_MG_PER_KG if severity == "CRITICAL" else _WARN_MG_PER_KG

    return {
        "id":       "PEDIATRIC_DOSE_EXCEEDED",
        "name": (
            f"Paediatric Dose Override — "
            f"{dose:.0f} mg exceeds {weight:.1f} kg × "
            f"{mg_per_kg_used:.0f} mg/kg "
            f"({threshold_used:.0f} mg {exceeded})"
        ),
        "severity": severity,
        # Extra keys for UI display
        "weight_kg":         weight,
        "dose_mg":           dose,
        "warn_ceiling_mg":   warn_ceiling,
        "crit_ceiling_mg":   crit_ceiling,
        "reason": (
            f"Proposed dose of **{dose:.0f} mg** for a **{weight:.1f} kg** patient "
            f"{'exceeds the critical hard ceiling of ' + str(_CRIT_MG_PER_KG) + ' mg/kg' if severity == 'CRITICAL' else 'exceeds the standard limit of ' + str(_WARN_MG_PER_KG) + ' mg/kg'}. "
            f"Maximum safe dose at {mg_per_kg_used:.0f} mg/kg = **{threshold_used:.0f} mg**. "
            f"{'Adult-dose tablets must NOT be used for this weight.' if severity == 'CRITICAL' else 'Verify weight-based dosing from the source guideline.'}"
        ),
        "correct_treatment": (
            f"**⚠️ Paediatric Weight-Based Dosing Safety Override**\n\n"
            f"| Parameter | Value |\n"
            f"|---|---|\n"
            f"| Patient weight | **{weight:.1f} kg** |\n"
            f"| Proposed dose (AI) | **{dose:.0f} mg** |\n"
            f"| Safe max (15 mg/kg) | **{warn_ceiling:.0f} mg** |\n"
            f"| Hard ceiling (20 mg/kg) | **{crit_ceiling:.0f} mg** |\n\n"
            f"🔴 **Do NOT administer {dose:.0f} mg to a {weight:.1f} kg patient.**\n\n"
            f"If the source guideline does not specify mg/kg dosing for this weight, "
            f"you MUST state:\n"
            f"> *'Safety Warning: Weight-based dosing (15 mg/kg) is required but not "
            f"found in current source. DO NOT use adult tablets without paediatric "
            f"dose calculation.'*\n\n"
            f"Consult a paediatric specialist or a paediatric pharmacopoeia "
            f"before prescribing."
        ),
        # Keep keyword lists empty — this rule uses arithmetic, not keywords
        "symptom_keywords": [],
        "drug_keywords":    [],
    }


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
