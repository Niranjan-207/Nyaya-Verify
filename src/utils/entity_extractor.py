"""
src/utils/entity_extractor.py — Primary Disease Entity Extractor
================================================================
Extracts the dominant medical condition from a user query or AI response
text using a three-stage local pipeline. No external API calls, no LLM
invocation — runs in microseconds.

Extraction Stages (in priority order)
--------------------------------------
  Stage 1 — Vocabulary scan
    Direct case-insensitive substring match against a curated vocabulary
    of ophthalmology and general medical conditions. The longest match
    wins to avoid "cataract" beating "posterior capsule opacification".

  Stage 2 — Syntactic pattern matching
    Regex patterns that capture the noun phrase after trigger phrases
    such as "treatment for", "management of", "diagnosis of", etc.

  Stage 3 — Chunk filename heuristic
    If retrieved chunks are supplied, strip the PDF extension from the
    first (highest-ranked) chunk's filename, un-slug it, and return it.
    This works because protocol PDFs are named after their condition
    (e.g. "Glaucoma.pdf", "Atopic dermatitis.pdf").

  Stage 4 — Query head words
    Last resort: return the first 4 content words from the query after
    stripping common question-starter phrases.

All stages are wrapped so the function never raises. Returns a non-empty
string suitable as a DuckDuckGo image search term.
"""

import os
import re
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Stage 1 — Curated condition vocabulary
# ---------------------------------------------------------------------------
# Listed longest-first so greedy matching picks the most specific term.
_OPHTHALMO_VOCABULARY: List[str] = [
    # Glaucoma subtypes
    "normal tension glaucoma", "open-angle glaucoma", "closed-angle glaucoma",
    "angle-closure glaucoma", "primary open angle glaucoma",
    "neovascular glaucoma", "pigmentary glaucoma", "pseudoexfoliation glaucoma",
    "glaucoma",
    # Cataract
    "posterior capsule opacification", "posterior subcapsular cataract",
    "nuclear sclerotic cataract", "cortical cataract", "cataract",
    # Retina
    "proliferative diabetic retinopathy", "non-proliferative diabetic retinopathy",
    "diabetic macular edema", "diabetic retinopathy",
    "age-related macular degeneration", "macular degeneration", "amd",
    "central retinal artery occlusion", "central retinal vein occlusion",
    "retinal detachment", "vitreous hemorrhage", "epiretinal membrane",
    "macular hole",
    # Cornea
    "acanthamoeba keratitis", "fungal keratitis", "bacterial keratitis",
    "herpetic keratitis", "keratitis", "corneal ulcer", "corneal abrasion",
    "keratoconus", "bullous keratopathy", "fuchs endothelial dystrophy",
    # Anterior segment
    "anterior uveitis", "posterior uveitis", "panuveitis", "uveitis",
    "hyphema", "hypopyon",
    # External
    "allergic conjunctivitis", "viral conjunctivitis", "conjunctivitis",
    "dry eye syndrome", "dry eye", "blepharitis", "chalazion", "stye", "hordeolum",
    "pterygium", "pinguecula",
    # Neuro-ophthalmo
    "optic neuritis", "papilledema", "ischemic optic neuropathy",
    "sixth nerve palsy", "third nerve palsy", "fourth nerve palsy",
    "diplopia", "nystagmus",
    # Paediatric
    "congenital cataract", "retinopathy of prematurity", "amblyopia",
    "strabismus", "esotropia", "exotropia",
    # Lid / orbit
    "orbital cellulitis", "dacryocystitis", "dacryoadenitis",
    "blepharospasm", "ptosis", "entropion", "ectropion",
    # General medical (from the 53 ingested PDFs)
    "alzheimer disease", "alzheimer's disease", "alzheimer",
    "anxiety disorder", "anxiety",
    "rheumatoid arthritis", "osteoarthritis", "arthritis",
    "atherosclerosis",
    "atopic dermatitis", "eczema",
    "type 2 diabetes", "diabetes mellitus", "diabetes",
    "hypertension",
    "acne vulgaris", "acne",
    "chronic obstructive pulmonary disease", "copd", "asthma",
    "myocardial infarction", "heart failure", "cardiac failure",
    "stroke", "cerebrovascular accident",
    "parkinson disease", "parkinson's disease",
    "epilepsy", "seizure",
    "migraine", "headache",
    "urinary tract infection", "uti",
    "pneumonia",
    "tuberculosis",
    "dengue", "malaria",
    "typhoid",
]

# Pre-sort by length descending for greedy longest-match
_OPHTHALMO_VOCABULARY_SORTED: List[str] = sorted(
    _OPHTHALMO_VOCABULARY, key=len, reverse=True
)


# ---------------------------------------------------------------------------
# Stage 2 — Syntactic extraction patterns
# ---------------------------------------------------------------------------
# Each pattern captures group 1 = the condition noun phrase.
_EXTRACTION_PATTERNS: List[str] = [
    r"treatment\s+(?:for|of)\s+([\w][\w\s\-]+?)(?:\s+with|\s+in|\s+at|\s+and|\?|,|$)",
    r"management\s+(?:of|for)\s+([\w][\w\s\-]+?)(?:\s+with|\s+in|\s+and|\?|,|$)",
    r"diagnosis\s+of\s+([\w][\w\s\-]+?)(?:\s+with|\s+in|\?|,|$)",
    r"(?:patient\s+with|case\s+of)\s+([\w][\w\s\-]+?)(?:\s+with|\s+and|\?|,|$)",
    r"protocol\s+for\s+([\w][\w\s\-]+?)(?:\s+with|\s+in|\?|,|$)",
    r"(?:drug|medication|medicine)\s+for\s+([\w][\w\s\-]+?)(?:\s+with|\s+in|\?|,|$)",
    r"(?:surgery|procedure)\s+for\s+([\w][\w\s\-]+?)(?:\s+with|\s+in|\?|,|$)",
    r"post[-\s]op(?:erative)?\s+(?:care\s+for\s+)?([\w][\w\s\-]+?)(?:\s+with|\s+in|\?|,|$)",
]

# Question-starter prefixes to strip before pattern matching
_FILLER_PATTERN = re.compile(
    r"^(?:what\s+is|what\s+are|how\s+(?:do|does|to)|when\s+(?:do|does|is)|"
    r"why\s+(?:is|are)|can\s+(?:you|i)|tell\s+me\s+about|"
    r"explain|describe|define|list|give\s+me)\s+(?:the\s+)?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_disease_entity(
    text: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Extract the primary disease / condition name from query or response text.

    Parameters
    ----------
    text : str
        User query string or LLM-generated response (first 500 chars used).
    chunks : list, optional
        Retrieved protocol chunks. Used as Stage 3 fallback — the filename
        of the first (best-ranked) chunk is decoded as the condition name.

    Returns
    -------
    str
        Clean condition name suitable for image search, e.g. "Glaucoma".
        Never empty — falls back to the first 4 words of the query.
    """
    if not text:
        return _chunk_fallback(chunks) or "ophthalmology"

    # Work on first 500 chars — entity is almost always in the opening
    sample = text[:500].strip()

    # ── Stage 1: Vocabulary scan ──────────────────────────────────────────
    lower = sample.lower()
    for term in _OPHTHALMO_VOCABULARY_SORTED:
        if term in lower:
            # Return title-cased canonical form
            return term.title()

    # ── Stage 2: Syntactic pattern matching ───────────────────────────────
    # Strip question starters first
    stripped = _FILLER_PATTERN.sub("", sample).strip()
    for pattern in _EXTRACTION_PATTERNS:
        m = re.search(pattern, stripped, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().strip(".,;:()")
            # Reject implausibly long or short candidates
            if 3 <= len(candidate) <= 60:
                return candidate.title()

    # ── Stage 3: PDF filename heuristic ───────────────────────────────────
    fallback = _chunk_fallback(chunks)
    if fallback:
        return fallback

    # ── Stage 4: Head words of the cleaned query ──────────────────────────
    words = stripped.split()[:4]
    return " ".join(words).strip("?.,;:()").title() or "ophthalmology"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _chunk_fallback(chunks: Optional[List[Dict[str, Any]]]) -> str:
    """
    Derive condition name from the first retrieved chunk's PDF filename.
    Returns empty string if chunks is None/empty or filename is blank.
    """
    if not chunks:
        return ""
    raw = chunks[0].get("metadata", {}).get("filename", "")
    if not raw:
        return ""
    name = os.path.splitext(os.path.basename(raw))[0]
    return name.replace("_", " ").replace("-", " ").strip().title()
