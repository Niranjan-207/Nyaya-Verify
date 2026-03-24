"""
src/verifier.py — ClinicalAuditor
==================================
Formal Natural Language Inference (NLI) pipeline for answer verification.

  Premise    = retrieved PDF / protocol chunk  (Ground Truth)
  Hypothesis = AI-generated answer             (to be verified)

3-Tier Classification
---------------------
  ENTAILED      → answer is logically proven by the source chunk      (Green ✅)
  CONTRADICTION → answer directly conflicts with the source chunk     (Red   🔴)
  NEUTRAL       → answer contains info not found in the source chunk  (Yellow ⚠️)

Confidence
----------
  Softmax probability of the dominant label, expressed as a percentage.

Model
-----
  cross-encoder/nli-deberta-v3-large  (sentence-transformers CrossEncoder)
  — Same family as PairwiseAuditor (nli-deberta-v3-small) but larger model
    for higher accuracy on answer-level verification.
  — Downloaded via sentence-transformers; no HuggingFace Hub token required.
  — Label index order: 0 = contradiction, 1 = entailment, 2 = neutral

VRAM Note
---------
  CrossEncoder auto-selects CUDA when available. Call .unload() immediately
  after use so Ollama can reclaim VRAM before next heavy process.
"""

import gc
import datetime
import torch
from typing import List, Dict, Any
from sentence_transformers.cross_encoder import CrossEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "cross-encoder/nli-deberta-v3-large"

# cross-encoder/nli-deberta-v3-* label index mapping
# (matches the comment in nli_judge.py: "Contradiction → Index 0")
_IDX_TO_STATUS: Dict[int, str] = {
    0: "CONTRADICTION",
    1: "ENTAILED",
    2: "NEUTRAL",
}

_STATUS_EMOJI: Dict[str, str] = {
    "ENTAILED":      "✅",
    "NEUTRAL":       "⚠️",
    "CONTRADICTION": "🔴",
}

_STATUS_LABEL: Dict[str, str] = {
    "ENTAILED":      "LOGICALLY VERIFIED",
    "NEUTRAL":       "UNVERIFIED — NEUTRAL",
    "CONTRADICTION": "CONTRADICTION DETECTED",
}

_STATUS_COLOR: Dict[str, str] = {
    "ENTAILED":      "green",
    "NEUTRAL":       "orange",
    "CONTRADICTION": "red",
}


# ---------------------------------------------------------------------------
# ClinicalAuditor
# ---------------------------------------------------------------------------
class ClinicalAuditor:
    """
    Wraps cross-encoder/nli-deberta-v3-large as a sequence-pair NLI classifier.
    Uses sentence_transformers.CrossEncoder — same API as PairwiseAuditor,
    no HuggingFace Hub token required.

    Usage
    -----
        auditor  = ClinicalAuditor()
        verdicts = auditor.verify_answer(answer_text, retrieved_chunks)
        summary  = auditor.aggregate_verdict(verdicts)
        auditor.unload()   # release VRAM before Ollama takes over
    """

    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] ClinicalAuditor loading {MODEL_ID} on {device}…")
        self.device = device
        self.model  = CrossEncoder(MODEL_ID, device=device)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _nli_pair(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Single NLI forward pass via CrossEncoder.

        Returns
        -------
        {
            "status":     "ENTAILED" | "NEUTRAL" | "CONTRADICTION",
            "confidence": float  (0–100),
            "breakdown":  {"ENTAILED": float, "NEUTRAL": float,
                           "CONTRADICTION": float}
        }
        """
        # CrossEncoder.predict returns raw logits; convert to probabilities
        raw     = self.model.predict([(premise, hypothesis)])
        probs   = torch.softmax(torch.tensor(raw), dim=1).numpy()[0]

        top_idx    = int(probs.argmax())
        status     = _IDX_TO_STATUS.get(top_idx, "NEUTRAL")
        confidence = round(float(probs[top_idx]) * 100, 1)

        breakdown: Dict[str, float] = {
            _IDX_TO_STATUS[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probs)
        }
        return {"status": status, "confidence": confidence,
                "breakdown": breakdown}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def verify_answer(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For every retrieved chunk, run NLI with:
          premise   = chunk["text"]
          hypothesis = answer

        Returns a list of verdict dicts (one per chunk) each containing:
          status, confidence, breakdown, emoji, color, label,
          chunk_text, metadata, timestamp
        """
        ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verdicts = []

        for chunk in chunks:
            premise = chunk.get("text", "")
            meta    = chunk.get("metadata", {})
            result  = self._nli_pair(premise, answer)

            verdicts.append({
                "status":     result["status"],
                "confidence": result["confidence"],
                "breakdown":  result["breakdown"],
                "emoji":      _STATUS_EMOJI[result["status"]],
                "color":      _STATUS_COLOR[result["status"]],
                "label":      _STATUS_LABEL[result["status"]],
                "chunk_text": premise,
                "metadata":   meta,
                "timestamp":  ts,
            })

        return verdicts

    def aggregate_verdict(
        self,
        verdicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Roll-up rule across all per-chunk verdicts:
          • Any CONTRADICTION present  → overall = CONTRADICTION
          • All ENTAILED               → overall = ENTAILED
          • Otherwise                  → overall = NEUTRAL
        """
        if not verdicts:
            return {
                "status":     "NEUTRAL",
                "confidence": 0.0,
                "emoji":      _STATUS_EMOJI["NEUTRAL"],
                "color":      _STATUS_COLOR["NEUTRAL"],
                "label":      _STATUS_LABEL["NEUTRAL"],
            }

        statuses = [v["status"] for v in verdicts]
        if "CONTRADICTION" in statuses:
            agg = "CONTRADICTION"
        elif all(s == "ENTAILED" for s in statuses):
            agg = "ENTAILED"
        else:
            agg = "NEUTRAL"

        avg_conf = round(
            sum(v["confidence"] for v in verdicts) / len(verdicts), 1
        )
        return {
            "status":     agg,
            "confidence": avg_conf,
            "emoji":      _STATUS_EMOJI[agg],
            "color":      _STATUS_COLOR[agg],
            "label":      _STATUS_LABEL[agg],
        }

    def unload(self) -> None:
        """
        Delete the CrossEncoder and clear CUDA cache.
        Call this immediately after verification so Ollama can reclaim VRAM.
        """
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[*] ClinicalAuditor unloaded — VRAM released.")
