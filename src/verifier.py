"""
src/verifier.py — ClinicalAuditor
==================================
Formal Natural Language Inference (NLI) pipeline for answer verification.

  Premise    = retrieved PDF chunk  (Ground Truth)
  Hypothesis = AI-generated answer  (to be verified)

3-Tier Classification
---------------------
  ENTAILED      → answer is logically proven by the PDF chunk         (Green ✅)
  CONTRADICTION → answer directly conflicts with the PDF chunk        (Red   🔴)
  NEUTRAL       → answer contains info not found in the PDF chunk     (Yellow ⚠️)

Confidence
----------
  Softmax probability of the dominant label, expressed as a percentage.

VRAM Note
---------
  Model is loaded in fp16 (~430 MB) so it can coexist with Ollama on
  a 6 GB GPU (RTX 4050).  Always call .unload() when done to free VRAM
  before the next heavy process takes over.
"""

import gc
import datetime
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli"

# Canonical 3-tier statuses
_NLI_TO_STATUS: Dict[str, str] = {
    "entailment":    "ENTAILED",
    "neutral":       "NEUTRAL",
    "contradiction": "CONTRADICTION",
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
    Wraps MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli as a direct NLI
    sequence-pair classifier.

    Usage
    -----
        auditor  = ClinicalAuditor()
        verdicts = auditor.verify_answer(answer_text, retrieved_chunks)
        summary  = auditor.aggregate_verdict(verdicts)
        auditor.unload()          # free VRAM before Ollama takes over
    """

    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        print(f"[*] ClinicalAuditor loading {MODEL_ID} on {device} "
              f"({'fp16' if dtype == torch.float16 else 'fp32'})…")

        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()

        # Build label-index → canonical status map from model config
        # (guards against version drift in id2label ordering)
        self._idx_to_status: Dict[int, str] = {
            idx: _NLI_TO_STATUS.get(lab.lower(), "NEUTRAL")
            for idx, lab in self.model.config.id2label.items()
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _nli_pair(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Single NLI forward pass.

        Returns
        -------
        {
            "status":     "ENTAILED" | "NEUTRAL" | "CONTRADICTION",
            "confidence": float  (0–100),
            "breakdown":  {"ENTAILED": float, "NEUTRAL": float,
                           "CONTRADICTION": float}
        }
        """
        enc = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        enc    = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits

        # Cast to float32 before softmax (safe even if model is fp16)
        probs  = torch.softmax(logits.float(), dim=-1)[0].cpu()

        top_idx    = int(probs.argmax())
        status     = self._idx_to_status.get(top_idx, "NEUTRAL")
        confidence = round(float(probs[top_idx]) * 100, 1)

        breakdown: Dict[str, float] = {}
        for idx, st in self._idx_to_status.items():
            breakdown[st] = round(float(probs[idx]) * 100, 1)

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

        Returns a list of verdict dicts (one per chunk), each containing:
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

        Returns a summary dict with status, confidence (average),
        emoji, color, and display label.
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
        Explicitly delete the model + tokenizer and clear the CUDA cache.
        Call this immediately after verification so Ollama can reclaim VRAM.
        """
        for attr in ("model", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[*] ClinicalAuditor unloaded — VRAM released.")
