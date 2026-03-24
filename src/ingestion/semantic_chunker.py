import re
import torch
from typing import Any, Dict, List, Optional

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# Pre-compiled header-detection patterns
_NUMBERED_SECTION = re.compile(r'^\d+(?:\.\d+)+\s+\S')   # "3.2 Title", "1.1.1 Clause"
_SIMPLE_NUMBERED  = re.compile(r'^\d+\.\s+[A-Z]')          # "3. Title"
_FIRST_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


class HybridHierarchicalChunker:
    """
    Three-stage Semantic + Structural Hybrid Chunker.

    Stage 1 — Structural split
        Groups raw PDF blocks into super-chunks bounded by section headers:
          - ALL-CAPS lines (>=3 alphabetic chars)
          - Numbered sections — "3.2 Heading", "1. Heading"
          - Bold-like short lines — <=10 words, starts with capital, no
            terminal period/comma/semicolon
        "Provided that" and "Explanation.—" are never treated as boundaries.

    Stage 2 — Semantic split
        Super-chunks exceeding token_threshold tokens are further divided
        by LangChain SemanticChunker backed by S-BioBERT
        (pritamdeka/S-BioBert-SNLI-Mean-Tokens).

    Stage 3 — Smart overlap
        Every Stage-2 sub-chunk is prefixed with:
          [section_header] + [first sentence of the prior sub-chunk]
        Preserves contextual coherence without blind character overlap.
    """

    def __init__(
        self,
        model_name: str = "pritamdeka/S-BioBert-SNLI-Mean-Tokens",
        token_threshold: int = 600,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
    ):
        self.token_threshold = token_threshold

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] HybridHierarchicalChunker embedding device: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_section_header(text: str) -> bool:
        """Returns True when text looks like a section heading."""
        text = text.strip()
        if not text or len(text) > 120:
            return False

        # Explicit exceptions — must never act as a split boundary
        if text.startswith("Provided that") or text.startswith("Explanation.—"):
            return False

        # ALL-CAPS line (digits/spaces/punctuation ignored when checking case)
        alpha = [c for c in text if c.isalpha()]
        if len(alpha) >= 3 and all(c.isupper() for c in alpha):
            return True

        # Numbered sections: "3.2 Heading" or "3. Heading"
        if _NUMBERED_SECTION.match(text) or _SIMPLE_NUMBERED.match(text):
            return True

        # Bold-like short line: <=10 words, capitalised, no terminal sentence punct
        words = text.split()
        if (
            1 <= len(words) <= 10
            and text[0].isupper()
            and text[-1] not in ".,;"
        ):
            return True

        return False

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Word-count x 1.3 — reasonable subword-token estimate."""
        return int(len(text.split()) * 1.3)

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Returns the first sentence of text for use as the overlap prefix."""
        parts = _FIRST_SENT_SPLIT.split(text.strip(), maxsplit=1)
        sentence = parts[0] if parts else text
        # Cap at 300 chars to avoid excessively long overlap prefixes
        return sentence[:300]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, parsed_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parameters
        ----------
        parsed_blocks : list of dicts with keys "text" and "metadata"
            Raw output from pdf_parser.extract_text_with_metadata().

        Returns
        -------
        list of dicts with keys "text" and "metadata"
        """

        # Stage 1: structural grouping into super-chunks
        super_chunks: List[Dict[str, Any]] = []
        current_header: Optional[str] = None
        current_texts:  List[str]     = []
        current_meta:   Optional[Dict] = None

        for block in parsed_blocks:
            text = block["text"].strip()
            if not text:
                continue

            if self._is_section_header(text):
                # Flush accumulated body under the previous header
                if current_texts:
                    super_chunks.append({
                        "header":   current_header,
                        "text":     " ".join(current_texts),
                        "metadata": current_meta,
                    })
                current_header = text
                current_texts  = []
                current_meta   = block["metadata"]
            else:
                if current_meta is None:
                    current_meta = block["metadata"]
                current_texts.append(text)

        # Flush the final accumulated body
        if current_texts:
            super_chunks.append({
                "header":   current_header,
                "text":     " ".join(current_texts),
                "metadata": current_meta,
            })

        # Stage 2 + 3: semantic chunking with smart overlap
        final_chunks: List[Dict[str, Any]] = []

        for sc in super_chunks:
            body     = sc["text"]
            header   = sc["header"] or ""
            metadata = sc["metadata"]

            if self._approx_tokens(body) <= self.token_threshold:
                # Small enough to keep whole — skip semantic splitting
                combined = f"{header}\n{body}".strip() if header else body
                final_chunks.append({"text": combined, "metadata": metadata})
                continue

            # Stage 2: semantic split for large sections
            docs      = self.semantic_splitter.create_documents([body])
            sub_texts = [d.page_content for d in docs]

            # Stage 3: prepend section_header + first sentence of prior chunk
            prev_first: Optional[str] = None
            for i, sub in enumerate(sub_texts):
                prefix = f"{header}\n" if header else ""
                if prev_first and i > 0:
                    prefix += f"{prev_first} "

                final_chunks.append({
                    "text":     (prefix + sub).strip(),
                    "metadata": metadata,
                })
                prev_first = self._first_sentence(sub)

        return final_chunks
