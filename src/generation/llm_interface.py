import ollama
import yaml
import os

# Load configuration
_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
with open(_config_path, 'r') as f:
    _config = yaml.safe_load(f)
model_name_default = _config.get('model_name', 'llama3.2:3b')
from typing import List, Dict, Any

class ClinicalSynthesizer:
    def __init__(self, model_name=model_name_default):
        self.model_name = model_name
        self.client = ollama.Client()
        
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown")
            page = meta.get("page", "Unknown")
            year = meta.get("doc_year", meta.get("statute_year", "Unknown"))
            
            c_text = f"[Citation {i}: {filename} (Year: {year}), Page {page}]\n{chunk['text']}"
            formatted_chunks.append(c_text)
            
        return "\n\n".join(formatted_chunks)

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conflict_payload: Dict[str, Any],
        view_mode: str = "Brief",
    ) -> str:
        """
        Generate a clinical answer via Ollama.

        Parameters
        ----------
        query : str
            The clinician's question.
        chunks : list
            Retrieved protocol chunks (context).
        conflict_payload : dict
            Output from PairwiseAuditor.detect_logic_flips().
        view_mode : str
            "Short"  → 3-sentence summary: primary diagnosis + 1st-line drug only.
            "Brief"  → Full clinical protocol: etiology, dosage, contraindications,
                       follow-up schedule.
        """
        logic_flip_alert = ""

        if conflict_payload.get("ConflictFound") is True:
            prioritized = conflict_payload.get("PrioritizedChunk", {}).get("metadata", {})
            rejected    = conflict_payload.get("RejectedChunk",    {}).get("metadata", {})
            doc_a            = rejected.get("filename", "Unknown")
            doc_b            = prioritized.get("filename", "Unknown")
            year_prioritized = prioritized.get("doc_year", prioritized.get("statute_year", "Unknown"))
            logic_flip_alert = (
                f"⚠️ **CLINICAL CONFLICT DETECTED:** Contradiction found between "
                f"[{doc_a}] and [{doc_b}]. Prioritizing the [{year_prioritized}] protocol."
            )

        context_str = self._format_context(chunks)

        # ── View-mode system prompt ──────────────────────────────────────────
        _base = (
            "You are an expert Clinical AI Assistant named Med-Verify, built for "
            "doctors and medical professionals. Answer STRICTLY from the provided "
            "<context>. DO NOT hallucinate, invent dosages, or use outside knowledge. "
            "Always cite inline (e.g. '[Citation 1]'). "
            "Format every dosage value in bold (e.g. **500 mg twice daily**) and "
            "highlight critical warnings in bold."
        )

        if view_mode == "Short":
            mode_instruction = (
                " RESPONSE FORMAT — SHORT: Respond in exactly 3 sentences. "
                "Sentence 1: state the primary diagnosis or condition. "
                "Sentence 2: state the first-line drug/intervention and its dosage. "
                "Sentence 3: state one critical warning or monitoring parameter. "
                "Do NOT include etiology, full protocols, or follow-up schedules."
            )
        else:  # "Brief" (full protocol)
            mode_instruction = (
                " RESPONSE FORMAT — FULL PROTOCOL: Provide a comprehensive clinical "
                "response structured as: (1) Etiology / Pathophysiology, "
                "(2) Diagnostic criteria, "
                "(3) First-line and second-line treatment with exact dosages, "
                "(4) Contraindications and drug interactions, "
                "(5) Follow-up schedule and monitoring parameters. "
                "Use markdown headings (##) for each section."
            )

        system_prompt = _base + mode_instruction

        user_prompt = f"""{logic_flip_alert}

<context>
{context_str}
</context>

<query>
{query}
</query>
"""
        response = self.client.generate(
            model=self.model_name,
            prompt=user_prompt,
            system=system_prompt,
            stream=False,
        )

        if logic_flip_alert:
            return logic_flip_alert + "\n\n" + response["response"]
        return response["response"]
