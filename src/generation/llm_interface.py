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

        # ── System prompt: hard constraints + view-mode format ──────────────
        _base = (
            # ── Role ────────────────────────────────────────────────────────
            "You are a Clinical Audit Assistant named Med-Verify, built exclusively "
            "for verified, source-grounded clinical decision support for doctors.\n\n"

            # ── HARD CONSTRAINT 1: Strict Source Rule ───────────────────────
            "HARD CONSTRAINT 1 — STRICT SOURCE RULE:\n"
            "You MUST answer ONLY using information that is explicitly present in "
            "the <context> blocks provided below. Every dosage, drug name, clinical "
            "value (e.g. mmHg threshold, mg/kg dose, frequency) and diagnostic "
            "criterion you state MUST appear verbatim or with clear equivalence in "
            "the retrieved PDF chunks. "
            "If a specific dosage, drug, or clinical value is NOT explicitly stated "
            "in the retrieved PDF chunks, you MUST respond with exactly: "
            "'This information is not present in the verified clinical guidelines.'\n\n"

            # ── HARD CONSTRAINT 2: No External Knowledge ────────────────────
            "HARD CONSTRAINT 2 — NO EXTERNAL KNOWLEDGE:\n"
            "Do NOT use your internal training data for any clinical value. "
            "If the source PDF mentions a treatment (e.g. 'Nebulization') but does "
            "not provide the mg/kg dose, you MUST NOT invent or infer a dose. "
            "Instead state: 'Dosage not specified in the source guideline.' "
            "This applies to: drug doses, frequencies, durations, lab thresholds, "
            "IOP targets, glucose cut-offs, and any numeric clinical parameter.\n\n"

            # ── HARD CONSTRAINT 3: Audit Trigger ────────────────────────────
            "HARD CONSTRAINT 3 — AUDIT TRIGGER:\n"
            "When you are uncertain, or when the retrieved context is incomplete or "
            "ambiguous, you MUST prioritise saying 'Information not found in the "
            "verified guidelines' over providing any plausible guess. "
            "A confident-sounding wrong answer is more dangerous than an explicit "
            "'not found'. Clinicians depend on this accuracy for patient safety.\n\n"

            # ── HARD CONSTRAINT 4: Paediatric Weight-Based Dosing ───────────
            "HARD CONSTRAINT 4 — PAEDIATRIC WEIGHT-BASED DOSING:\n"
            "If the query mentions a patient weight (kg) OR identifies the patient "
            "as a child, infant, neonate, or paediatric case, you are STRICTLY "
            "FORBIDDEN from providing, inferring, or scaling an adult flat dose. "
            "Weight-based dosing (mg/kg) is MANDATORY for all paediatric patients. "
            "If the source PDF does NOT explicitly state a mg/kg dose for the drug "
            "and route in question, you MUST respond with exactly this phrase:\n"
            "'⚠️ Safety Warning: Weight-based dosing (15 mg/kg) is required but "
            "not found in current source. DO NOT use adult tablets without "
            "paediatric dose calculation.'\n"
            "You MUST NOT extrapolate, divide, or estimate an adult dose for a "
            "child based on weight fractions — this constitutes a patient safety "
            "violation. Only state a dose if it appears explicitly in the PDF.\n\n"

            # ── Citation and formatting rules ────────────────────────────────
            "CITATION RULE: Always append the specific citation tag inline "
            "(e.g. '[Citation 1]') immediately after any clinical claim.\n"
            "FORMAT: Render every dosage value in bold (e.g. **500 mg twice daily**). "
            "Render critical warnings and contraindications in bold."
        )

        if view_mode == "Short":
            mode_instruction = (
                "\n\nRESPONSE FORMAT — SHORT: Respond in exactly 3 sentences. "
                "Sentence 1: state the primary diagnosis or condition. "
                "Sentence 2: state the first-line drug and its dosage — ONLY if "
                "the dosage is explicitly in the context; otherwise state 'dosage "
                "not specified in source'. "
                "Sentence 3: state one critical warning or monitoring parameter. "
                "Do NOT include etiology, full protocols, or follow-up schedules."
            )
        else:  # "Brief" — full protocol
            mode_instruction = (
                "\n\nRESPONSE FORMAT — FULL PROTOCOL: Provide a comprehensive "
                "clinical response structured with markdown headings (##) as:\n"
                "## Etiology / Pathophysiology\n"
                "## Diagnostic Criteria\n"
                "## Treatment (First-line and Second-line with exact dosages)\n"
                "## Contraindications and Drug Interactions\n"
                "## Follow-up and Monitoring\n"
                "For any section where the retrieved context provides no data, "
                "write: 'Not specified in the available clinical guidelines.'"
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
