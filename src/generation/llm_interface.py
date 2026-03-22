import ollama
import yaml
import os

# Load configuration
_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
with open(_config_path, 'r') as f:
    _config = yaml.safe_load(f)
model_name_default = _config.get('model_name', 'llama3.2:3b')
from typing import List, Dict, Any

class LegalSynthesizer:
    def __init__(self, model_name=model_name_default):
        self.model_name = model_name
        self.client = ollama.Client()
        
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown")
            page = meta.get("page", "Unknown")
            year = meta.get("statute_year", "Unknown")
            
            c_text = f"[Citation {i}: {filename} (Year: {year}), Page {page}]\n{chunk['text']}"
            formatted_chunks.append(c_text)
            
        return "\n\n".join(formatted_chunks)

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]], conflict_payload: Dict[str, Any]) -> str:
        logic_flip_alert = ""
        
        # Inject warning payload natively if contradictory axioms were discovered
        if conflict_payload.get("ConflictFound") is True:
            prioritized = conflict_payload.get("PrioritizedChunk", {}).get("metadata", {})
            rejected = conflict_payload.get("RejectedChunk", {}).get("metadata", {})
            
            doc_a = rejected.get("filename", "Unknown")
            doc_b = prioritized.get("filename", "Unknown")
            year_prioritized = prioritized.get("statute_year", "Unknown")
            
            logic_flip_alert = f"⚠️ **LOGIC FLIP DETECTED:** Contradiction found between [{doc_a}] and [{doc_b}]. Prioritizing the [{year_prioritized}] mandate."

        context_str = self._format_context(chunks)
        
        system_prompt = (
            "You are an expert Indian Legal AI Assistant named Nyaya-Verify. "
            "You must answer the user's query STRICTLY based on the provided <context> text below. "
            "If the answer cannot be found in the context blocks, clearly state that you do not have enough "
            "information. DO NOT hallucinate, invent precedents, or use outside knowledge. "
            "Always append the specific citation tag (e.g. '[Citation 1]') directly inline where appropriate."
        )

        user_prompt = f"""
{logic_flip_alert}

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
            stream=False
        )
        
        # The logic_flip_alert is passed to prompt payload and echoed to end user interface
        if logic_flip_alert:
            return logic_flip_alert + "\n\n" + response['response']
        return response['response']
