import itertools
import torch
import gc
from typing import List, Dict, Any
from sentence_transformers.cross_encoder import CrossEncoder

class PairwiseAuditor:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small"):
        self.model = CrossEncoder(model_name)
        
    def detect_logic_flips(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate candidate chunks for direct logical contradictions.
        Only compares chunks retrieved from different domains/years to avoid O(N^2) latency spikes.
        """
        # Constrain matrix to top K=5 chunks
        chunks = chunks[:5]
        
        pairs = []
        chunk_pairs = []
        
        # Build permutations exclusively if sources are unaligned temporally or physically
        for idx1, c1 in enumerate(chunks):
            for idx2, c2 in enumerate(chunks[idx1+1:], start=idx1+1):
                m1 = c1.get("metadata", {})
                m2 = c2.get("metadata", {})
                
                # Isolation block: bypass comparison if derived from the same exact source artifact and year
                if m1.get("statute_year") == m2.get("statute_year") and m1.get("filename") == m2.get("filename"):
                    continue
                    
                pairs.append((c1["text"], c2["text"]))
                chunk_pairs.append((c1, c2))
                
        if not pairs:
            return {"ConflictFound": False}
            
        scores = self.model.predict(pairs)
        
        # Transform logits to softmax probabilities
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()
        
        for i, prob_mapping in enumerate(probs):
            # DeBERTa-v3 NLI maps Contradiction to Index 0.
            contradiction_prob = prob_mapping[0]
            
            if contradiction_prob > 0.8:
                c1, c2 = chunk_pairs[i]
                
                year1 = c1.get("metadata", {}).get("statute_year")
                year2 = c2.get("metadata", {}).get("statute_year")
                
                # Graceful parsing
                try:
                    y1_val = int(year1) if year1 else 0
                except (ValueError, TypeError):
                    y1_val = 0
                try:
                    y2_val = int(year2) if year2 else 0
                except (ValueError, TypeError):
                    y2_val = 0
                
                # Conflict Resolution: Return the overriding context prioritising recency
                prioritized_chunk = c1 if y1_val >= y2_val else c2
                rejected_chunk = c2 if y1_val >= y2_val else c1
                
                return {
                    "ConflictFound": True,
                    "PrioritizedChunk": prioritized_chunk,
                    "RejectedChunk": rejected_chunk,
                    "ContradictionScore": float(contradiction_prob)
                }
                
        return {"ConflictFound": False}
        
    def unload(self):
        """
        Explicitly wipes the model footprint and rebalances VRAM arrays for LLM allocation.
        """
        if hasattr(self, 'model'):
            del self.model
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
