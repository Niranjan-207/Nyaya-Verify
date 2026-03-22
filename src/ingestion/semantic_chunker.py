import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import scipy.spatial.distance as distance

class HybridHierarchicalChunker:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", window_size=5, distance_multiplier=1.5):
        self.model = SentenceTransformer(model_name, device='cuda')
        self.window_size = window_size
        self.distance_multiplier = distance_multiplier
        
        # Layer 1: Structural Boundaries (match from beginning of string)
        self.structural_regex = re.compile(r'^(?:\d+\.|Section \d+|CHAPTER)', re.IGNORECASE)

    def is_structural_boundary(self, text: str) -> bool:
        """Determines if the text block demarcates a new section or chapter."""
        if self.structural_regex.match(text):
            return True
        return False

    def chunk(self, parsed_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Layer 1: Group physical blocks into Structural Units
        structural_units = []
        current_unit = None
        
        for block in parsed_blocks:
            text = block["text"]
            is_boundary = self.is_structural_boundary(text)
            
            # Additional constraint: Provided that / Explanation.— MUST stay appended to preceding logic
            if text.startswith("Provided that") or text.startswith("Explanation.—"):
                is_boundary = False
                
            if current_unit is None:
                current_unit = {"text": [block["text"]], "metadata": block["metadata"]}
            elif is_boundary:
                structural_units.append(current_unit)
                # Restart unit holding previous block's metadata context
                current_unit = {"text": [block["text"]], "metadata": block["metadata"]}
            else:
                current_unit["text"].append(block["text"])
                
        if current_unit is not None:
            structural_units.append(current_unit)
            
        # Layer 2: Semantic Chunking strictly inside structural units
        final_chunks = []
        
        for unit in structural_units:
            # Naive sentence split to prepare items for the vector comparison array
            sentences = " ".join(unit["text"]).split(". ")
            sentences = [s.strip() + "." if not s.endswith(".") else s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                continue
                
            embeddings = self.model.encode(sentences)
            
            current_chunk_sentences = [sentences[0]]
            rolling_distances = []
            
            for i in range(1, len(sentences)):
                dist = distance.cosine(embeddings[i-1], embeddings[i])
                
                # Check threshold against rolling history
                if len(rolling_distances) > 0:
                    rolling_avg = np.mean(rolling_distances[-self.window_size:])
                    threshold = rolling_avg * self.distance_multiplier
                    
                    if dist > threshold:
                        # Break chunk, distance signifies isolated semantic gap
                        final_chunks.append({
                            "text": " ".join(current_chunk_sentences),
                            "metadata": unit["metadata"]
                        })
                        current_chunk_sentences = []
                
                current_chunk_sentences.append(sentences[i])
                rolling_distances.append(dist)
                
            if current_chunk_sentences:
                final_chunks.append({
                    "text": " ".join(current_chunk_sentences),
                    "metadata": unit["metadata"]
                })
                
        return final_chunks
