import torch
import re
from typing import List, Dict, Any
from src.audit.nli_judge import PairwiseAuditor

def score_faithfulness(answer: str, context_chunks: list[dict]) -> dict:
    """
    Scores the faithfulness of an answer against retrieved context chunks using DeBERTa-v3.
    """
    # 1. Clean answer entirely of citation blocks to avoid fractured sentence splitting on .pdf extensions
    answer_clean = re.sub(r'\[Citation \d+[^\]]*\]', '', answer)
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +|\n+', answer_clean) if s.strip()]
    
    sentences = []
    for s in raw_sentences:
        # Strip leading bullets/dashes
        s = re.sub(r'^[-*•]\s*', '', s).strip()
        
        s_lower = s.lower()
        # Filter system alerts
        if "logic flip" in s_lower or "⚠️" in s or "prioritizing" in s_lower:
            continue
            
        # Filter transitional phrases
        transitions = [
            "based on the provided context", 
            "please note", 
            "as follows:",
            "it's worth noting",
            "remains applicable",
            "i must note",
            "i would recommend",
            "given this information",
            "therefore, yes",
            "furthermore, as mandated"
        ]
        if any(t in s_lower for t in transitions):
            continue
            
        # Filter short sentences
        if len(s.split()) < 8:
            continue
            
        sentences.append(s)
    
    if not sentences:
        return {"faithfulness_score": 0.0, "supported": 0, "total": 0, "unsupported_sentences": []}
    
    auditor = PairwiseAuditor()
    
    supported_count = 0
    unsupported_sentences = []
    
    for sentence in sentences:
        is_supported = False
        
        # Build pairs of (Context, Sentence) -> "Does Context entail Sentence?"
        pairs = [(chunk["text"], sentence) for chunk in context_chunks]
        
        if pairs:
            # Run NLI
            scores = auditor.model.predict(pairs)
            
            # Predict returns logits. Apply softmax
            probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()
            
            # DeBERTa-v3 NLI mapping: 0=Contradiction, 1=Entailment, 2=Neutral
            for prob_mapping in probs:
                entailment_prob = prob_mapping[1]
                if entailment_prob > 0.35:
                    is_supported = True
                    break
        
        if is_supported:
            supported_count += 1
        else:
            unsupported_sentences.append(sentence)
            
    # Unload auditor to free VRAM as requested
    auditor.unload()
    
    total = len(sentences)
    score = supported_count / total if total > 0 else 0.0
    
    return {
        "faithfulness_score": score,
        "supported": supported_count,
        "total": total,
        "unsupported_sentences": unsupported_sentences
    }

if __name__ == "__main__":
    from src.retrieval.vector_store import ChromaDataStore
    from src.audit.nli_judge import PairwiseAuditor
    from src.generation.llm_interface import LegalSynthesizer
    import sys

    # Test cases:
    # 3 queries dynamically generating answers via local pipeline, then scoring logic directly
    
    test_queries = [
        "What is the punishment for theft?",
        "Can a person be arrested without a warrant?",
        "What are the rights of an accused person after arrest?"
    ]
    
    print("Initializing ChromaDataStore for Live Query Context Retrieval...")
    try:
        store = ChromaDataStore()
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        sys.exit(1)
        
    print("Beginning Custom DeBERTa-v3 Faithfulness Evaluation...\n")
    print("=" * 60)
    
    total_score = 0.0
    
    for i in range(3):
        query = test_queries[i]
        print(f"Query {i+1}: {query}")
        
        # 1. Pipeline: Retrieve Context
        live_chunks = store.search(query)
        
        # 2. Pipeline: Logic Flips
        auditor = PairwiseAuditor()
        conflict_payload = auditor.detect_logic_flips(live_chunks)
        auditor.unload()
        
        # 3. Pipeline: Generate Answer
        synthesizer = LegalSynthesizer()
        answer = synthesizer.generate_answer(query, live_chunks, conflict_payload)
        
        print(f"Generated Answer: {answer}")
        
        # 4. Pipeline: Output Evaluate Metrics
        result = score_faithfulness(answer, live_chunks)
        score = result["faithfulness_score"]
        total_score += score
        
        print(f"Supported Sentences: {result['supported']} / {result['total']}")
        print(f"Faithfulness Score: {score:.2f}")
        
        if result["unsupported_sentences"]:
            print("Unsupported Sentences found:")
            for s in result["unsupported_sentences"]:
                print(f"  - {s}")
        print("-" * 60)
        
    avg_score = total_score / 3
    print(f"Overall Average Faithfulness Score: {avg_score:.2f}")
