import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.vector_store import ChromaDataStore
from src.audit.nli_judge import PairwiseAuditor
from src.generation.llm_interface import LegalSynthesizer

def main():
    parser = argparse.ArgumentParser(description="Nyaya-Verify V-RAG Query Interface")
    parser.add_argument("--query", required=True, type=str, help="Legal question to ask the system.")
    args = parser.parse_args()
    
    print(f"[*] Executing Query: {args.query}")
    
    try:
        print("[*] Retrieving semantic context from ChromaDB...")
        store = ChromaDataStore()
        results = store.search(args.query)
        print(f"[+] Retrieved {len(results)} chunks.")
        
        if not results:
            print("[!] No context found. Please run ingest.py first.")
            sys.exit(1)
            
        print("[*] Auditor running DeBERTa-v3 logic flip checks...")
        auditor = PairwiseAuditor()
        conflict_payload = auditor.detect_logic_flips(results)
        
        if conflict_payload.get("ConflictFound"):
            print(f"[!] Logic Flip Detected! Prioritizing Year: {conflict_payload.get('PrioritizedChunk', {}).get('metadata', {}).get('statute_year')}")
        else:
            print("[+] No temporal logic flips detected in context bounds.")
            
        print("[*] Unloading PyTorch/DeBERTa VRAM payload explicitly...")
        auditor.unload()
        
        print("[*] Contacting Ollama Llama 3.2 3B instance...")
        synthesizer = LegalSynthesizer()
        answer = synthesizer.generate_answer(args.query, results, conflict_payload)
        
        print("\n" + "="*60)
        print(" NYAYA-VERIFY GENERATED RESPONSE")
        print("="*60 + "\n")
        print(answer)
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n[!] Critical Pipeline Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
