import argparse
import sys
import os

# Add src to Python Path context to allow module resolution when running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.pdf_parser import extract_text_with_metadata
from src.ingestion.semantic_chunker import HybridHierarchicalChunker
from src.retrieval.vector_store import ChromaDataStore

def main():
    parser = argparse.ArgumentParser(description="Nyaya-Verify Legal PDF Ingestion")
    parser.add_argument("--pdf", required=True, type=str, help="Path to the source PDF document.")
    parser.add_argument("--year", required=False, type=int, default=None, help="Explicit statute year (Optional/Fallback).")
    args = parser.parse_args()
    
    print(f"[*] Ingestion Started")
    print(f"[+] Loading PDF Target: {args.pdf}")
    
    try:
        blocks = extract_text_with_metadata(args.pdf, args.year)
        print(f"[+] Extracted {len(blocks)} raw text blocks from PyMuPDF.")
        
        print(f"[*] Applying Hybrid Hierarchical Chunking logic...")
        chunker = HybridHierarchicalChunker()
        chunks = chunker.chunk(blocks)
        print(f"[+] Output: {len(chunks)} formal semantic chunks generated.")
        
        print(f"[*] Upserting to persistent ChromaDB store...")
        store = ChromaDataStore()
        stored_count = store.ingest(chunks)
        
        print(f"==================================================")
        print(f"SUCCESS: {stored_count} chunks successfully embedded arrayed and committed to Vector DB.")
        print(f"==================================================")
        
    except Exception as e:
        print(f"[!] Critical Error during ingestion pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
