import argparse
import sys
import os
import glob
import chromadb

# Add src to Python Path context to allow module resolution when running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.pdf_parser import extract_text_with_metadata
from src.ingestion.semantic_chunker import HybridHierarchicalChunker
from src.retrieval.vector_store import ChromaDataStore

def get_ordered_pdfs(pdf_dir):
    priority_order = [
        "Indian_Penal_Code_1860.pdf",
        "Bharatiya_Nyaya_Sanhita_2023.pdf",
        "Code_of_Criminal_Procedure_1973.pdf",
        "Bharatiya_Nagarik_Suraksha_Sanhita_2023.pdf",
        "Indian_Evidence_Act_1872.pdf",
        "Bharatiya_Sakshya_Adhiniyam_2023.pdf"
    ]
    
    all_pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    all_pdf_names = {os.path.basename(p): p for p in all_pdfs}
    
    ordered_paths = []
    
    # Insert priority files in exact order if they exist
    for name in priority_order:
        if name in all_pdf_names:
            ordered_paths.append(all_pdf_names[name])
            del all_pdf_names[name]
            
    # Append the remaining files alphabetically
    remaining_names = sorted(list(all_pdf_names.keys()))
    for name in remaining_names:
        ordered_paths.append(all_pdf_names[name])
        
    return ordered_paths

def main():
    parser = argparse.ArgumentParser(description="Nyaya-Verify Bulk Directory Ingestion")
    parser.add_argument("--pdf-dir", required=True, type=str, help="Directory containing the target PDFs.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.pdf_dir):
        print(f"[!] Directory '{args.pdf_dir}' not found.")
        sys.exit(1)
        
    # Phase 2: Complete Clean Slate Reset
    print("[*] Initializing explicit clean state for ChromaDB store...")
    client = chromadb.PersistentClient(path="data/chroma_db")
    try:
        client.delete_collection("nyaya_legal")
        print("[+] Successfully deleted existing collection 'nyaya_legal'.")
    except Exception:
        print("[-] Collection 'nyaya_legal' did not exist yet. Proceeding cleanly.")
        
    print("[*] Spin up hardware embeddings pipelines on CUDA...")
    store = ChromaDataStore(persist_directory="data/chroma_db", collection_name="nyaya_legal", model_name="BAAI/bge-small-en-v1.5")
    chunker = HybridHierarchicalChunker(model_name="BAAI/bge-small-en-v1.5")
    
    ordered_pdfs = get_ordered_pdfs(args.pdf_dir)
    print(f"[*] Total {len(ordered_pdfs)} PDFs found to process in queue.")
    
    total_chunks = 0
    total_pdfs = 0
    
    for pdf_path in ordered_pdfs:
        filename = os.path.basename(pdf_path)
        print(f"\n[>] Processing '{filename}'...")
        
        try:
            # pdf_parser automatically extracts \d{4} if a year parameter isn't supplied natively
            blocks = extract_text_with_metadata(pdf_path)
            chunks = chunker.chunk(blocks)
            stored = store.ingest(chunks)
            total_chunks += stored
            total_pdfs += 1
            print(f"   [+] Processed. {stored} structured semantic chunks embedded and safely stored.")
        except Exception as e:
            print(f"   [!] Critical Error processing '{filename}': {str(e)}")
            continue
            
    print("\n" + "="*60)
    print(" INGESTION WORKFLOW COMPLETE")
    print("="*60)
    print(f" Total PDFs Processed: {total_pdfs} / {len(ordered_pdfs)}")
    print(f" Total System Vector Chunks Generated: {total_chunks}")
    print("="*60)

if __name__ == "__main__":
    main()
