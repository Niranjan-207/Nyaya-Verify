import os
from typing import List, Dict, Any
import chromadb
import torch
from sentence_transformers import SentenceTransformer

class ChromaDataStore:
    def __init__(self, persist_directory="data/chroma_db", collection_name="nyaya_legal", model_name="BAAI/bge-small-en-v1.5"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] ChromaDataStore embedding device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        
    def ingest(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0
            
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        for m in metadatas:
            if m.get("statute_year") is None:
                m["statute_year"] = "Unknown"
            
        embeddings = self.model.encode(texts).tolist()
        ids = [f'{m["filename"]}_{m["page"]}_{i}' for i, m in enumerate(metadatas)]
        
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return len(texts)

    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        import re
        if k is None:
            k = 5
            query_lower = query.lower()
            words = set(re.findall(r'\b\w+\b', query_lower))
            
            keywords = {"and", "or", "under", "between", "versus", "compare", "difference"}
            keyword_matches = len(keywords.intersection(words))
            k += (keyword_matches * 2)
            
            years = set(re.findall(r'\b(?:18|19|20)\d{2}\b', query))
            acts = set(re.findall(r'\b(?:ipc|bns|bnss|rti|act|code|sanhita)\b', query_lower))
            
            if len(years) >= 2 or len(acts) >= 2:
                k += 2
                
            k = min(k, 12)
            print(f"Dynamic K selected: {k}")

        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        chunks = []
        if results and results["documents"] and len(results["documents"]) > 0:
            doc_array = results["documents"][0]
            meta_array = results["metadatas"][0]
            for i in range(len(doc_array)):
                chunks.append({
                    "text": doc_array[i],
                    "metadata": meta_array[i]
                })
        return chunks
