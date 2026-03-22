import os
import sys
import yaml
import streamlit as st

# Load configuration for model name
_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(_config_path, 'r') as f:
    _config = yaml.safe_load(f)
model_name = _config.get('model_name', 'llama3.2:3b')

from src.retrieval.vector_store import ChromaDataStore
from src.audit.nli_judge import PairwiseAuditor
from src.generation.llm_interface import LegalSynthesizer

st.set_page_config(page_title="Nyaya-Verify", page_icon="⚖️", layout="wide")

st.title("⚖️ Nyaya-Verify Legal V-RAG")
st.markdown("A localized Indian Legal Assistant protected by DeBERTa-v3 NLI Auditor logic.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("ChromaDB Vector Store Connected (`nyaya_legal`)")
    st.info(f"LLM Engine: Ollama `{model_name}` (4-bit)")
    st.info("Auditor: `bge-small` + `cross-encoder/nli-deberta-v3-small`")
    st.markdown("---")
    st.markdown("**Hardware Limits Maintained:** < 6GB VRAM (Mid-Run CUDA offloading)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
if query := st.chat_input("Enter your legal query here..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Retrieving semantic fragments from Vector Store..."):
            store = ChromaDataStore()
            results = store.search(query)
            
        if not results:
            response_placeholder.error("Context undefined. Please parse datasets via `ingest.py` before querying.")
            st.stop()
            
        with st.spinner("Executing Pairwise NLI Logic Flips..."):
            auditor = PairwiseAuditor()
            conflict_payload = auditor.detect_logic_flips(results)
            
        if conflict_payload.get("ConflictFound"):
            st.warning(f"⚠️ **LOGIC FLIP DETECTED:** Conflict identified between sources! Prioritizing context extracted from `{conflict_payload.get('PrioritizedChunk', {}).get('metadata', {}).get('statute_year', 'Unknown')}`.")
            
        with st.spinner("Flushing PyTorch caching allocations from GPU..."):
            # Execute necessary cleanup prior to spinning identical bounds against LLM parameters
            auditor.unload()
            
        with st.spinner("Generating synthesized local advice..."):
            synthesizer = LegalSynthesizer()
            answer = synthesizer.generate_answer(query, results, conflict_payload)
            
        response_placeholder.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        with st.expander("📚 Source Context"):
            for i, chunk in enumerate(results, 1):
                meta = chunk.get("metadata", {})
                st.markdown(f"**Citation {i}: `{meta.get('filename')}` (Year: {meta.get('statute_year')}, Page {meta.get('page')})**")
                st.write(chunk.get("text"))
