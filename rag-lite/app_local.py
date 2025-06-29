import streamlit as st
from local_services import LocalEmbeddingService, LocalGenerationService, LocalCacheService
from rag_pipeline import (
    Corpus, RetrievalService, PromptAugmenter, QueryProcessor,
    ProcessorConfig, RetrievalConfig, CosineSimilarity
)
from parser_local import DocumentParser
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# === Streamlit Setup ===
st.set_page_config(page_title="RAG Chat MVP (Local)", layout="wide")
st.title("🔍📚 Retrieval-Augmented Chatbot (Local Demo)")

# === Local Cache & Model Setup ===
cache_dir    = os.getenv("LOCAL_CACHE_DIR", "local_cache")
bucket_name  = os.getenv("LOCAL_CACHE_BUCKET", "default")
  
# === Session State Initialization ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "corpus" not in st.session_state:
    st.session_state.corpus = Corpus()

if "base_services" not in st.session_state:
    with st.spinner("Initializing local models…"):
        embedding_service = LocalEmbeddingService()
        generation_service = LocalGenerationService()
        similarity_metric  = CosineSimilarity()
        retrieval_service  = RetrievalService(
            st.session_state.corpus,
            similarity_metric
        )
        augmenter     = PromptAugmenter('rag_prompt.md')
        cache_service = LocalCacheService(cache_dir)

        st.session_state.base_services = {
            "embedding_service": embedding_service,
            "generation_service": generation_service,
            "retrieval_service": retrieval_service,
            "augmenter": augmenter,
            "cache_service": cache_service,
        }
    st.success(f"Local LLM initialized: {generation_service.model}", icon="✅")

# === File Upload ===
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload docx file")
uploaded_file = st.sidebar.file_uploader("Upload a docx (<2MB)", type="docx")

if uploaded_file and uploaded_file.size < 2 * 1024 * 1024:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    with st.spinner("Parsing…"):
        parser = DocumentParser(
            st.session_state.base_services["embedding_service"],
            st.session_state.base_services["cache_service"],
            bucket_name
        )
        new_chunks = parser.parse_docx(uploaded_file)
    st.sidebar.success("Document parsed", icon="✅")

    # Add to corpus
    added = st.session_state.corpus.add_chunks(new_chunks)
    if added:
        st.sidebar.success(f"Added {added} chunks")
    else:
        st.sidebar.info("No new chunks (already cached)")
    st.sidebar.info(f"Total chunks: {len(st.session_state.corpus.get_all_chunks())}")

elif uploaded_file:
    st.sidebar.error("File too large. Must be <2MB.")

# === Retrieval Settings ===
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Retrieval Settings")
top_k               = st.sidebar.slider("Top K Chunks", 1, 10, 3)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.6)

# === Chat Input ===
user_input = st.chat_input("Ask a question…")
if user_input:
    with st.spinner("Generating response…"):
        config = ProcessorConfig(
            retrieval=RetrievalConfig(
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        )
        processor = QueryProcessor(
            corpus=st.session_state.corpus,
            embedding_service=st.session_state.base_services["embedding_service"],
            retrieval_service=st.session_state.base_services["retrieval_service"],
            prompt_augmenter=st.session_state.base_services["augmenter"],
            generation_service=st.session_state.base_services["generation_service"],
            config=config
        )
        answer = processor.process_query(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": answer})

# === Display Chat ===
for exchange in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(exchange["user"])
    with st.chat_message("assistant"):
        st.markdown(exchange["bot"])
