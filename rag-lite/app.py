import streamlit as st
from openai_services import OpenAIEmbeddingService, OpenAIGenerationService
from rag_pipeline import (
    Corpus, RetrievalService, PromptAugmenter, QueryProcessor,
    ProcessorConfig, RetrievalConfig, CosineSimilarity, DocumentChunk,
    DocumentMetadata
)
from parser import DocumentParser
from typing import List
from datetime import datetime
import os

# === Streamlit Setup ===
st.set_page_config(page_title="RAG Chat MVP", layout="wide")
st.title("🔍📚 Retrieval-Augmented Chatbot (MVP)")

# === Sidebar Configuration ===
st.sidebar.header("🔧 Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
top_k = st.sidebar.slider("Top K Chunks", 1, 10, 3)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.75)

# === Session State Initialization ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "corpus" not in st.session_state:
    st.session_state.corpus = Corpus()

if "processor" not in st.session_state and api_key:
    embedding_service = OpenAIEmbeddingService(api_key)
    generation_service = OpenAIGenerationService(api_key)
    similarity_metric = CosineSimilarity()
    retrieval_service = RetrievalService(st.session_state.corpus, similarity_metric)
    augmenter = PromptAugmenter()

    config = ProcessorConfig(
        retrieval=RetrievalConfig(top_k=top_k, similarity_threshold=similarity_threshold)
    )

    processor = QueryProcessor(
        corpus=st.session_state.corpus,
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
        prompt_augmenter=augmenter,
        generation_service=generation_service,
        config=config
    )

    st.session_state.embedding_service = embedding_service
    st.session_state.processor = processor



# === File Upload ===
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF (<2MB)", type="pdf")

if uploaded_file and uploaded_file.size < 2 * 1024 * 1024:
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    parser = DocumentParser(st.session_state.embedding_service)
    new_chunks = parser.parse_pdf(uploaded_file)

    for chunk in new_chunks:
        st.session_state.corpus.add_chunk(chunk)

    st.sidebar.info(f"Added {len(new_chunks)} chunks to corpus")
elif uploaded_file:
    st.sidebar.error("File too large. Please upload files under 2MB.")

# === Chat Input ===
user_input = st.chat_input("Ask a question...")

if user_input and api_key:
    with st.spinner("Generating response..."):
        response = st.session_state.processor.process_query(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})

# === Display Chat ===
for exchange in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(exchange["user"])
    with st.chat_message("assistant"):
        st.markdown(exchange["bot"])
