from dataclasses import dataclass
from typing import List, Protocol, Optional
from datetime import datetime
from logger import logger

# === Data Classes ===

@dataclass
class DocumentMetadata:
    file_name: str
    file_version: str
    file_date: datetime
    section_number: str
    page: int
    document_id: Optional[str] = None  # Optional trace ID

@dataclass
class DocumentChunk:
    content: str
    metadata: DocumentMetadata
    embedding: List[float]

@dataclass
class Query:
    text: str
    embedding: List[float]

@dataclass
class RetrievalConfig:
    top_k: int
    similarity_threshold: float

@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    similarity_score: float

@dataclass
class ProcessorConfig:
    retrieval: RetrievalConfig

# === Core Corpus ===

class Corpus:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []

    def add_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks.append(chunk)

    def get_all_chunks(self) -> List[DocumentChunk]:
        return self.chunks

# === Service Interfaces ===

class EmbeddingService(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

class SimilarityMetric(Protocol):
    def compute(self, a: List[float], b: List[float]) -> float:
        ...

# === Example Similarity Metric ===

class CosineSimilarity:
    def compute(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-10)

# === Retrieval ===

class RetrievalService:
    def __init__(self, corpus: Corpus, similarity_metric: SimilarityMetric):
        self.corpus = corpus
        self.similarity_metric = similarity_metric
        logger.info("Initialized RetrievalService")

    def retrieve_similar_chunks(
        self, query: Query, config: RetrievalConfig
    ) -> List[RetrievedChunk]:
        results = []
        all_chunks = self.corpus.get_all_chunks()
        logger.info("Searching through %d chunks in corpus", len(all_chunks))
        
        for chunk in all_chunks:
            score = self.similarity_metric.compute(query.embedding, chunk.embedding)
            if score >= config.similarity_threshold:
                results.append(RetrievedChunk(chunk=chunk, similarity_score=score))
        
        results.sort(key=lambda rc: rc.similarity_score, reverse=True)
        results = results[:config.top_k]
        
        logger.info("Retrieved %d chunks above similarity threshold %.2f", 
                   len(results), config.similarity_threshold)
        for rc in results:
            logger.debug("Retrieved chunk from section %s with score %.3f", 
                        rc.chunk.metadata.section_number, rc.similarity_score)
        
        return results

# === Prompt Augmentation ===

class PromptAugmenter:
    def augment_query(self, query: Query, retrieved_chunks: List[RetrievedChunk]) -> str:
        logger.info("Augmenting query with %d retrieved chunks", len(retrieved_chunks))
        
        prompt = query.text + "\n\nContext:\n"
        for rc in retrieved_chunks:
            md = rc.chunk.metadata
            context_line = f"[{md.file_name} p{md.section_number}]: {rc.chunk.content}"
            prompt += context_line + "\n"
            
        estimated_tokens = len(prompt) // 4
        logger.debug("Generated augmented prompt with ~%d tokens", estimated_tokens)
        return prompt + "\n\nAnswer the question based on the context provided, and always cite the section number in the format e.g. (Section 1.2)."

# === Generation Service ===

class GenerationService(Protocol):
    def generate_response(self, augmented_prompt: str) -> str:
        ...

# === Query Processor ===

class QueryProcessor:
    def __init__(
        self,
        corpus: Corpus,
        embedding_service: EmbeddingService,
        retrieval_service: RetrievalService,
        prompt_augmenter: PromptAugmenter,
        generation_service: GenerationService,
        config: ProcessorConfig
    ):
        self.corpus = corpus
        self.embedding_service = embedding_service
        self.retrieval_service = retrieval_service
        self.prompt_augmenter = prompt_augmenter
        self.generation_service = generation_service
        self.config = config
        logger.info("Initialized QueryProcessor with config: %s", config)

    def process_query(self, query_text: str) -> str:
        if not query_text.strip():
            raise ValueError("Query text cannot be empty.")

        logger.info("Processing query: %s", query_text)
        query_embedding = self.embedding_service.embed_text(query_text)
        query = Query(text=query_text, embedding=query_embedding)

        retrieved_chunks = self.retrieval_service.retrieve_similar_chunks(
            query, self.config.retrieval
        )

        augmented_prompt = self.prompt_augmenter.augment_query(query, retrieved_chunks)
        response = self.generation_service.generate_response(augmented_prompt)
        logger.info("Query processing completed")
        return response
