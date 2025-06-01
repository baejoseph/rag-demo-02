import fitz  # PyMuPDF
from typing import List
from datetime import datetime
from openai_services import OpenAIEmbeddingService
from rag_pipeline import DocumentChunk, DocumentMetadata

# === Document Parser Class ===
class DocumentParser:
    def __init__(self, embedding_service: OpenAIEmbeddingService):
        self.embedding_service = embedding_service

    def parse_pdf(self, file) -> List[DocumentChunk]:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue

            metadata = DocumentMetadata(
                file_name=file.name,
                file_version="v1",
                file_date=datetime.now(),
                section_number=str(page_num + 1),
                page=page_num + 1,
                document_id=file.name
            )

            # Naive chunking: full page per chunk
            embedding = self.embedding_service.embed_text(text)
            chunk = DocumentChunk(content=text, metadata=metadata, embedding=embedding)
            chunks.append(chunk)

        return chunks