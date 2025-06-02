import os
import re
import subprocess
import json
import hashlib
from typing import List
from datetime import datetime
from docx import Document as DocxDocument
from rag_pipeline import DocumentChunk, DocumentMetadata

class DocumentParser:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.cache_root = "local_cache"
        os.makedirs(self.cache_root, exist_ok=True)

    def _hash_docx_metadata(self, docx_file: DocxDocument) -> str:
        core_props = docx_file.core_properties
        key_data = f"{core_props.title}-{core_props.modified}-{core_props.author}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def parse_docx(self, docx_file: DocxDocument) -> List[DocumentChunk]:
        
        docx_file = DocxDocument(docx_file)
        file_hash = self._hash_docx_metadata(docx_file)
        cache_dir = os.path.join(self.cache_root, file_hash)
        os.makedirs(cache_dir, exist_ok=True)

        file_path = os.path.join(cache_dir, "uploaded.docx")
        chunks_path = os.path.join(cache_dir, "chunks.json")
        markdown_path = os.path.join(cache_dir, "converted.md")

        # Return from cache if exists
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunk_dicts = json.load(f)
            return [DocumentChunk(**chunk) for chunk in chunk_dicts]

        # Save file to file_path for pandoc subprocess
        docx_file.save(file_path)
        
        # Load file again for fresh metadata
        core_props = docx_file.core_properties
        file_date = core_props.modified or datetime.now()

        # Convert DOCX to Markdown using pandoc
        subprocess.run([
            "pandoc", file_path,
            "--from=docx",
            "--to=markdown",
            "--output", markdown_path,
            "--wrap=none",
            "--markdown-headings=atx",
            "--shift-heading-level-by=0"
        ], check=True)

        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown = f.read()

        pattern = r"(?=^#{1,3} .*)"  # Split on headings (up to ###)
        raw_chunks = re.split(pattern, markdown, flags=re.MULTILINE)

        chunks: List[DocumentChunk] = []
        section_counter = {}

        for chunk in raw_chunks:
            if not chunk.strip():
                continue

            heading_match = re.match(r"^(#{1,6}) (.+)", chunk.strip())
            if heading_match:
                level = len(heading_match.group(1))
                if level not in section_counter:
                    section_counter[level] = 1
                else:
                    section_counter[level] += 1
                for k in list(section_counter.keys()):
                    if k > level:
                        del section_counter[k]
                section_numbers = [str(section_counter[l]) for l in sorted(section_counter.keys()) if l <= level]
                section_number = ".".join(section_numbers)
            else:
                section_number = ".".join(str(section_counter[l]) for l in sorted(section_counter.keys()))

            embedding = self.embedding_service.embed_text(chunk)
            metadata = DocumentMetadata(
                file_name=docx_file.core_properties.title,
                file_version="v1",
                file_date=file_date,
                section_number=section_number,
                page=0,
                document_id=docx_file.core_properties.title,
            )
            chunks.append(DocumentChunk(content=chunk.strip(), metadata=metadata, embedding=embedding))

        # Save processed chunks to cache
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([chunk.__dict__ for chunk in chunks], f, default=str, indent=2)

        return chunks
