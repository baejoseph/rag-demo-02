import os
import re
import json
import hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime

import mammoth
import html2text            # pip install html2text
from docx import Document as DocxDocument

from rag_pipeline import DocumentChunk, DocumentMetadata
from logger import logger


class DocumentParser:
    def __init__(self, embedding_service, cache_service, bucket_name):
        self.embedding_service = embedding_service
        self.cache = cache_service
        self.bucket = bucket_name

        # cache root for all artifacts
        self.cache_root = "local_cache"
        os.makedirs(self.cache_root, exist_ok=True)
        logger.info("DocumentParser initialized with local cache bucket: %s", self.bucket)

    def _hash_docx_metadata(self, doc: DocxDocument) -> str:
        p = doc.core_properties
        key_data = f"{p.title}-{p.modified}-{p.author}"
        return hashlib.sha256(key_data.encode("utf-8")).hexdigest()

    def _serialize_chunk(self, chunk: DocumentChunk) -> Dict[str, Any]:
        return {
            "content": chunk.content,
            "metadata": {
                "file_name":      chunk.metadata.file_name,
                "file_version":   chunk.metadata.file_version,
                "file_date":      chunk.metadata.file_date.isoformat(),
                "section_number": chunk.metadata.section_number,
                "section_heading":chunk.metadata.section_heading,
                "document_id":    chunk.metadata.document_id,
            },
            "embedding": chunk.embedding,
        }

    def _reconstruct_chunk_from_dict(self, d: Dict[str, Any]) -> DocumentChunk:
        m = d["metadata"]
        if isinstance(m, str):
            logger.warning("Old cache format — resetting metadata")
            return DocumentChunk(
                content=d["content"],
                metadata=DocumentMetadata(
                    file_name="", file_version="v1",
                    file_date=datetime.now(), section_number="1",
                    section_heading="", document_id=""
                ),
                embedding=d["embedding"],
            )
        m["file_date"] = datetime.fromisoformat(m["file_date"])
        meta = DocumentMetadata(**m)
        return DocumentChunk(content=d["content"], metadata=meta, embedding=d["embedding"])

    def _process_heading(self, text: str) -> Tuple[str, str, str]:
        match = re.match(r"^(#{1,6})\s+(.+?)(?:\n|$)(.*)", text.strip(), re.DOTALL)
        if match:
            lvl = len(match.group(1))
            heading = match.group(2).strip()
            body = (heading + "\n" + match.group(3)).strip()
            return body, str(lvl), heading
        return text.strip(), "", ""

    def parse_docx(self, docx_file) -> List[DocumentChunk]:
        # —————————————————————
        # 1) Hash & cache paths
        # —————————————————————
        file_name = getattr(docx_file, "name", "uploaded.docx")
        docx_obj  = DocxDocument(docx_file)
        doc_hash  = self._hash_docx_metadata(docx_obj)
        prefix    = f"cache/{doc_hash}/"
        chunks_key, md_key, up_key = (
            prefix + "chunks.json",
            prefix + "converted.md",
            prefix + "uploaded.docx",
        )

        # —————————————————————
        # 2) Try loading existing chunks
        # —————————————————————
        try:
            resp = self.cache.get_object(Bucket=self.bucket, Key=chunks_key)
            raw  = resp["Body"].read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            chunk_dicts = json.loads(raw)
            logger.info("Loaded %d chunks from cache", len(chunk_dicts))
            return [self._reconstruct_chunk_from_dict(cd) for cd in chunk_dicts]

        except FileNotFoundError:
            logger.info("No cached chunks for hash %s, re-parsing", doc_hash)

        # —————————————————————
        # 3) Save & “upload” the .docx itself
        # —————————————————————
        local_docx = os.path.join(self.cache_root, f"{doc_hash}_uploaded.docx")
        docx_obj.save(local_docx)
        self.cache.upload_file(Filename=local_docx, Bucket=self.bucket, Key=up_key)

        # fallback date
        core = docx_obj.core_properties
        file_date = core.modified or datetime.now()

        # —————————————————————
        # 4) Convert → clean Markdown
        # —————————————————————
        with open(local_docx, "rb") as f:
            html = mammoth.convert_to_html(f).value

        # use html2text for much cleaner markdown
        converter = html2text.HTML2Text()
        converter.body_width = 0       # no forced wraps
        markdown = converter.handle(html)

        # write out for inspection & cache
        local_md = os.path.join(self.cache_root, f"{doc_hash}_converted.md")
        with open(local_md, "w", encoding="utf-8") as f:
            f.write(markdown)
        self.cache.upload_file(Filename=local_md, Bucket=self.bucket, Key=md_key)
        logger.info("Converted DOCX→HTML→Markdown and cached")

        # —————————————————————
        # 5) Chunk + embed with heading-based numbering
        # —————————————————————
        pattern    = r"(?=^#{1,3} .*)"  # Split on headings (up to ###)
        raw_chunks = re.split(pattern, markdown, flags=re.MULTILINE)
        logger.info("Split markdown into %d initial chunks", len(raw_chunks))

        chunks: List[DocumentChunk] = []
        section_counter = {}
        current_heading = ""

        for chunk in raw_chunks:
            if not chunk.strip():
                continue

            # extract the clean text, heading level, and heading text
            chunk_text, level_str, heading = self._process_heading(chunk)

            if level_str:
                # this is a new heading
                level = int(level_str)
                # increment or initialize this level’s counter
                section_counter[level] = section_counter.get(level, 0) + 1
                # drop any deeper-level counters
                for lvl in list(section_counter):
                    if lvl > level:
                        del section_counter[lvl]
                # build section number from counters up to current level
                section_numbers = [
                    str(section_counter[l]) 
                    for l in sorted(section_counter) 
                    if l <= level
                ]
                section_number  = ".".join(section_numbers)
                current_heading = heading
            else:
                # carry forward numbering if no new heading
                section_number = ".".join(
                    str(section_counter[l]) 
                    for l in sorted(section_counter)
                )

            estimated_tokens = len(chunk_text) // 4
            logger.debug(
                "Processing chunk with ~%d tokens from section %s: %s",
                estimated_tokens, section_number, current_heading
            )

            # get embedding (assuming your service has embed_text)
            embedding = self.embedding_service.embed_text(chunk_text)

            metadata = DocumentMetadata(
                file_name       = file_name,
                file_version    = "v1",
                file_date       = file_date,
                section_number  = section_number,
                section_heading = current_heading,
                document_id     = doc_hash,
            )
            chunks.append(DocumentChunk(
                content   = chunk_text,
                metadata  = metadata,
                embedding = embedding
            ))

        logger.info("Successfully processed %d chunks with embeddings", len(chunks))


        # —————————————————————
        # 6) Cache the new chunks
        # —————————————————————
        ser = [self._serialize_chunk(c) for c in chunks]
        local_json = os.path.join(self.cache_root, f"{doc_hash}_chunks.json")
        with open(local_json, "w", encoding="utf-8") as f:
            json.dump(ser, f, indent=2)

        self.cache.upload_file(Filename=local_json, Bucket=self.bucket, Key=chunks_key)
        logger.info("Stored %d chunks in cache %s", len(ser), chunks_key)

        return chunks
