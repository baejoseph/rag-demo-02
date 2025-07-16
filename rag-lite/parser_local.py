import os
import re
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import mammoth
import html2text            # pip install html2text
from docx import Document as DocxDocument

from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

from rag_pipeline import DocumentChunk, DocumentMetadata
from logger import logger
from log_time import log_time


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

    @log_time("Parsing Docx or Loading Cached")
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
            embedding = list(self.embedding_service.embed_text(chunk_text))

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

    @log_time("Pre-processing markdown")
    def preprocess_markdown(self, markdown_text: str) -> Tuple[str, str, str, datetime]:
        """
        Clean up Marker-PDF markdown for smart chunking and extract file metadata:
        - extract file version and file date from lines like 'vX.Y.Z - DD-MM-YYYY'
        - remove <span> tags
        - remove ** wrappers around section numbers
        - drop image-only lines
        - extract the first H1 as the document title
        - remove all content before the first numeric section heading (level 2)
        - normalize section headings:
          * level-1 sections (e.g. 1,2,3...) => '# '
          * level-2 sections (e.g. 1.1,1.2,1.3...) => '## '
          * sub-fields (e.g. Profile Applicability, Description) => '### '
        Returns:
            processed_text (str), file_version (e.g. 'v1.0.0'), file_date (datetime)
        """
        # 1) Extract version and date
        version = "v1"
        file_date = datetime.now()
        for line in markdown_text.splitlines():
            m = re.match(r'^\s*v(\d+(?:\.\d+)*)\s*-\s*(\d{2}-\d{2}-\d{4})', line)
            if m:
                version = f"v{m.group(1)}"
                try:
                    file_date = datetime.strptime(m.group(2), "%d-%m-%Y")
                except ValueError:
                    file_date = datetime.now()
                break

        # 2) Remove span tags
        cleaned = re.sub(r'</?span[^>]*>', '', markdown_text)

        # 3) Remove bold wrappers around numeric headings
        cleaned = re.sub(r'\*\*(\d+(?:\.\d+)*.*?)\*\*', r'\1', cleaned)

        # 4) Split into lines
        lines = cleaned.splitlines()

        # 5) Extract document title from first H1
        title = ""
        for ln in lines:
            m = re.match(r'^#\s*(.+)', ln)
            if m:
                title = m.group(1).strip()
                break

        # 6) Find start index at first level-2 numeric section '## 1', '## 1.1', etc.
        start_idx: Optional[int] = None
        for idx, ln in enumerate(lines):
            s = ln.strip()
            if re.match(r'^##\s*\d+(?:\.\d+)?\s+', s):
                start_idx = idx
                break
        if start_idx is None:
            return markdown_text, version, file_date

        # 7) Build processed lines
        output = [f'TITLE OF DOCUMENT: {title}', '']
        for ln in lines[start_idx:]:
            stripped = ln.strip()
            # Skip image‐only lines
            if re.match(r'^!\[.*\]\(.*\)', stripped):
                continue
            if stripped.startswith('#'):
                # Remove leading hashes
                content = re.sub(r'^#{1,6}\s*', '', stripped).strip()
                # Remove any surrounding asterisks or underscores
                content = content.strip('*_ ')

                # All numbered sections/subsections (1, 1.3, 1.5.3, 5.3.2.5, etc.) => '##'
                if re.match(r'^\d+(?:\.\d+)*\s+', content):
                    prefix = '## '
                else:
                    prefix = '### '

                output.append(f'{prefix}{content}')
            else:
                output.append(ln)

        processed_text = "\n".join(output)
        return processed_text, title, version, file_date

    @log_time("Parsing PDF or Loading Cached")
    def parse_pdf(self, pdf_file) -> List[DocumentChunk]:
        # 1) Determine file name
        if isinstance(pdf_file, (str, Path)):
            file_path = str(pdf_file)
            file_name = os.path.basename(file_path)
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
        else:
            file_name = getattr(pdf_file, 'name', 'uploaded.pdf')
            pdf_bytes = pdf_file.read()

        # 2) Compute document hash for caching
        doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
        prefix = f'cache/{doc_hash}/'
        chunks_key = prefix + 'chunks.json'
        md_key = prefix + 'converted.md'
        up_key = prefix + 'uploaded.pdf'

        # 3) Try loading from cache
        try:
            resp = self.cache.get_object(Bucket=self.bucket, Key=chunks_key)
            raw = resp['Body'].read()
            raw = raw.decode('utf-8') if isinstance(raw, bytes) else raw
            chunk_dicts = json.loads(raw)
            logger.info('Loaded %d chunks from cache', len(chunk_dicts))
            return [self._reconstruct_chunk_from_dict(cd) for cd in chunk_dicts]
        except FileNotFoundError:
            logger.info('No cached chunks for hash %s, parsing afresh', doc_hash)

        # 4) Save and upload PDF
        local_pdf = os.path.join(self.cache_root, f'{doc_hash}_uploaded.pdf')
        with open(local_pdf, 'wb') as f:
            f.write(pdf_bytes)
        self.cache.upload_file(Filename=local_pdf, Bucket=self.bucket, Key=up_key)

        # 5) Convert via Marker
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(local_pdf)
        raw_md = rendered.markdown

        # 6) Preprocess markdown
        processed_md, title, file_version, file_date = self.preprocess_markdown(raw_md)

        # 7) Cache processed markdown
        local_md = os.path.join(self.cache_root, f'{doc_hash}_converted.md')
        with open(local_md, 'w', encoding='utf-8') as f:
            f.write(processed_md)
        self.cache.upload_file(Filename=local_md, Bucket=self.bucket, Key=md_key)
        logger.info('Converted and preprocessed PDF→Markdown')

        # 8) Split into chunks on '## ' (numeric sections)
        pattern = r'(?=^##\s+)'
        raw_chunks = re.split(pattern, processed_md, flags=re.MULTILINE)
        logger.info('Split markdown into %d initial chunks', len(raw_chunks))

        chunks: List[DocumentChunk] = []
        for chunk in raw_chunks:
            if not chunk.strip().startswith('## '):
                continue
            lines = chunk.splitlines()
            # parse section heading
            heading_line = lines[0][3:].strip()  # remove '## '
            parts = heading_line.split(' ', 1)
            section_number = parts[0]
            section_heading = parts[1] if len(parts) > 1 else ''
            # body is everything after the heading line
            chunk_body = '\n'.join(lines[1:]).strip()

            # Append title and section number and heading to chunk
            preamble = f"File: {title}, Version {file_version}\n"
            preamble += f"Section: {section_number} {section_heading}\n\n"
            chunk_body = preamble + chunk_body

            # embedding
            embedding = list(self.embedding_service.embed_text(chunk_body))

            # metadata
            metadata = DocumentMetadata(
                file_name=file_name,
                file_version=file_version,
                file_date=file_date,
                section_number=section_number,
                section_heading=section_heading,
                document_id=doc_hash,
            )
            chunks.append(DocumentChunk(content=chunk_body, metadata=metadata, embedding=embedding))

        logger.info('Processed %d chunks with embeddings', len(chunks))

        # 9) Cache chunks to JSON
        serialized = [self._serialize_chunk(c) for c in chunks]
        local_json = os.path.join(self.cache_root, f'{doc_hash}_chunks.json')
        with open(local_json, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2)
        self.cache.upload_file(Filename=local_json, Bucket=self.bucket, Key=chunks_key)
        logger.info('Stored %d chunks in cache %s', len(serialized), chunks_key)

        return chunks
