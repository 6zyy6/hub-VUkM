"""
Document parsing service using MinerU or DeepSeek-OCR.
Converts PDF documents to markdown and extracted images.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests

from app.core.config import settings


@dataclass
class ParsedDocument:
    """Result of document parsing."""
    document_id: str
    markdown_content: str
    images: List[Dict[str, Any]]  # [{path, page_num, description}]
    metadata: Dict[str, Any]


class DocumentParser:
    """Parses PDF documents using MinerU or DeepSeek-OCR."""

    def __init__(self, parser_type: str = "mineru"):
        self.parser_type = parser_type

    def parse(self, pdf_path: str, output_dir: str) -> ParsedDocument:
        """
        Parse a PDF document.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to store parsed outputs

        Returns:
            ParsedDocument with markdown content and extracted images
        """
        # Generate document ID from PDF content hash
        doc_id = self._generate_doc_id(pdf_path)

        if self.parser_type == "mineru":
            return self._parse_with_mineru(pdf_path, output_dir, doc_id)
        elif self.parser_type == "deepseek":
            return self._parse_with_deepseek(pdf_path, output_dir, doc_id)
        else:
            raise ValueError(f"Unknown parser type: {self.parser_type}")

    def _generate_doc_id(self, pdf_path: str) -> str:
        """Generate unique document ID from file hash."""
        with open(pdf_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash[:12]

    def _parse_with_mineru(self, pdf_path: str, output_dir: str, doc_id: str) -> ParsedDocument:
        """Parse PDF using MinerU API."""
        output_path = os.path.join(output_dir, doc_id)
        os.makedirs(output_path, exist_ok=True)

        # Call MinerU API
        with open(pdf_path, "rb") as f:
            files = {"file": ("document.pdf", f, "application/pdf")}
            data = {"output_path": output_path}
            response = requests.post(
                f"{settings.MINERU_API_URL}/parse",
                files=files,
                data=data,
                timeout=300
            )

        if response.status_code != 200:
            raise RuntimeError(f"MinerU parsing failed: {response.text}")

        result = response.json()

        # Read generated markdown
        md_path = os.path.join(output_path, "output.md")
        markdown_content = ""
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

        # Collect extracted images
        images = []
        images_dir = os.path.join(output_path, "images")
        if os.path.exists(images_dir):
            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                images.append({
                    "path": img_path,
                    "page_num": self._extract_page_num(img_file),
                    "description": ""
                })

        return ParsedDocument(
            document_id=doc_id,
            markdown_content=markdown_content,
            images=images,
            metadata={"parser": "mineru", "source_file": pdf_path}
        )

    def _parse_with_deepseek(self, pdf_path: str, output_dir: str, doc_id: str) -> ParsedDocument:
        """Parse PDF using DeepSeek-OCR."""
        # DeepSeek-OCR uses visual tokens, similar interface
        output_path = os.path.join(output_dir, doc_id)
        os.makedirs(output_path, exist_ok=True)

        # Placeholder for DeepSeek-OCR integration
        # In practice, would convert PDF pages to images and process with DeepSeek-OCR
        return ParsedDocument(
            document_id=doc_id,
            markdown_content="",
            images=[],
            metadata={"parser": "deepseek", "source_file": pdf_path}
        )

    def _extract_page_num(self, filename: str) -> int:
        """Extract page number from image filename."""
        # Assume format: page_001.png or similar
        try:
            parts = filename.replace(".png", "").replace(".jpg", "").split("_")
            return int(parts[-1]) if parts else 0
        except:
            return 0


class ChunkProcessor:
    """Splits markdown into chunks for embedding."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_markdown(self, markdown_content: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Split markdown into overlapping chunks.

        Args:
            markdown_content: Full markdown text
            document_id: Document identifier

        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        lines = markdown_content.split("\n")

        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)
            if current_size + line_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "document_id": document_id,
                    "chunk_index": len(chunks)
                })
                # Start new chunk with overlap
                overlap_lines = current_chunk[-self.overlap // 10:]
                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) for l in current_chunk)
            else:
                current_chunk.append(line)
                current_size += line_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "document_id": document_id,
                "chunk_index": len(chunks)
            })

        return chunks