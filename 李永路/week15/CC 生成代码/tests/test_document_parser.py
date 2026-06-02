"""
Tests for document parser service.
"""

import pytest
import os
import tempfile
from pathlib import Path

from app.services.document_parser import DocumentParser, ChunkProcessor, ParsedDocument


class TestDocumentParser:
    """Test cases for DocumentParser."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = DocumentParser(parser_type="mineru")

    def test_generate_doc_id(self):
        """Test document ID generation."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            doc_id = self.parser._generate_doc_id(temp_path)
            assert doc_id is not None
            assert len(doc_id) == 12
        finally:
            os.unlink(temp_path)

    def test_extract_page_num(self):
        """Test page number extraction from filename."""
        assert self.parser._extract_page_num("page_001.png") == 1
        assert self.parser._extract_page_num("image_042.jpg") == 42
        assert self.parser._extract_page_num("unknown") == 0


class TestChunkProcessor:
    """Test cases for ChunkProcessor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.processor = ChunkProcessor(chunk_size=100, overlap=20)

    def test_chunk_markdown_basic(self):
        """Test basic markdown chunking."""
        markdown = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = self.processor.chunk_markdown(markdown, "doc123")

        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("document_id" in chunk for chunk in chunks)
        assert all(chunk["document_id"] == "doc123" for chunk in chunks)

    def test_chunk_markdown_small_content(self):
        """Test chunking when content is smaller than chunk size."""
        markdown = "Short text"
        chunks = self.processor.chunk_markdown(markdown, "doc123")

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text"
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_markdown_empty_content(self):
        """Test chunking empty content."""
        markdown = ""
        chunks = self.processor.chunk_markdown(markdown, "doc123")

        # Empty content still creates one chunk with empty text
        assert len(chunks) == 1
        assert chunks[0]["text"] == ""

    def test_chunk_metadata(self):
        """Test that chunks have correct metadata."""
        markdown = "\n".join([f"Line {i}" for i in range(50)])
        chunks = self.processor.chunk_markdown(markdown, "test_doc")

        for i, chunk in enumerate(chunks):
            assert "text" in chunk
            assert "document_id" in chunk
            assert "chunk_index" in chunk
            assert chunk["document_id"] == "test_doc"
            assert chunk["chunk_index"] == i


class TestParsedDocument:
    """Test cases for ParsedDocument dataclass."""

    def test_parsed_document_creation(self):
        """Test creating a ParsedDocument."""
        doc = ParsedDocument(
            document_id="test123",
            markdown_content="# Test\nSome content",
            images=[
                {"path": "/path/to/image.png", "page_num": 1, "description": "test image"}
            ],
            metadata={"parser": "mineru"}
        )

        assert doc.document_id == "test123"
        assert "Test" in doc.markdown_content
        assert len(doc.images) == 1
        assert doc.metadata["parser"] == "mineru"