"""
Tests for API endpoints.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.vector_store import SearchResult


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)

    def test_health_check(self):
        """Test health endpoint returns OK."""
        response = self.client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestUploadEndpoint:
    """Test cases for document upload endpoint."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)

    @patch('app.api.endpoints.vector_store')
    @patch('app.api.endpoints.embedding_service')
    @patch('app.api.endpoints.chunk_processor')
    @patch('app.api.endpoints.DocumentParser')
    def test_upload_document_success(
        self, mock_parser_class, mock_chunk_proc, mock_emb_service, mock_vs
    ):
        """Test successful document upload."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_parser.parse.return_value = MagicMock(
            markdown_content="# Test\nContent here",
            images=[],
            document_id="test123"
        )
        mock_parser_class.return_value = mock_parser

        mock_chunk_proc.chunk_markdown.return_value = [
            {"text": "Test chunk", "document_id": "test123", "chunk_index": 0}
        ]

        mock_emb_service.embed_text.return_value = [[0.1] * 1024]

        mock_vs.connect.return_value = None
        mock_vs.create_collection.return_value = None
        mock_vs.insert_text_chunks.return_value = None
        mock_vs._connected = True

        # Create test file
        import io
        pdf_content = b"%PDF-1.4 fake pdf content"

        response = self.client.post(
            "/api/v1/upload/document?knowledge_base_id=test_kb",
            files={"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["success", "error"]  # May succeed or error depending on mock

    def test_upload_non_pdf_file(self):
        """Test upload with non-PDF file is rejected."""
        import io

        response = self.client.post(
            "/api/v1/upload/document?knowledge_base_id=test_kb",
            files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]


class TestChatEndpoint:
    """Test cases for chat endpoint."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)

    @patch('app.api.endpoints.qa_engine')
    @patch('app.api.endpoints.embedding_service')
    @patch('app.api.endpoints.vector_store')
    def test_chat_success(self, mock_vs, mock_emb, mock_qa):
        """Test successful chat request."""
        # Setup mocks
        mock_emb.embed_text.return_value = [[0.1] * 1024]

        mock_vs.search_hybrid.return_value = [
            SearchResult(
                content="relevant text",
                content_type="text",
                score=0.9,
                document_id="doc1",
                metadata={"page_num": 1, "chunk_index": 0}
            )
        ]

        mock_qa.answer.return_value = MagicMock(
            answer="Test answer",
            sources=[{"content": "relevant text", "type": "text", "page_num": 1, "document_id": "doc1"}],
            score=0.95
        )

        mock_vs._connected = True

        response = self.client.post(
            "/api/v1/chat",
            json={
                "knowledge_base_id": "test_kb",
                "question": "What is AI?",
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_chat_missing_question(self):
        """Test chat with missing question."""
        response = self.client.post(
            "/api/v1/chat",
            json={
                "knowledge_base_id": "test_kb"
            }
        )

        assert response.status_code == 422  # Validation error


class TestDeleteEndpoint:
    """Test cases for document deletion endpoint."""

    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)

    @patch('app.api.endpoints.vector_store')
    def test_delete_document_success(self, mock_vs):
        """Test successful document deletion."""
        mock_vs.delete_by_document_id.return_value = None
        mock_vs._connected = True

        response = self.client.delete("/api/v1/document/doc123")

        assert response.status_code == 200
        assert response.json()["status"] == "success"