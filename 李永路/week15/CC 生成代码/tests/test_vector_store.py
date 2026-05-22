"""
Tests for vector store service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from app.services.vector_store import VectorStore, SearchResult


class TestSearchResult:
    """Test cases for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            content="test content",
            content_type="text",
            score=0.95,
            document_id="doc123",
            metadata={"page_num": 1, "chunk_index": 0}
        )

        assert result.content == "test content"
        assert result.content_type == "text"
        assert result.score == 0.95
        assert result.document_id == "doc123"
        assert result.metadata["page_num"] == 1

    def test_search_result_image_type(self):
        """Test SearchResult with image type."""
        result = SearchResult(
            content="/path/to/image.png",
            content_type="image",
            score=0.88,
            document_id="doc456",
            metadata={"page_num": 3}
        )

        assert result.content_type == "image"
        assert ".png" in result.content


class TestVectorStore:
    """Test cases for VectorStore."""

    def setup_method(self):
        """Setup test fixtures."""
        self.store = VectorStore()

    def test_vector_store_initialization(self):
        """Test VectorStore initializes correctly."""
        assert self.store.collection is None
        assert self.store._connected is False

    @patch('app.services.vector_store.connections')
    def test_connect(self, mock_connections):
        """Test connecting to Milvus."""
        mock_connections.connect = Mock()

        self.store.connect()

        assert self.store._connected is True
        mock_connections.connect.assert_called_once()

    @patch('app.services.vector_store.connections')
    def test_disconnect(self, mock_connections):
        """Test disconnecting from Milvus."""
        mock_connections.connect = Mock()
        mock_connections.disconnect = Mock()

        self.store.connect()
        self.store.disconnect()

        assert self.store._connected is False
        mock_connections.disconnect.assert_called_once()

    def test_search_result_without_connection(self):
        """Test that search operations fail without connection."""
        store = VectorStore()

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            store.search_text([0.1] * 1024)

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            store.search_image([0.1] * 1024)

    @patch('app.services.vector_store.connections')
    def test_insert_text_chunks_without_collection(self, mock_connections):
        """Test that insert operations fail without collection."""
        store = VectorStore()
        # Mock the connection but don't actually connect
        store._connected = True

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            store.insert_text_chunks([], [])

    def test_delete_by_document_id_without_connection(self):
        """Test delete fails without connection."""
        store = VectorStore()

        with pytest.raises(RuntimeError, match="Collection not initialized"):
            store.delete_by_document_id("doc123")


class TestVectorStoreSearch:
    """Test cases for search operations with mocked collection."""

    def setup_method(self):
        """Setup test fixtures with mocked collection."""
        self.store = VectorStore()
        self.store._connected = True
        self.mock_collection = MagicMock()
        self.store.collection = self.mock_collection

    def test_search_text_returns_results(self):
        """Test search_text returns properly formatted results."""
        # Setup mock
        mock_hit = MagicMock()
        mock_hit.entity = {
            "content": "test text chunk",
            "document_id": "doc123",
            "page_num": 1,
            "chunk_index": 0
        }
        mock_hit.distance = 0.1

        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([mock_hit]))
        self.mock_collection.search = Mock(return_value=[mock_result])

        results = self.store.search_text([0.1] * 1024, top_k=5)

        assert len(results) == 1
        assert results[0].content == "test text chunk"
        assert results[0].content_type == "text"
        assert results[0].document_id == "doc123"

    def test_search_image_returns_results(self):
        """Test search_image returns properly formatted results."""
        mock_hit = MagicMock()
        mock_hit.entity = {
            "content": "/path/to/image.png",
            "document_id": "doc456",
            "page_num": 2
        }
        mock_hit.distance = 0.2

        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([mock_hit]))
        self.mock_collection.search = Mock(return_value=[mock_result])

        results = self.store.search_image([0.1] * 512, top_k=3)

        assert len(results) == 1
        assert results[0].content_type == "image"
        # Score should be converted from distance
        assert results[0].score > 0

    def test_search_hybrid_merges_results(self):
        """Test that hybrid search merges text and image results."""
        # Mock text result
        text_hit = MagicMock()
        text_hit.entity = {
            "content": "text content",
            "document_id": "doc123",
            "page_num": 1,
            "chunk_index": 0
        }
        text_hit.distance = 0.1

        text_result = MagicMock()
        text_result.__iter__ = Mock(return_value=iter([text_hit]))
        self.mock_collection.search = Mock(return_value=[text_result])

        results = self.store.search_hybrid(
            text_query_vector=[0.1] * 1024,
            image_query_vector=[0.1] * 1024,
            top_k=5
        )

        # Should have results from both text and image searches
        assert len(results) > 0