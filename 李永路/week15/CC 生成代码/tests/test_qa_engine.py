"""
Tests for QA engine service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from app.services.qa_engine import MultimodalQA, QAResponse
from app.services.vector_store import SearchResult


class TestQAResponse:
    """Test cases for QAResponse dataclass."""

    def test_qa_response_creation(self):
        """Test creating a QAResponse."""
        response = QAResponse(
            answer="This is the answer",
            sources=[
                {"content": "source 1", "type": "text", "page_num": 1, "document_id": "doc1"}
            ],
            score=0.95
        )

        assert response.answer == "This is the answer"
        assert len(response.sources) == 1
        assert response.score == 0.95


class TestMultimodalQA:
    """Test cases for MultimodalQA."""

    def setup_method(self):
        """Setup test fixtures."""
        self.qa = MultimodalQA()

    def test_qa_initialization(self):
        """Test QA engine initializes correctly."""
        assert self.qa.api_key is None or isinstance(self.qa.api_key, str)
        assert self.qa.base_url is not None

    def test_answer_empty_retrieved_content(self):
        """Test answer with no retrieved content."""
        response = self.qa.answer(
            question="What is AI?",
            retrieved_content=[],
            knowledge_base_id="kb123"
        )

        assert response.answer is not None
        assert response.sources == []
        assert isinstance(response.score, float)

    def test_answer_with_text_content(self):
        """Test answer with text content only."""
        search_result = SearchResult(
            content="AI stands for Artificial Intelligence",
            content_type="text",
            score=0.9,
            document_id="doc123",
            metadata={"page_num": 1, "chunk_index": 0}
        )

        response = self.qa.answer(
            question="What does AI stand for?",
            retrieved_content=[search_result],
            knowledge_base_id="kb123"
        )

        assert response.answer is not None
        assert len(response.sources) == 1
        assert response.sources[0]["type"] == "text"

    def test_answer_with_image_content(self):
        """Test answer with image content."""
        search_result = SearchResult(
            content="/path/to/diagram.png",
            content_type="image",
            score=0.85,
            document_id="doc456",
            metadata={"page_num": 2}
        )

        response = self.qa.answer(
            question="What does this diagram show?",
            retrieved_content=[search_result],
            knowledge_base_id="kb123"
        )

        assert response.answer is not None
        assert len(response.sources) == 1
        assert response.sources[0]["type"] == "image"

    def test_answer_with_mixed_content(self):
        """Test answer with both text and image content."""
        text_result = SearchResult(
            content="The chart shows revenue growth",
            content_type="text",
            score=0.9,
            document_id="doc123",
            metadata={"page_num": 1, "chunk_index": 0}
        )

        image_result = SearchResult(
            content="/path/to/chart.png",
            content_type="image",
            score=0.85,
            document_id="doc123",
            metadata={"page_num": 1}
        )

        response = self.qa.answer(
            question="What is the revenue trend?",
            retrieved_content=[text_result, image_result],
            knowledge_base_id="kb123"
        )

        assert response.answer is not None
        assert len(response.sources) == 2
        source_types = [s["type"] for s in response.sources]
        assert "text" in source_types
        assert "image" in source_types

    def test_build_prompt(self):
        """Test prompt building."""
        prompt = self.qa._build_prompt(
            question="What is machine learning?",
            text_contents=["ML is a subset of AI", "It enables computers to learn"],
            image_contents=["/path/to/ml.png"]
        )

        assert "What is machine learning?" in prompt
        assert "ML is a subset of AI" in prompt
        assert "检索到的图片数量: 1" in prompt

    def test_build_prompt_empty_images(self):
        """Test prompt building without images."""
        prompt = self.qa._build_prompt(
            question="What is Python?",
            text_contents=["Python is a programming language"],
            image_contents=[]
        )

        assert "Python is a programming language" in prompt
        assert "检索到的图片数量" not in prompt


class TestMultimodalQAApiCall:
    """Test cases for QA API calls with mocking."""

    @patch('app.services.qa_engine.httpx.post')
    def test_call_qwen_vl_success(self, mock_post):
        """Test successful Qwen-VL API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test answer"}}]
        }
        mock_post.return_value = mock_response

        qa = MultimodalQA()
        qa.api_key = "test_key"

        result = qa._call_qwen_vl("Test prompt", [])

        assert result == "Test answer"

    @patch('app.services.qa_engine.httpx.post')
    def test_call_qwen_vl_no_api_key(self, mock_post):
        """Test API call without API key."""
        qa = MultimodalQA()
        qa.api_key = None

        result = qa._call_qwen_vl("Test prompt", [])

        assert "API key not configured" in result
        mock_post.assert_not_called()

    @patch('app.services.qa_engine.httpx.post')
    def test_call_qwen_vl_api_error(self, mock_post):
        """Test API call with error response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        qa = MultimodalQA()
        qa.api_key = "test_key"

        result = qa._call_qwen_vl("Test prompt", [])

        assert "API error" in result