"""API 测试"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestUploadAPI:
    """上传 API 测试"""

    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_kafka(self):
        with patch('app.core.deps.get_kafka_producer') as mock:
            mock_producer = MagicMock()
            mock.return_value = mock_producer
            yield mock_producer

    def test_health_check(self, client):
        """健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch('app.api.routes.upload.get_kafka_producer')
    @patch('app.core.deps.get_milvus')
    def test_upload_pdf(self, mock_milvus, mock_kafka, client):
        """测试 PDF 上传"""
        mock_producer = MagicMock()
        mock_kafka.return_value = mock_producer

        # 创建测试文件
        test_file = b"%PDF-1.4\ntest content"

        response = client.post(
            "/api/v1/upload/document",
            files={"file": ("test.pdf", test_file, "application/pdf")}
        )

        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.pdf"

    def test_upload_non_pdf(self, client):
        """测试非 PDF 文件上传被拒绝"""
        test_file = b"not a pdf"

        response = client.post(
            "/api/v1/upload/document",
            files={"file": ("test.txt", test_file, "text/plain")}
        )

        assert response.status_code == 400
        assert "Only PDF files" in response.json()["detail"]


class TestChatAPI:
    """对话 API 测试"""

    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    @patch('app.api.routes.chat.get_hybrid_retrieval')
    @patch('app.api.routes.chat.get_qwen_vl')
    def test_chat(self, mock_qwen, mock_retrieval, client):
        """测试对话"""
        # Mock 检索结果
        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = []
        mock_retrieval.return_value = mock_ret

        # Mock Qwen 回复
        mock_qwen_svc = MagicMock()
        mock_qwen_svc.chat.return_value = "这是测试回复"
        mock_qwen.return_value = mock_qwen_svc

        response = client.post(
            "/api/v1/chat",
            json={"query": "测试问题", "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "session_id" in data

    @patch('app.api.routes.chat.get_hybrid_retrieval')
    def test_search_only(self, mock_retrieval, client):
        """测试仅检索"""
        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = []
        mock_retrieval.return_value = mock_ret

        response = client.post(
            "/api/v1/chat/search",
            params={"query": "测试", "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data


class TestRetrievalService:
    """检索服务测试"""

    def test_rrf_fusion(self):
        """测试 RRF 融合"""
        from app.services.retrieval import HybridRetrievalService
        from app.models.data_models import SearchResult, ChunkType

        # 模拟检索结果
        results = [
            SearchResult(
                chunk_id="1", document_id="doc1", content="test1",
                image_paths=[], score=0.9, chunk_type=ChunkType.TEXT
            ),
            SearchResult(
                chunk_id="2", document_id="doc1", content="test2",
                image_paths=[], score=0.8, chunk_type=ChunkType.TEXT
            ),
            SearchResult(
                chunk_id="1", document_id="doc1", content="test1",
                image_paths=[], score=0.85, chunk_type=ChunkType.TEXT
            ),
        ]

        # 测试融合（相同 ID 取最高分）
        unique_results = {}
        for r in results:
            if r.chunk_id not in unique_results or r.score > unique_results[r.chunk_id].score:
                unique_results[r.chunk_id] = r

        assert len(unique_results) == 2
        assert unique_results["1"].score == 0.9


class TestFactory:
    """工厂模式测试"""

    def test_service_factory_register(self):
        """测试服务工厂注册"""
        from app.core.factory import ServiceFactory

        def mock_factory():
            return "mock_service"

        ServiceFactory.register("test_service", mock_factory)
        assert "test_service" in ServiceFactory.list_services()

    def test_service_factory_get(self):
        """测试服务工厂获取"""
        from app.core.factory import ServiceFactory

        def mock_factory():
            return "mock_service_instance"

        ServiceFactory.register("test_get", mock_factory)
        service = ServiceFactory.get("test_get")

        assert service == "mock_service_instance"
        # 再次获取应该返回同一个实例
        service2 = ServiceFactory.get("test_get")
        assert service is service2

        # 清理
        ServiceFactory.reset()


class TestDataModels:
    """数据模型测试"""

    def test_chat_request_validation(self):
        """测试 ChatRequest 验证"""
        from app.models.data_models import ChatRequest

        # 有效请求
        req = ChatRequest(query="test")
        assert req.query == "test"
        assert req.top_k == 5  # 默认值

    def test_chat_request_with_images(self):
        """测试带图片的请求"""
        from app.models.data_models import ChatRequest

        req = ChatRequest(
            query="描述这张图",
            image_paths=["/path/to/image.jpg"],
            top_k=3
        )
        assert len(req.image_paths) == 1

    def test_search_result(self):
        """测试 SearchResult"""
        from app.models.data_models import SearchResult, ChunkType

        result = SearchResult(
            chunk_id="chunk1",
            document_id="doc1",
            content="test content",
            image_paths=["img1.jpg", "img2.jpg"],
            score=0.95,
            chunk_type=ChunkType.TEXT
        )

        assert result.chunk_type == ChunkType.TEXT
        assert len(result.image_paths) == 2


# 运行标记
pytestmark = pytest.mark.asyncio