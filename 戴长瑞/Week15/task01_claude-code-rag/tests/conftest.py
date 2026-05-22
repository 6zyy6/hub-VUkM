"""pytest 配置"""
import pytest
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """pytest 配置"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """测试数据目录"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def mock_milvus():
    """Mock Milvus 服务"""
    from unittest.mock import MagicMock

    milvus = MagicMock()
    milvus.search.return_value = [[{
        "id": "test_chunk",
        "document_id": "test_doc",
        "content": "test content",
        "page_number": 1,
        "image_paths": "[]",
        "score": 0.9
    }]]
    return milvus


@pytest.fixture
def mock_kafka():
    """Mock Kafka 服务"""
    from unittest.mock import MagicMock

    kafka = MagicMock()
    kafka.send_pdf_process_message = MagicMock()
    kafka.send_image_process_message = MagicMock()
    return kafka