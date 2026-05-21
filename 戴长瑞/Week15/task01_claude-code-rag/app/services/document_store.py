"""文档存储服务 - 关系型数据库（MongoDB/PostgreSQL）"""
from typing import Optional, List
from datetime import datetime
from abc import ABC, abstractmethod
import logging

from app.models.data_models import Document, DocumentStatus

logger = logging.getLogger(__name__)


class BaseDocumentStore(ABC):
    """文档存储基类"""

    @abstractmethod
    def save(self, document: Document) -> Document:
        """保存文档"""
        pass

    @abstractmethod
    def get(self, document_id: str) -> Optional[Document]:
        """获取文档"""
        pass

    @abstractmethod
    def update_status(self, document_id: str, status: DocumentStatus, **kwargs):
        """更新状态"""
        pass

    @abstractmethod
    def list(self, limit: int = 100) -> List[Document]:
        """列出文档"""
        pass


class InMemoryDocumentStore(BaseDocumentStore):
    """内存文档存储（开发/测试用）"""

    def __init__(self):
        self._documents = {}

    def save(self, document: Document) -> Document:
        self._documents[document.id] = document
        logger.info(f"Document saved: {document.id}")
        return document

    def get(self, document_id: str) -> Optional[Document]:
        return self._documents.get(document_id)

    def update_status(self, document_id: str, status: DocumentStatus, **kwargs):
        if document_id in self._documents:
            doc = self._documents[document_id]
            doc.status = status
            doc.updated_at = datetime.now()
            if "error_message" in kwargs:
                doc.error_message = kwargs["error_message"]
            if "page_count" in kwargs:
                doc.metadata.page_count = kwargs["page_count"]

    def list(self, limit: int = 100) -> List[Document]:
        return list(self._documents.values())[:limit]


# 全局实例
_document_store: Optional[BaseDocumentStore] = None


def get_document_store() -> BaseDocumentStore:
    """获取文档存储实例"""
    global _document_store
    if _document_store is None:
        _document_store = InMemoryDocumentStore()
    return _document_store


def set_document_store(store: BaseDocumentStore):
    """设置文档存储"""
    global _document_store
    _document_store = store