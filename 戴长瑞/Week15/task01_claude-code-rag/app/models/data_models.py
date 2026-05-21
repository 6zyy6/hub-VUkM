"""数据模型定义"""
from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    MIXED = "mixed"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============ 文档相关模型 ============

class DocumentMetadata(BaseModel):
    """文档元数据"""
    filename: str
    file_size: int
    file_type: str
    page_count: Optional[int] = None
    uploader_id: Optional[str] = None


class Document(BaseModel):
    """文档模型"""
    id: str
    metadata: DocumentMetadata
    status: DocumentStatus = DocumentStatus.UPLOADED
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None


# ============ Chunk 相关模型 ============

class Chunk(BaseModel):
    """文档块模型"""
    id: str
    document_id: str
    chunk_type: ChunkType
    content: str  # Markdown 格式
    image_paths: List[str] = []  # 关联的图片路径
    page_number: Optional[int] = None
    chunk_index: int
    metadata: dict = {}


# ============ 检索相关模型 ============

class SearchResult(BaseModel):
    """检索结果"""
    chunk_id: str
    document_id: str
    content: str
    image_paths: List[str] = []
    score: float
    chunk_type: ChunkType


class RetrievalRequest(BaseModel):
    """检索请求"""
    query: str
    top_k: int = Field(default=5, le=20)
    document_ids: Optional[List[str]] = None  # 可选：限定文档范围


class RetrievalResponse(BaseModel):
    """检索响应"""
    results: List[SearchResult]
    query: str


# ============ 对话相关模型 ============

class ChatMessage(BaseModel):
    """对话消息"""
    role: MessageRole
    content: str
    image_paths: List[str] = []  # 多模态对话中的图片


class ChatRequest(BaseModel):
    """对话请求"""
    query: str
    session_id: Optional[str] = None
    image_paths: List[str] = []  # 上传的图片
    top_k: int = Field(default=5, le=20)


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str
    sources: List[SearchResult] = []
    session_id: str
    images: List[str] = []  # 生成的回答中包含的图片


# ============ API 请求/响应 ============

class UploadResponse(BaseModel):
    """上传响应"""
    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class ProcessingStatus(BaseModel):
    """处理状态"""
    document_id: str
    status: DocumentStatus
    progress: float = 0.0  # 0.0 - 1.0
    message: Optional[str] = None


# ============ Kafka 消息模型 ============

class PDFProcessMessage(BaseModel):
    """PDF 处理消息"""
    document_id: str
    file_path: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ImageProcessMessage(BaseModel):
    """图片处理消息"""
    document_id: str
    chunk_id: str
    image_path: str
    timestamp: datetime = Field(default_factory=datetime.now)