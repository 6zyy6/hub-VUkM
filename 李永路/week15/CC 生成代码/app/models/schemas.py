"""
Pydantic models for API request/response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadDocumentRequest(BaseModel):
    """Request to upload a document."""
    knowledge_base_id: str = Field(..., description="Knowledge base identifier")
    filename: Optional[str] = None


class UploadDocumentResponse(BaseModel):
    """Response after document upload."""
    document_id: str
    status: str
    message: str


class ChatRequest(BaseModel):
    """Request for multimodal QA chat."""
    knowledge_base_id: str = Field(..., description="Knowledge base identifier")
    question: str = Field(..., description="User's question")
    top_k: Optional[int] = Field(default=5, description="Number of results to retrieve")


class SourceInfo(BaseModel):
    """Information about a source for the answer."""
    content: str
    type: str  # "text" or "image"
    page_num: int
    document_id: str


class ChatResponse(BaseModel):
    """Response from multimodal QA."""
    answer: str
    sources: List[SourceInfo]
    score: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "1.0.0"