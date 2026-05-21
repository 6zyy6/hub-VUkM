from pydantic import BaseModel
from typing import Optional, List

# API Requests
class ChatRequest(BaseModel):
    query: str
    kb_id: str

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

# Message formats for Kafka
class ParseMessage(BaseModel):
    doc_id: str
    file_path: str
    kb_id: str

# MinerU/DeepSeek OCR mock response
class ParseResult(BaseModel):
    markdown: str
    images: List[str] # List of image paths saved in OSS
