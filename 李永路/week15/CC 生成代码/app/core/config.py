from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Storage paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOCUMENT_PATH: str = os.path.join(BASE_DIR, "storage", "documents")
    IMAGE_PATH: str = os.path.join(BASE_DIR, "storage", "images")
    MARKDOWN_PATH: str = os.path.join(BASE_DIR, "storage", "markdown")

    # Milvus settings
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "pdf_knowledge_base"

    # Model settings
    QWEN_API_KEY: Optional[str] = None
    QWEN_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MINERU_API_URL: str = "http://localhost:8000"

    # CLIP settings
    CLIP_MODEL: str = "openai/clip-vit-base-patch32"

    # BGE settings
    BGE_MODEL: str = "BAAI/bge-large-en-v1.5"

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()