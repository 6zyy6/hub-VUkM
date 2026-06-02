"""配置管理 - 工厂模式配置"""
import os
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MilvusConfig:
    host: str = field(default_factory=lambda: os.getenv("MILVUS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("MILVUS_PORT", "19530")))
    collection_text: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION_TEXT", "mmrag_text"))
    collection_image: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION_IMAGE", "mmrag_image"))
    dimension_text: int = 1024  # BGE-M3
    dimension_image: int = 512   # CLIP


@dataclass
class KafkaConfig:
    bootstrap_servers: str = field(default_factory=lambda: os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    topic_pdf: str = field(default_factory=lambda: os.getenv("KAFKA_TOPIC_PDF", "pdf-processing"))
    topic_image: str = field(default_factory=lambda: os.getenv("KAFKA_TOPIC_IMAGE", "image-processing"))
    consumer_group: str = field(default_factory=lambda: os.getenv("KAFKA_CONSUMER_GROUP", "mmrag-workers"))


@dataclass
class ModelConfig:
    # MinerU / DeepSeek-OCR
    pdf_parser: str = field(default_factory=lambda: os.getenv("PDF_PARSER", "mineru"))
    mineru_model_path: Optional[str] = field(default_factory=lambda: os.getenv("MINERU_MODEL_PATH", None))

    # CLIP (图搜图 / 图文检索)
    clip_model: str = field(default_factory=lambda: os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32"))

    # BGE (文本检索)
    bge_model: str = field(default_factory=lambda: os.getenv("BGE_MODEL", "BAAI/bge-m3"))

    # Qwen-VL (看图理解)
    qwen_model: str = field(default_factory=lambda: os.getenv("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct"))
    qwen_device: str = field(default_factory=lambda: os.getenv("QWEN_DEVICE", "cuda"))


@dataclass
class StorageConfig:
    upload_dir: str = field(default_factory=lambda: os.getenv("UPLOAD_DIR", "./data/uploads"))
    parsed_dir: str = field(default_factory=lambda: os.getenv("PARSED_DIR", "./data/parsed"))
    image_dir: str = field(default_factory=lambda: os.getenv("IMAGE_DIR", "./data/images"))


@dataclass
class AppConfig:
    app_name: str = "Claude Code RAG"
    app_version: str = "1.0.0"
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))

    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


class ConfigFactory:
    """配置工厂"""
    _instance: Optional[AppConfig] = None

    @classmethod
    def get_config(cls) -> AppConfig:
        if cls._instance is None:
            cls._instance = AppConfig()
        return cls._instance

    @classmethod
    def reload(cls) -> AppConfig:
        cls._instance = AppConfig()
        return cls._instance


def get_config() -> AppConfig:
    return ConfigFactory.get_config()