"""工厂模式核心 - 服务实例化管理"""
from typing import Optional, Dict, Type, Any, Callable
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    服务工厂 - 管理所有服务的单例生命周期
    使用工厂模式统一管理：PDF解析器、Embedding模型、向量数据库、LLM等
    """
    _services: Dict[str, Any] = {}
    _factories: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, factory: Callable):
        """注册服务工厂"""
        cls._factories[name] = factory
        logger.info(f"Registered factory: {name}")

    @classmethod
    def get(cls, name: str, **kwargs) -> Any:
        """获取服务实例（单例）"""
        if name not in cls._services:
            if name not in cls._factories:
                raise ValueError(f"Factory '{name}' not registered")
            cls._services[name] = cls._factories[name](**kwargs)
            logger.info(f"Created service instance: {name}")
        return cls._services[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """强制创建新实例（绕过单例）"""
        if name not in cls._factories:
            raise ValueError(f"Factory '{name}' not registered")
        return cls._factories[name](**kwargs)

    @classmethod
    def reset(cls, name: Optional[str] = None):
        """重置服务实例"""
        if name:
            cls._services.pop(name, None)
        else:
            cls._services.clear()
        logger.info(f"Reset service(s): {name or 'all'}")

    @classmethod
    def list_services(cls) -> list:
        """列出已注册的服务"""
        return list(cls._factories.keys())


class PDFParserFactory:
    """PDF 解析器工厂"""

    @staticmethod
    def create_mineru(**kwargs):
        from app.services.pdf_parser import MinerUParser
        return MinerUParser(**kwargs)

    @staticmethod
    def create_deepseek_ocr(**kwargs):
        from app.services.pdf_parser import DeepSeekOCRParser
        return DeepSeekOCRParser(**kwargs)

    @classmethod
    def register_all(cls):
        ServiceFactory.register("pdf_parser_mineru", cls.create_mineru)
        ServiceFactory.register("pdf_parser_deepseek", cls.create_deepseek_ocr)


class EmbeddingFactory:
    """Embedding 模型工厂"""

    @staticmethod
    def create_clip(**kwargs):
        from app.services.embedding.clip_service import CLIPEmbedding
        return CLIPEmbedding(**kwargs)

    @staticmethod
    def create_bge(**kwargs):
        from app.services.embedding.bge_service import BGEEmbedding
        return BGEEmbedding(**kwargs)

    @classmethod
    def register_all(cls):
        ServiceFactory.register("embedding_clip", cls.create_clip)
        ServiceFactory.register("embedding_bge", cls.create_bge)


class VectorDBFactory:
    """向量数据库工厂"""

    @staticmethod
    def create_milvus(**kwargs):
        from app.services.vector_db.milvus_service import MilvusService
        return MilvusService(**kwargs)

    @classmethod
    def register_all(cls):
        ServiceFactory.register("vector_db_milvus", cls.create_milvus)


class LLMFactory:
    """LLM 工厂"""

    @staticmethod
    def create_qwen_vl(**kwargs):
        from app.services.llm.qwen_vl_service import QwenVLService
        return QwenVLService(**kwargs)

    @classmethod
    def register_all(cls):
        ServiceFactory.register("llm_qwen_vl", cls.create_qwen_vl)


class KafkaFactory:
    """Kafka 工厂"""

    @staticmethod
    def create_producer(**kwargs):
        from app.services.mq.kafka_producer import KafkaProducerService
        return KafkaProducerService(**kwargs)

    @staticmethod
    def create_consumer(**kwargs):
        from app.services.mq.kafka_consumer import KafkaConsumerService
        return KafkaConsumerService(**kwargs)

    @classmethod
    def register_all(cls):
        ServiceFactory.register("kafka_producer", cls.create_producer)
        ServiceFactory.register("kafka_consumer", cls.create_consumer)


class RetrievalFactory:
    """检索服务工厂"""

    @staticmethod
    def create_hybrid_retrieval(**kwargs):
        from app.services.retrieval import HybridRetrievalService
        return HybridRetrievalService(**kwargs)

    @classmethod
    def register_all(cls):
        ServiceFactory.register("retrieval_hybrid", cls.create_hybrid_retrieval)


def initialize_factories():
    """初始化所有工厂注册"""
    PDFParserFactory.register_all()
    EmbeddingFactory.register_all()
    VectorDBFactory.register_all()
    LLMFactory.register_all()
    KafkaFactory.register_all()
    RetrievalFactory.register_all()
    logger.info("All factories initialized")


def get_service(name: str, **kwargs):
    """便捷方法：获取服务实例"""
    return ServiceFactory.get(name, **kwargs)