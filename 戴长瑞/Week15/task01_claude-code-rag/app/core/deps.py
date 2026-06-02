"""核心依赖注入 - FastAPI 依赖"""
from typing import Generator, Optional
from app.core.factory import get_service, initialize_factories, ServiceFactory
from app.core.config import get_config


# 延迟初始化
_initialized = False


def init_services():
    """初始化所有服务"""
    global _initialized
    if _initialized:
        return

    initialize_factories()
    config = get_config()

    # 初始化向量数据库
    milvus = get_service("vector_db_milvus",
                          host=config.milvus.host,
                          port=config.milvus.port)
    milvus.initialize_collections()

    _initialized = True


def get_milvus() -> "MilvusService":
    """获取 Milvus 服务"""
    init_services()
    from app.services.vector_db.milvus_service import MilvusService
    return ServiceFactory.get("vector_db_milvus")


def get_kafka_producer() -> "KafkaProducerService":
    """获取 Kafka 生产者"""
    init_services()
    from app.services.mq.kafka_producer import KafkaProducerService
    config = get_config()
    return ServiceFactory.get("kafka_producer",
                              bootstrap_servers=config.kafka.bootstrap_servers)


def get_clip_embedding() -> "CLIPEmbedding":
    """获取 CLIP Embedding 服务"""
    init_services()
    from app.services.embedding.clip_service import CLIPEmbedding
    config = get_config()
    return ServiceFactory.get("embedding_clip",
                              model_name=config.model.clip_model)


def get_bge_embedding() -> "BGEEmbedding":
    """获取 BGE Embedding 服务"""
    init_services()
    from app.services.embedding.bge_service import BGEEmbedding
    config = get_config()
    return ServiceFactory.get("embedding_bge",
                              model_name=config.model.bge_model)


def get_pdf_parser() -> "BasePDFParser":
    """获取 PDF 解析器"""
    init_services()
    from app.services.pdf_parser import MinerUParser
    return ServiceFactory.get("pdf_parser_mineru")


def get_qwen_vl() -> "QwenVLService":
    """获取 Qwen-VL 服务"""
    init_services()
    from app.services.llm.qwen_vl_service import QwenVLService
    config = get_config()
    return ServiceFactory.get("llm_qwen_vl",
                              model_name=config.model.qwen_model,
                              device=config.model.qwen_device)


def get_hybrid_retrieval() -> "HybridRetrievalService":
    """获取混合检索服务"""
    init_services()
    return ServiceFactory.get("retrieval_hybrid")


# 导出类型别名供类型注解使用
from app.services.vector_db.milvus_service import MilvusService
from app.services.mq.kafka_producer import KafkaProducerService
from app.services.mq.kafka_consumer import KafkaConsumerService
from app.services.embedding.clip_service import CLIPEmbedding
from app.services.embedding.bge_service import BGEEmbedding
from app.services.pdf_parser import BasePDFParser
from app.services.llm.qwen_vl_service import QwenVLService
from app.services.retrieval import HybridRetrievalService