"""
统一配置文件
从配置文件中读取Redis连接配置和模型配置
"""
import os
from typing import Optional


class RedisConfig:
    """Redis连接配置"""
    
    # Redis服务器地址
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Redis密码（如果需要）
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    
    # Redis数据库编号
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # 连接超时时间（秒）
    CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "5"))
    
    # 最大连接数
    MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "10"))


class ModelConfig:
    """向量化模型配置"""
    
    # 默认使用的向量模型
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 向量维度（all-MiniLM-L6-v2的维度是384）
    VECTOR_DIMENSION = 384
    
    # 批量处理大小
    BATCH_SIZE = 32


class CacheConfig:
    """缓存配置"""
    
    # 语义缓存默认TTL（秒）- 1小时
    SEMANTIC_CACHE_TTL = 3600
    
    # 语义缓存距离阈值
    SEMANTIC_CACHE_DISTANCE_THRESHOLD = 0.1
    
    # 嵌入缓存默认TTL（秒）- 1小时
    EMBEDDINGS_CACHE_TTL = 3600
    
    # 消息历史距离阈值
    MESSAGE_HISTORY_DISTANCE_THRESHOLD = 0.7
    
    # 语义路由距离阈值
    SEMANTIC_ROUTER_DISTANCE_THRESHOLD = 0.3


class IndexConfig:
    """索引配置"""
    
    # LLM缓存索引名称
    LLM_CACHE_INDEX = "llmcache"
    
    # 嵌入缓存索引名称
    EMBEDDINGS_CACHE_INDEX = "embed_cache"
    
    # 消息历史索引前缀
    MESSAGE_HISTORY_PREFIX = "msg_history"
    
    # 语义路由索引前缀
    SEMANTIC_ROUTER_PREFIX = "semantic_router"


# 导出配置实例
redis_config = RedisConfig()
model_config = ModelConfig()
cache_config = CacheConfig()
index_config = IndexConfig()
