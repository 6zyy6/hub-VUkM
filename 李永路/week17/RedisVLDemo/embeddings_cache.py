"""
嵌入缓存模块 - Embeddings Cache
用于缓存文本到向量的转换结果，避免重复调用embedding模型
"""
from typing import List, Optional, Union
import numpy as np
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.utils.vectorize import HFTextVectorizer
from config import redis_config, cache_config, model_config, index_config


class EmbeddingsCacheManager:
    """嵌入缓存管理器"""
    
    def __init__(
        self,
        name: str = None,
        ttl: int = None,
        model_name: str = None,
        redis_url: str = None
    ):
        """
        初始化嵌入缓存
        
        Args:
            name: 缓存名称
            ttl: 缓存过期时间（秒）
            model_name: 向量模型名称
            redis_url: Redis连接地址
        """
        self.name = name or index_config.EMBEDDINGS_CACHE_INDEX
        self.ttl = ttl if ttl is not None else cache_config.EMBEDDINGS_CACHE_TTL
        self.model_name = model_name or model_config.DEFAULT_MODEL
        self.redis_url = redis_url or redis_config.REDIS_URL
        
        # 初始化嵌入缓存
        self.embed_cache = EmbeddingsCache(
            name=self.name,
            redis_url=self.redis_url,
            ttl=self.ttl
        )
        
        # 初始化向量化工具（带缓存）
        self.vectorizer = HFTextVectorizer(
            model=self.model_name,
            cache=self.embed_cache
        )
    
    def embed(self, text: str) -> List[float]:
        """
        将文本转换为向量（自动使用缓存）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        try:
            embedding = self.vectorizer.embed(text)
            return embedding
        except Exception as e:
            print(f"文本向量化失败: {e}")
            return []
    
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        try:
            embeddings = []
            for text in texts:
                embedding = self.embed(text)
                if embedding:
                    embeddings.append(embedding)
            return embeddings
        except Exception as e:
            print(f"批量文本向量化失败: {e}")
            return []
    
    def clear(self) -> bool:
        """
        清空缓存
        
        Returns:
            bool: 是否清空成功
        """
        try:
            self.embed_cache.clear()
            return True
        except Exception as e:
            print(f"清空缓存失败: {e}")
            return False
    
    def delete(self) -> bool:
        """
        删除缓存索引
        
        Returns:
            bool: 是否删除成功
        """
        try:
            self.embed_cache.delete()
            return True
        except Exception as e:
            print(f"删除缓存索引失败: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            dict: 缓存统计信息
        """
        try:
            stats = self.embed_cache.info()
            return stats
        except Exception as e:
            print(f"获取缓存统计信息失败: {e}")
            return {}
    
    def get_vectorizer(self) -> HFTextVectorizer:
        """
        获取向量化工具实例
        
        Returns:
            HFTextVectorizer: 向量化工具
        """
        return self.vectorizer


# 便捷函数
def create_embeddings_cache(
    name: str = None,
    ttl: int = None,
    model_name: str = None,
    redis_url: str = None
) -> EmbeddingsCacheManager:
    """
    创建嵌入缓存实例
    
    Args:
        name: 缓存名称
        ttl: 缓存过期时间（秒）
        model_name: 向量模型名称
        redis_url: Redis连接地址
        
    Returns:
        EmbeddingsCacheManager: 缓存管理器实例
    """
    return EmbeddingsCacheManager(
        name=name,
        ttl=ttl,
        model_name=model_name,
        redis_url=redis_url
    )
