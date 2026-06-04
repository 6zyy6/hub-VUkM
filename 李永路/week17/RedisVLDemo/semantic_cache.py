"""
语义缓存模块 - Semantic Cache
用于缓存LLM的提问和回答，基于语义相似度快速检索
"""
from typing import List, Dict, Optional, Any
from redisvl.extensions.cache.llm import SemanticCache
from config import redis_config, cache_config, index_config


class LLMCacheManager:
    """LLM语义缓存管理器"""
    
    def __init__(
        self,
        name: str = None,
        ttl: int = None,
        distance_threshold: float = None,
        redis_url: str = None
    ):
        """
        初始化语义缓存
        
        Args:
            name: 缓存名称
            ttl: 缓存过期时间（秒）
            distance_threshold: 语义距离阈值，小于此值认为相似
            redis_url: Redis连接地址
        """
        self.name = name or index_config.LLM_CACHE_INDEX
        self.ttl = ttl if ttl is not None else cache_config.SEMANTIC_CACHE_TTL
        self.distance_threshold = distance_threshold or cache_config.SEMANTIC_CACHE_DISTANCE_THRESHOLD
        self.redis_url = redis_url or redis_config.REDIS_URL
        
        # 初始化语义缓存
        self.cache = SemanticCache(
            name=self.name,
            ttl=self.ttl,
            redis_url=self.redis_url,
            distance_threshold=self.distance_threshold
        )
    
    def store(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> bool:
        """
        存储LLM的提问和回答到缓存
        
        Args:
            prompt: 用户提问
            response: LLM回答
            metadata: 额外的元数据（可选）
            
        Returns:
            bool: 是否存储成功
        """
        try:
            self.cache.store(
                prompt=prompt,
                response=response,
                metadata=metadata
            )
            return True
        except Exception as e:
            print(f"存储缓存失败: {e}")
            return False
    
    def check(self, prompt: str, num_results: int = 1) -> List[Dict[str, Any]]:
        """
        检查缓存中是否有相似的提问
        
        Args:
            prompt: 用户提问
            num_results: 返回结果数量
            
        Returns:
            List[Dict]: 匹配的缓存结果列表
        """
        try:
            results = self.cache.check(prompt=prompt, num_results=num_results)
            return results if results else []
        except Exception as e:
            print(f"查询缓存失败: {e}")
            return []
    
    def clear(self) -> bool:
        """
        清空缓存
        
        Returns:
            bool: 是否清空成功
        """
        try:
            self.cache.clear()
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
            self.cache.delete()
            return True
        except Exception as e:
            print(f"删除缓存索引失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 缓存统计信息
        """
        try:
            stats = self.cache.info()
            return stats
        except Exception as e:
            print(f"获取缓存统计信息失败: {e}")
            return {}


# 便捷函数
def create_llm_cache(
    name: str = None,
    ttl: int = None,
    distance_threshold: float = None,
    redis_url: str = None
) -> LLMCacheManager:
    """
    创建LLM语义缓存实例
    
    Args:
        name: 缓存名称
        ttl: 缓存过期时间（秒）
        distance_threshold: 语义距离阈值
        redis_url: Redis连接地址
        
    Returns:
        LLMCacheManager: 缓存管理器实例
    """
    return LLMCacheManager(
        name=name,
        ttl=ttl,
        distance_threshold=distance_threshold,
        redis_url=redis_url
    )
