"""
语义消息历史模块 - Semantic Message History
用于存储和检索对话历史，支持基于语义的快速检索
"""
from typing import List, Dict, Optional, Any
from redisvl.extensions.message_history import SemanticMessageHistory
from config import redis_config, cache_config, index_config


class MessageHistoryManager:
    """语义消息历史管理器"""
    
    def __init__(
        self,
        name: str = None,
        distance_threshold: float = None,
        redis_url: str = None
    ):
        """
        初始化消息历史
        
        Args:
            name: 会话名称（可用于区分不同用户的对话）
            distance_threshold: 语义距离阈值
            redis_url: Redis连接地址
        """
        self.name = name or f"{index_config.MESSAGE_HISTORY_PREFIX}_default"
        self.distance_threshold = distance_threshold or cache_config.MESSAGE_HISTORY_DISTANCE_THRESHOLD
        self.redis_url = redis_url or redis_config.REDIS_URL
        
        # 初始化语义消息历史
        self.history = SemanticMessageHistory(
            name=self.name,
            redis_url=self.redis_url,
            distance_threshold=self.distance_threshold
        )
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加单条消息
        
        Args:
            role: 角色类型 (system, user, llm, tool)
            content: 消息内容
            metadata: 额外的元数据（可选）
            
        Returns:
            bool: 是否添加成功
        """
        try:
            message = {
                "role": role,
                "content": content
            }
            if metadata:
                message["metadata"] = metadata
            
            self.history.add_messages([message])
            return True
        except Exception as e:
            print(f"添加消息失败: {e}")
            return False
    
    def add_messages(self, messages: List[Dict[str, Any]]) -> bool:
        """
        批量添加消息
        
        Args:
            messages: 消息列表，每条消息包含role、content和可选的metadata
                     例如：[
                         {"role": "user", "content": "你好"},
                         {"role": "llm", "content": "你好！有什么可以帮助你的？", "metadata": {"model": "gpt-4"}}
                     ]
            
        Returns:
            bool: 是否添加成功
        """
        try:
            self.history.add_messages(messages)
            return True
        except Exception as e:
            print(f"批量添加消息失败: {e}")
            return False
    
    def get_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的消息历史
        
        Args:
            limit: 返回消息数量限制
            
        Returns:
            List[Dict]: 消息列表
        """
        try:
            messages = self.history.get_recent(limit=limit)
            return messages if messages else []
        except Exception as e:
            print(f"获取消息历史失败: {e}")
            return []
    
    def search_similar(
        self,
        query: str,
        limit: int = 5,
        distance_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        搜索语义相似的历史消息
        
        Args:
            query: 查询文本
            limit: 返回结果数量限制
            distance_threshold: 语义距离阈值（可选，覆盖默认值）
            
        Returns:
            List[Dict]: 相似的消息列表
        """
        try:
            results = self.history.search(
                query=query,
                num_results=limit,
                distance_threshold=distance_threshold or self.distance_threshold
            )
            return results if results else []
        except Exception as e:
            print(f"搜索相似消息失败: {e}")
            return []
    
    def clear(self) -> bool:
        """
        清空当前会话的消息历史
        
        Returns:
            bool: 是否清空成功
        """
        try:
            self.history.clear()
            return True
        except Exception as e:
            print(f"清空消息历史失败: {e}")
            return False
    
    def delete(self) -> bool:
        """
        删除当前会话的消息历史索引
        
        Returns:
            bool: 是否删除成功
        """
        try:
            self.history.delete()
            return True
        except Exception as e:
            print(f"删除消息历史索引失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取消息历史统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            stats = self.history.info()
            return stats
        except Exception as e:
            print(f"获取消息历史统计信息失败: {e}")
            return {}
    
    def get_message_count(self) -> int:
        """
        获取消息总数
        
        Returns:
            int: 消息数量
        """
        stats = self.get_stats()
        return stats.get("num_messages", 0)


# 便捷函数
def create_message_history(
    name: str = None,
    distance_threshold: float = None,
    redis_url: str = None
) -> MessageHistoryManager:
    """
    创建消息历史实例
    
    Args:
        name: 会话名称
        distance_threshold: 语义距离阈值
        redis_url: Redis连接地址
        
    Returns:
        MessageHistoryManager: 消息历史管理器实例
    """
    return MessageHistoryManager(
        name=name,
        distance_threshold=distance_threshold,
        redis_url=redis_url
    )
