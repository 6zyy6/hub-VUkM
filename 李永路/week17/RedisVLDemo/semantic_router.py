"""
语义路由模块 - Semantic Router
用于实现意图识别，基于语义相似度将用户输入分类到不同的路由
"""
from typing import List, Dict, Optional, Any
from redisvl.extensions.router import Route, SemanticRouter
from config import redis_config, cache_config, index_config


class SemanticRouterManager:
    """语义路由管理器"""
    
    def __init__(
        self,
        name: str = None,
        routes: List[Route] = None,
        distance_threshold: float = None,
        redis_url: str = None
    ):
        """
        初始化语义路由
        
        Args:
            name: 路由器名称
            routes: 路由列表
            distance_threshold: 默认语义距离阈值
            redis_url: Redis连接地址
        """
        self.name = name or f"{index_config.SEMANTIC_ROUTER_PREFIX}_default"
        self.routes = routes or []
        self.distance_threshold = distance_threshold or cache_config.SEMANTIC_ROUTER_DISTANCE_THRESHOLD
        self.redis_url = redis_url or redis_config.REDIS_URL
        
        # 初始化语义路由
        if self.routes:
            self.router = SemanticRouter(
                name=self.name,
                routes=self.routes,
                redis_url=self.redis_url
            )
        else:
            self.router = None
    
    def add_route(
        self,
        name: str,
        references: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        distance_threshold: float = None
    ) -> bool:
        """
        添加路由
        
        Args:
            name: 路由名称（类别名）
            references: 参考示例列表
            metadata: 额外的元数据（可选）
            distance_threshold: 该路由的距离阈值（可选）
            
        Returns:
            bool: 是否添加成功
        """
        try:
            route = Route(
                name=name,
                references=references,
                metadata=metadata,
                distance_threshold=distance_threshold or self.distance_threshold
            )
            self.routes.append(route)
            
            # 重新初始化路由器
            self._rebuild_router()
            return True
        except Exception as e:
            print(f"添加路由失败: {e}")
            return False
    
    def add_routes(self, routes: List[Route]) -> bool:
        """
        批量添加路由
        
        Args:
            routes: 路由列表
            
        Returns:
            bool: 是否添加成功
        """
        try:
            self.routes.extend(routes)
            self._rebuild_router()
            return True
        except Exception as e:
            print(f"批量添加路由失败: {e}")
            return False
    
    def route(self, query: str) -> Optional[Dict[str, Any]]:
        """
        对查询进行路由分类
        
        Args:
            query: 用户输入文本
            
        Returns:
            Dict: 匹配的路由信息，包含name、metadata等；如果没有匹配则返回None
        """
        if not self.router:
            print("路由器未初始化，请先添加路由")
            return None
        
        try:
            result = self.router(query)
            return result
        except Exception as e:
            print(f"路由分类失败: {e}")
            return None
    
    def route_with_fallback(
        self,
        query: str,
        default_route: str = "unknown"
    ) -> Dict[str, Any]:
        """
        对查询进行路由分类，如果未匹配则返回默认路由
        
        Args:
            query: 用户输入文本
            default_route: 默认路由名称
            
        Returns:
            Dict: 匹配的路由信息或默认路由
        """
        result = self.route(query)
        
        if result is None:
            return {
                "name": default_route,
                "metadata": {"type": default_route}
            }
        
        return result
    
    def get_all_routes(self) -> List[Dict[str, Any]]:
        """
        获取所有路由信息
        
        Returns:
            List[Dict]: 路由列表
        """
        routes_info = []
        for route in self.routes:
            routes_info.append({
                "name": route.name,
                "references": route.references,
                "metadata": route.metadata,
                "distance_threshold": route.distance_threshold
            })
        return routes_info
    
    def remove_route(self, route_name: str) -> bool:
        """
        移除指定路由
        
        Args:
            route_name: 路由名称
            
        Returns:
            bool: 是否移除成功
        """
        try:
            self.routes = [r for r in self.routes if r.name != route_name]
            self._rebuild_router()
            return True
        except Exception as e:
            print(f"移除路由失败: {e}")
            return False
    
    def clear(self) -> bool:
        """
        清空所有路由
        
        Returns:
            bool: 是否清空成功
        """
        try:
            self.routes = []
            self.router = None
            return True
        except Exception as e:
            print(f"清空路由失败: {e}")
            return False
    
    def delete(self) -> bool:
        """
        删除路由索引
        
        Returns:
            bool: 是否删除成功
        """
        try:
            if self.router:
                self.router.delete()
            return True
        except Exception as e:
            print(f"删除路由索引失败: {e}")
            return False
    
    def _rebuild_router(self):
        """重建路由器"""
        if self.routes:
            self.router = SemanticRouter(
                name=self.name,
                routes=self.routes,
                redis_url=self.redis_url
            )
        else:
            self.router = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取路由统计信息
        
        Returns:
            Dict: 统计信息
        """
        try:
            if self.router:
                stats = self.router.info()
                return stats
            return {}
        except Exception as e:
            print(f"获取路由统计信息失败: {e}")
            return {}


# 便捷函数
def create_semantic_router(
    name: str = None,
    routes: List[Route] = None,
    distance_threshold: float = None,
    redis_url: str = None
) -> SemanticRouterManager:
    """
    创建语义路由实例
    
    Args:
        name: 路由器名称
        routes: 路由列表
        distance_threshold: 默认语义距离阈值
        redis_url: Redis连接地址
        
    Returns:
        SemanticRouterManager: 语义路由管理器实例
    """
    return SemanticRouterManager(
        name=name,
        routes=routes,
        distance_threshold=distance_threshold,
        redis_url=redis_url
    )


# 预定义常用路由模板
def create_greeting_route(distance_threshold: float = None) -> Route:
    """创建问候语路由"""
    return Route(
        name="greeting",
        references=["hello", "hi", "您好", "你好", "早上好", "晚上好"],
        metadata={"type": "greeting"},
        distance_threshold=distance_threshold or cache_config.SEMANTIC_ROUTER_DISTANCE_THRESHOLD
    )


def create_farewell_route(distance_threshold: float = None) -> Route:
    """创建告别语路由"""
    return Route(
        name="farewell",
        references=["bye", "goodbye", "再见", "拜拜", "再会"],
        metadata={"type": "farewell"},
        distance_threshold=distance_threshold or cache_config.SEMANTIC_ROUTER_DISTANCE_THRESHOLD
    )


def create_question_route(distance_threshold: float = None) -> Route:
    """创建问题路由"""
    return Route(
        name="question",
        references=["what", "how", "why", "when", "where", "谁", "什么", "哪里", "怎么", "为什么"],
        metadata={"type": "question"},
        distance_threshold=distance_threshold or cache_config.SEMANTIC_ROUTER_DISTANCE_THRESHOLD
    )
