"""
SemanticRouter — 基于 RedisVL 的语义路由器

对照课程原始实现：
- 原始仅定义了空接口（add_route / route 方法体均为 pass）
- 这里用 RedisVL extensions.router.SemanticRouter 完整实现

功能：
- add_route(name, references, metadata)  — 注册路由规则
- route(question)                       — 语义匹配，返回最佳路由

原理：
- 每个路由的 references 被向量化存入 Redis 索引
- 新问题向量化后在索引中检索最近邻
- 距离 < threshold → 命中该路由
"""

from typing import Optional, List, Dict, Any
from redisvl.extensions.router import SemanticRouter as RedisVLSemanticRouter, Route
from redisvl.utils.vectorize import HFTextVectorizer


class SemanticRouter:
    """基于 RedisVL 的语义路由器。

    使用场景：
    - 意图识别：将用户问题路由到不同 Agent / Prompt / 处理分支
    - 多模型调度：不同类别问题路由到不同的 LLM
    - FAQ 分发：匹配到最合适的知识库条目
    """

    def __init__(
        self,
        name: str = "semantic_router",
        redis_url: str = "redis://localhost:6379",
        redis_password: str = None,
        distance_threshold: float = 0.3,
        vectorizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.name = name
        self.routes: List[Route] = []
        self._router: Optional[RedisVLSemanticRouter] = None
        self.redis_url = redis_url
        self.redis_password = redis_password
        self.distance_threshold = distance_threshold
        self.vectorizer = HFTextVectorizer(model=vectorizer_model)

    def add_route(
        self,
        name: str,
        references: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        distance_threshold: Optional[float] = None,
    ) -> None:
        """注册一条路由规则。

        Args:
            name:               路由名称（如 "greeting", "refund", "tech_support"）
            references:         示例问句列表，用于语义匹配
            metadata:           附加元数据
            distance_threshold: 该路由的距离阈值，默认使用实例级别阈值
        """
        route = Route(
            name=name,
            references=references,
            metadata=metadata or {},
            distance_threshold=distance_threshold or self.distance_threshold,
        )
        self.routes.append(route)

    def _build(self) -> RedisVLSemanticRouter:
        """构建底层 RedisVL 路由器。"""
        if self._router is None:
            conn_kwargs = {"redis_url": self.redis_url}
            if self.redis_password:
                conn_kwargs["connection_args"] = {"password": self.redis_password}

            self._router = RedisVLSemanticRouter(
                name=self.name,
                routes=self.routes,
                vectorizer=self.vectorizer,
                **conn_kwargs,
            )
        return self._router

    def route(self, question: str) -> Optional[Dict[str, Any]]:
        """对问题执行语义路由。

        Args:
            question: 用户问题
        Returns:
            命中结果，含 name / distance / metadata 字段；未命中返回 None
        """
        router = self._build()
        result = router(question)
        if result:
            return {
                "name": result.name,
                "distance": result.distance,
                "metadata": getattr(result, "metadata", {}),
            }
        return None

    def set_threshold(self, threshold: float) -> None:
        """动态调整全局距离阈值。"""
        self.distance_threshold = threshold
        self._router = None  # 需要重建


if __name__ == "__main__":
    router = SemanticRouter(
        name="test_router",
        redis_url="redis://localhost:6379",
        distance_threshold=0.3,
    )

    router.add_route(
        name="greeting",
        references=["Hi, good morning", "Hello there", "Good afternoon", "Hey!"],
        metadata={"handler": "greeting_bot"},
    )
    router.add_route(
        name="refund",
        references=["如何退货", "我要退款", "订单不满意怎么退", "refund process"],
        metadata={"handler": "refund_bot"},
    )
    router.add_route(
        name="tech_support",
        references=["电脑蓝屏怎么办", "网络连不上", "软件崩溃了", "system error"],
        metadata={"handler": "tech_bot"},
    )

    print("=== SemanticRouter 测试 ===")

    tests = [
        "Hi, good morning",
        "你好，早上好",  # 语义相近但不同语言
        "我要退款，这个商品质量太差了",
        "我的电脑蓝屏了，怎么修复？",
        "今天天气真好",  # 无匹配路由
    ]

    for q in tests:
        result = router.route(q)
        if result:
            print(f"问题: {q[:30]:<30} → 路由: {result['name']:<15} (distance={result['distance']:.4f})")
        else:
            print(f"问题: {q[:30]:<30} → 未命中任何路由")

    print("测试完成")
