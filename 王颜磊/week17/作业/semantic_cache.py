"""
SemanticCache — 基于 RedisVL extensions 的 LLM 语义缓存

对照课程原始实现：
- 原始用 FAISS 本地索引 + Redis String 存储 QA 对
- 升级为 RedisVL 内置 SemanticCache，向量索引在 Redis 服务端

功能：
- store(prompt, response)   — 缓存提问-回答对
- check(prompt)             — 语义检索缓存，命中返回历史回答
- clear()                   — 清空缓存

核心差异：
- 原始需要手动管理 FAISS index 文件、距离计算、Redis list
- RedisVL 一行 check() 完成向量检索 + 距离过滤 + 回答提取
"""

from typing import Optional, List, Union, Dict, Any
from redisvl.extensions.llmcache import SemanticCache as RedisVLSemanticCache
from redisvl.utils.vectorize import HFTextVectorizer


class SemanticCache:
    """基于 RedisVL SemanticCache 的 LLM 语义缓存。

    核心流程：
    1. 用户提问 → embedding → Redis 向量检索
    2. 距离 < threshold → 返回缓存回答（跳过 LLM）
    3. 距离 > threshold → 调用 LLM → 存入缓存
    """

    def __init__(
        self,
        name: str = "llm_semantic_cache",
        ttl: int = 86400,
        redis_url: str = "redis://localhost:6379",
        redis_password: str = None,
        distance_threshold: float = 0.2,
        vectorizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.name = name
        self.ttl = ttl
        self.distance_threshold = distance_threshold

        conn_kwargs = {"redis_url": redis_url}
        if redis_password:
            conn_kwargs["connection_args"] = {"password": redis_password}

        self.vectorizer = HFTextVectorizer(model=vectorizer_model)

        self.cache = RedisVLSemanticCache(
            name=name,
            ttl=ttl,
            distance_threshold=distance_threshold,
            vectorizer=self.vectorizer,
            **conn_kwargs,
        )

    def store(
        self,
        prompt: Union[str, List[str]],
        response: Union[str, List[str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """缓存提问-回答对。

        Args:
            prompt:   用户提问
            response: LLM 回答
            metadata: 附加元数据（如 model、timestamp 等）
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        for p, r in zip(prompt, response):
            self.cache.store(
                prompt=p,
                response=r,
                metadata=metadata or {},
            )

    def check(self, prompt: str, num_results: int = 1) -> List[Dict[str, Any]]:
        """语义检索缓存。

        Args:
            prompt:       新的用户提问
            num_results:  期望返回的最大结果数
        Returns:
            命中结果列表，每项含 prompt / response / distance 等字段
            未命中返回空列表
        """
        return self.cache.check(prompt=prompt, num_results=num_results)

    def clear(self) -> None:
        """清空全部缓存索引。"""
        self.cache.clear()

    def delete_entry(self, prompt: str) -> None:
        """删除指定 prompt 对应的缓存条目。"""
        self.cache.delete(prompt=prompt)


if __name__ == "__main__":
    cache = SemanticCache(
        name="test_semantic_cache",
        redis_url="redis://localhost:6379",
        distance_threshold=0.2,
    )

    print("=== SemanticCache 测试 ===")

    # 存入
    cache.store(
        prompt="What is the capital city of France?",
        response="Paris",
        metadata={"source": "test"},
    )

    # 精确命中
    result = cache.check(prompt="What is the capital city of France?")
    if result:
        print(f"精确命中: {result[0]['response']} (distance={result[0].get('distance', 'N/A')})")
    else:
        print("精确命中: 未命中")

    # 语义命中 — 不同措辞
    result = cache.check(prompt="What is France's capital?")
    if result:
        print(f"语义命中: {result[0]['response']} (distance={result[0].get('distance', 'N/A')})")
    else:
        print("语义命中: 未命中")

    # 不相关查询
    result = cache.check(prompt="How to cook pasta?")
    if result:
        print(f"不相关查询: {result[0]['response']}")
    else:
        print("不相关查询: 未命中（正确）")

    cache.clear()
    print("测试完成")
