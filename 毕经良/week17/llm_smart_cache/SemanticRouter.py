from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Union
import numpy as np
import redis
import hashlib
import json


@dataclass
class Route:
    """路由定义"""
    name: str  # 路由名称
    references: List[str]  # 参考示例
    metadata: Optional[Dict[str, Any]] = None  # 元数据
    distance_threshold: float = 0.3  # 距离阈值


class SemanticRouter:
    def __init__(
            self,
            name: str,
            routes: Optional[List[Route]] = None,
            embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            ttl: int = 3600 * 24,
    ):
        self.name = name
        self.embedding_method = embedding_method
        self.ttl = ttl

        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False
        )

        self.routes: Dict[str, Route] = {}
        if routes:
            for route in routes:
                self.add_route_by_object(route)

    def add_route_by_object(self, route: Route):
        """添加路由对象"""
        self.routes[route.name] = route

        if self.embedding_method:
            embeddings = self.embedding_method(route.references)

            pipe = self.redis.pipeline()
            for i, ref in enumerate(route.references):
                ref_hash = hashlib.md5(ref.encode()).hexdigest()
                key = f"{self.name}:route:{route.name}:ref:{ref_hash}"
                embedding_bytes = embeddings[i].tobytes() if hasattr(embeddings[i], 'tobytes') else np.array(embeddings[i]).tobytes()
                pipe.setex(key, self.ttl, embedding_bytes)

                pipe.sadd(f"{self.name}:route:{route.name}:refs", key)

            pipe.execute()

            self.redis.sadd(f"{self.name}:route_names", route.name)
            self.redis.expire(f"{self.name}:route_names", self.ttl)

    def add_route(self, questions: List[str], target: str, metadata: Optional[Dict[str, Any]] = None, distance_threshold: float = 0.3):
        """添加路由（通过问题列表和目标）"""
        route = Route(
            name=target,
            references=questions,
            metadata=metadata,
            distance_threshold=distance_threshold
        )
        self.add_route_by_object(route)

    def _get_cached_result(self, question: str) -> Optional[str]:
        """从缓存获取结果"""
        q_hash = hashlib.md5(question.encode()).hexdigest()
        cached = self.redis.get(f"{self.name}:cache:{q_hash}")
        if cached:
            return cached.decode() if isinstance(cached, bytes) else cached
        return None

    def _cache_result(self, question: str, result: str):
        """缓存结果"""
        q_hash = hashlib.md5(question.encode()).hexdigest()
        self.redis.setex(f"{self.name}:cache:{q_hash}", self.ttl, result)

    def route(self, question: str) -> Optional[str]:
        """根据问题路由到最匹配的类别"""
        cached = self._get_cached_result(question)
        if cached:
            return cached

        if not self.embedding_method:
            return None

        if not self.routes:
            return None

        question_embedding = self.embedding_method(question)
        if isinstance(question_embedding, list):
            question_embedding = np.array(question_embedding)

        best_match = None
        best_distance = float('inf')

        for route_name, route in self.routes.items():
            refs_key = f"{self.name}:route:{route_name}:refs"
            ref_keys = self.redis.smembers(refs_key)

            if not ref_keys:
                continue

            embeddings = []
            for key in ref_keys:
                emb = self.redis.get(key)
                if emb:
                    embeddings.append((key, np.frombuffer(emb, dtype=np.float32)))

            if not embeddings:
                continue

            route_embeddings = np.array([e[1] for e in embeddings])

            distances = np.linalg.norm(route_embeddings - question_embedding, axis=1)
            min_dist = float(np.min(distances))

            if min_dist < route.distance_threshold and min_dist < best_distance:
                best_distance = min_dist
                best_match = route_name

        if best_match:
            self._cache_result(question, best_match)

        return best_match

    def __call__(self, question: str) -> Optional[str]:
        """调用路由功能"""
        return self.route(question)

    def clear_cache(self):
        """清除路由缓存"""
        cache_keys = self.redis.keys(f"{self.name}:cache:*")
        if cache_keys:
            self.redis.delete(*cache_keys)

    def get_all_routes(self) -> List[str]:
        """获取所有路由名称"""
        return list(self.routes.keys())


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        return np.array([np.ones(128) * (hash(t) % 100) for t in text])

    router = SemanticRouter(
        name="topic-router",
        embedding_method=get_embedding,
        redis_url="localhost",
    )

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting",
        distance_threshold=300
    )
    router.add_route(
        questions=["How to return product", "如何退货"],
        target="refund",
        distance_threshold=300
    )

    print(router.route("Hi, good morning"))
    print(router.route("Hi, good afternoon"))
    print(router.route("如何退货"))
    print(router.get_all_routes())