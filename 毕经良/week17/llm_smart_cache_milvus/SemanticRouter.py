"""
SemanticRouter - 语义路由 (Milvus版本)
基于语义相似度的意图识别/路由
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable, Union
import numpy as np
import redis
import hashlib
import json
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


@dataclass
class Route:
    """路由定义"""
    name: str
    references: List[str]
    metadata: Optional[Dict[str, Any]] = None
    distance_threshold: float = 0.3


class SemanticRouter:
    def __init__(
            self,
            name: str,
            embedding_method: Optional[Callable[[Union[str, List[str]]], Any]] = None,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            ttl: int = 3600 * 24,
            milvus_uri: Optional[str] = None,
            milvus_token: Optional[str] = None,
            vector_dimension: Optional[int] = None,
    ):
        """
        语义路由 - 使用Milvus存储路由参考向量，Redis存储缓存

        Args:
            name: 路由名称
            embedding_method: 文本嵌入方法
            ttl: 过期时间（秒）
            redis_url: Redis主机地址
            redis_port: Redis端口
            redis_password: Redis密码
            milvus_uri: Milvus连接URI
            milvus_token: Milvus连接token
            vector_dimension: 向量维度
        """
        self.name = name
        self.embedding_method = embedding_method
        self.ttl = ttl
        self.vector_dimension = vector_dimension

        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False
        )

        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self._collection = None
        self._connected = False
        self.routes: Dict[str, Route] = {}

    def _connect(self):
        """连接Milvus"""
        if self._connected:
            return

        if self.milvus_uri and self.milvus_token:
            alias = f"router_{self.name}"
            connections.connect(
                alias=alias,
                uri=self.milvus_uri,
                token=self.milvus_token
            )
            self._alias = alias
            self._connected = True
        else:
            raise ValueError("Milvus URI and token are required")

    def _ensure_collection(self):
        """确保Collection存在"""
        self._connect()

        collection_name = f"router_{self.name}".replace("-", "_")
        self._alias = f"router_{self.name}"

        if utility.has_collection(collection_name, using=self._alias):
            self._collection = Collection(collection_name, using=self._alias)
            self._collection.load()
        else:
            if self.vector_dimension is None:
                raise ValueError("vector_dimension must be specified for new collection")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="route_name", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="ref_text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
            ]
            schema = CollectionSchema(fields=fields, description=f"Router for {self.name}")
            self._collection = Collection(name=collection_name, schema=schema, using=self._alias)

            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            self._collection.create_index(field_name="embedding", index_params=index_params)
            self._collection.load()

    def add_route_by_object(self, route: Route):
        """添加路由对象"""
        self.routes[route.name] = route

        if self.embedding_method:
            embeddings = self.embedding_method(route.references)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)

            if self.vector_dimension is None:
                self.vector_dimension = embeddings.shape[1]

            self._ensure_collection()

            try:
                ref_texts = route.references
                route_names = [route.name] * len(ref_texts)
                embeddings_list = embeddings.tolist()

                data = [route_names, ref_texts, embeddings_list]
                self._collection.insert(data)
                self._collection.flush()
            except Exception as e:
                print(f"Add route error: {e}")

            # 存储到Redis用于快速查找
            pipe = self.redis.pipeline()
            for ref in route.references:
                ref_hash = hashlib.md5(ref.encode()).hexdigest()
                pipe.sadd(f"{self.name}:route:{route.name}:refs", ref_hash)
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

        self._ensure_collection()

        try:
            question_embedding = self.embedding_method(question)
            if isinstance(question_embedding, list):
                question_embedding = np.array(question_embedding, dtype=np.float32)
            question_embedding = question_embedding.reshape(1, -1)

            search_params = {"params": {"nprobe": 10}, "metric_type": "L2", "offset": 0}
            results = self._collection.search(
                data=question_embedding.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=10,
                output_fields=["route_name", "ref_text"]
            )

            if not results or not results[0]:
                return None

            # 按距离分组，找最小距离
            route_distances = {}
            for result in results[0]:
                route_name = result.entity.get("route_name", "")
                distance = result.distance

                if route_name in self.routes:
                    threshold = self.routes[route_name].distance_threshold
                    if distance <= threshold:
                        if route_name not in route_distances or distance < route_distances[route_name]:
                            route_distances[route_name] = distance

            if not route_distances:
                return None

            # 返回距离最小的
            best_match = min(route_distances, key=route_distances.get)
            self._cache_result(question, best_match)
            return best_match

        except Exception as e:
            print(f"Route error: {e}")
            return None

    def __call__(self, question: str) -> Optional[str]:
        """调用路由功能"""
        return self.route(question)

    def clear_cache(self):
        """清除路由缓存"""
        cache_keys = self.redis.keys(f"{self.name}:cache:*")
        if cache_keys:
            self.redis.delete(*cache_keys)

        # 删除Milvus Collection
        if self._connected:
            collection_name = f"router_{self.name}".replace("-", "_")
            try:
                utility.drop_collection(collection_name, using=self._alias)
            except:
                pass
            self._collection = None

    def get_all_routes(self) -> List[str]:
        """获取所有路由名称"""
        return list(self.routes.keys())

    def close(self):
        """关闭连接"""
        if self._connected:
            connections.disconnect(alias=self._alias)
            self._connected = False


if __name__ == "__main__":
    MILVUS_URI = "https://in03-6fc9fda7586c8a5.serverless.aws-eu-central-1.cloud.zilliz.com"
    MILVUS_TOKEN = "319f97861036cbada2e4af735478028c1dda6e728b875e7d698472763eed54c46927310d70760cad623df9071587e2cb19f48637"

    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        return np.random.rand(len(text), 128).astype(np.float32)

    router = SemanticRouter(
        name="test-router",
        embedding_method=get_embedding,
        redis_url="localhost",
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        vector_dimension=128,
    )

    print("添加路由...")
    router.add_route(
        questions=["hello", "hi", "good morning"],
        target="greeting",
        distance_threshold=1.0
    )
    router.add_route(
        questions=["how to return", "如何退货", "return product"],
        target="refund",
        distance_threshold=1.0
    )

    print("路由测试...")
    result1 = router.route("hello")
    print(f"  'hello' -> {result1}")

    result2 = router.route("如何退货")
    print(f"  '如何退货' -> {result2}")

    print("缓存测试（第二次调用应走缓存）...")
    result1_cached = router.route("hello")
    print(f"  'hello' (cached) -> {result1_cached}")

    print("获取所有路由:")
    routes = router.get_all_routes()
    print(f"  routes: {routes}")

    router.clear_cache()
    router.close()
    print("测试完成!")