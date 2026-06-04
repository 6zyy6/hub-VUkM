"""
SemanticCache - 语义缓存 (Milvus版本)
存储LLM调用的问题和回答，通过语义相似度快速获取答案
"""

import numpy as np
import redis
from typing import Optional, List, Union, Any, Callable
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold: float = 0.1,
            milvus_uri: Optional[str] = None,
            milvus_token: Optional[str] = None,
            vector_dimension: Optional[int] = None,
    ):
        """
        语义缓存 - 使用Milvus存储向量，Redis存储问答对

        Args:
            name: 缓存名称
            embedding_method: 文本嵌入方法
            ttl: 过期时间（秒）
            redis_url: Redis主机地址
            redis_port: Redis端口
            redis_password: Redis密码
            distance_threshold: 距离阈值，小于此阈值认为相似
            milvus_uri: Milvus连接URI
            milvus_token: Milvus连接token
            vector_dimension: 向量维度，需要预先指定
        """
        self.name = name
        self.embedding_method = embedding_method
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.vector_dimension = vector_dimension

        # Redis客户端 - 存储问答
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False
        )

        # Milvus连接
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self._collection = None
        self._connected = False

    def _connect(self):
        """连接Milvus"""
        if self._connected:
            return

        if self.milvus_uri and self.milvus_token:
            alias = f"semcache_{self.name}"
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

        collection_name = f"semantic_cache_{self.name}".replace("-", "_")
        self._alias = f"semcache_{self.name}"

        if utility.has_collection(collection_name, using=self._alias):
            self._collection = Collection(collection_name, using=self._alias)
            self._collection.load()
        else:
            # 创建Collection
            if self.vector_dimension is None:
                raise ValueError("vector_dimension must be specified for new collection")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
            ]
            schema = CollectionSchema(fields=fields, description=f"Semantic cache for {self.name}")
            self._collection = Collection(name=collection_name, schema=schema, using=self._alias)

            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            self._collection.create_index(field_name="embedding", index_params=index_params)
            self._collection.load()

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]) -> int:
        """
        存储问题-回答对

        Args:
            prompt: 问题（单条或列表）
            response: 回答（单条或列表）

        Returns:
            操作结果状态
        """
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        if len(prompt) != len(response):
            raise ValueError("prompt和response数量必须一致")

        # 获取embeddings
        embeddings = self.embedding_method(prompt)
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        # 确保vector_dimension
        if self.vector_dimension is None:
            self.vector_dimension = embeddings.shape[1]

        # 确保Collection存在
        self._ensure_collection()

        try:
            # 转换为浮点数数组
            embeddings = np.array(embeddings, dtype=np.float32)
            embeddings_list = embeddings.tolist()

            # 批量插入Milvus
            data = [prompt, embeddings_list]
            result = self._collection.insert(data)
            ids = result.primary_keys

            self._collection.flush()

            # 存储到Redis (id -> response)
            with self.redis.pipeline() as pipe:
                for q_id, q, a in zip(ids, prompt, response):
                    a_bytes = a.encode() if isinstance(a, str) else a
                    pipe.setex(f"{self.name}:response:{q_id}", self.ttl, a_bytes)
                    # 存储 id -> prompt 的映射，方便后面查找
                    q_bytes = q.encode() if isinstance(q, str) else q
                    pipe.setex(f"{self.name}:prompt:{q_id}", self.ttl, q_bytes)

                return pipe.execute()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: str, top_k: int = 10) -> Optional[List[bytes]]:
        """
        通过语义相似度查找缓存的回答

        Args:
            prompt: 问题
            top_k: 返回最多k条结果

        Returns:
            匹配的回答列表
        """
        if not self._collection:
            self._ensure_collection()

        try:
            # 获取embedding
            embedding = self.embedding_method(prompt)
            if isinstance(embedding, list):
                embedding = np.array(embedding)

            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)

            embedding = np.array(embedding, dtype=np.float32)

            # 搜索
            search_params = {"params": {"nprobe": 10}, "metric_type": "L2", "offset": 0}
            results = self._collection.search(
                data=embedding.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["id", "prompt"]
            )

            if not results or not results[0]:
                return None

            # 过滤距离超过阈值的
            matched_ids = []
            for result in results[0]:
                distance = result.distance
                if distance <= self.distance_threshold:
                    matched_ids.append(result.id)

            if not matched_ids:
                return None

            # 从Redis获取responses
            response_keys = [f"{self.name}:response:{id}" for id in matched_ids]
            responses = self.redis.mget(response_keys)

            return responses

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def check(self, prompt: str) -> Optional[str]:
        """检查缓存并返回第一个匹配结果"""
        results = self.call(prompt)
        if results and results[0]:
            result = results[0]
            return result.decode() if isinstance(result, bytes) else result
        return None

    def clear_cache(self):
        """清除所有缓存数据"""
        # 删除Redis数据
        response_keys = self.redis.keys(f"{self.name}:response:*")
        prompt_keys = self.redis.keys(f"{self.name}:prompt:*")

        all_keys = response_keys + prompt_keys
        if all_keys:
            self.redis.delete(*all_keys)

        # 删除Milvus Collection
        if self._connected:
            collection_name = f"semantic_cache_{self.name}".replace("-", "_")
            try:
                utility.drop_collection(collection_name, using=self._alias)
            except:
                pass
            self._collection = None

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

    cache = SemanticCache(
        name="test_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        vector_dimension=128,
        distance_threshold=1.0
    )

    cache.clear_cache()

    print("存储: 你好世界")
    cache.store(prompt="你好世界", response="你好世界的回答")

    print("检查: 你好世界")
    result = cache.check(prompt="你好世界")
    print(f"结果: {result}")

    cache.close()
    print("测试完成!")