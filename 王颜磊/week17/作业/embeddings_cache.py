"""
EmbeddingsCache — 基于 RedisVL 的 Embedding 向量缓存

对照课程原始实现：
- 原始用 MD5 哈希 + Redis String 存储 numpy 向量 bytes
- 升级为 RedisVL SearchIndex + 向量字段，支持更丰富的元数据与索引管理

功能：
- store(text, embedding)  — 将文本+Embedding 存入 Redis 向量索引
- call(text)               — 根据文本检索已缓存的 Embedding
- delete(text)              — 删除指定文本的缓存条目
"""

import hashlib
import numpy as np
from typing import Optional, List, Union
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery
from redisvl.query.filter import Tag
from redisvl.schema import IndexSchema


INDEX_SCHEMA = {
    "index": {
        "name": "embedding_cache",
        "prefix": "emb",
        "storage_type": "json",
    },
    "fields": [
        {"name": "text_hash", "type": "tag"},
        {"name": "text", "type": "text"},
        {"name": "embedding", "type": "vector",
         "attrs": {
             "algorithm": "flat",
             "dims": 768,
             "distance_metric": "cosine",
             "datatype": "float32",
         }},
    ],
}


class EmbeddingsCache:
    """基于 RedisVL 的 Embedding 缓存层。

    将文本的 MD5 作为 tag 字段索引，embedding 向量存入 Redis 向量索引。
    相比原始实现的优势：
    - 不再需要手动 tobytes() / frombuffer() 序列化
    - 支持通过 RedisVL 的 SearchIndex 进行批量操作
    - 向量字段可被下游语义搜索复用
    """

    def __init__(
        self,
        name: str = "embedding_cache",
        ttl: int = 86400,
        redis_url: str = "redis://localhost:6379",
        redis_password: str = None,
        dims: int = 768,
    ):
        self.name = name
        self.ttl = ttl
        self.dims = dims

        conn_kwargs = {}
        if redis_password:
            conn_kwargs["password"] = redis_password

        self.redis = Redis.from_url(redis_url, **conn_kwargs)

        # 构建 schema
        schema_dict = INDEX_SCHEMA.copy()
        schema_dict["index"]["name"] = self.name
        schema_dict["fields"][2]["attrs"]["dims"] = self.dims
        self.schema = IndexSchema.from_dict(schema_dict)

        self.index = SearchIndex(schema=self.schema, redis_client=self.redis)
        self.index.create(overwrite=False)

    def store(self, text: Union[List[str], str], embedding: np.ndarray) -> bool:
        """存储文本及其 Embedding 向量。

        Args:
            text:  单个文本或文本列表
            embedding: shape (n, dims) 的 numpy 数组
        """
        if isinstance(text, str):
            text = [text]
            embedding = embedding.reshape(1, -1)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        records = []
        for i, t in enumerate(text):
            t_hash = hashlib.md5(t.encode()).hexdigest()
            records.append({
                "text_hash": t_hash,
                "text": t,
                "embedding": embedding[i].astype(np.float32).tolist(),
            })

        self.index.load(records, id_field="text_hash")
        return True

    def delete(self, text: Union[List[str], str]) -> int:
        """删除指定文本的缓存条目。

        Args:
            text: 单个文本或文本列表
        Returns:
            成功删除的 key 数量
        """
        if isinstance(text, str):
            text = [text]

        keys = []
        for t in text:
            t_hash = hashlib.md5(t.encode()).hexdigest()
            keys.append(f"{self.schema.index.prefix}:{t_hash}")

        return self.redis.delete(*keys)

    def call(self, text: Union[List[str], str]) -> Optional[List[Optional[np.ndarray]]]:
        """根据文本检索已缓存的 Embedding。

        Args:
            text: 单个文本或文本列表
        Returns:
            Embedding 列表，未命中项为 None
        """
        if isinstance(text, str):
            text = [text]

        keys = []
        hash_to_idx = {}
        for i, t in enumerate(text):
            t_hash = hashlib.md5(t.encode()).hexdigest()
            keys.append(f"{self.schema.index.prefix}:{t_hash}")
            hash_to_idx[t_hash] = i

        # 批量 JSON.MGET
        try:
            results = self.redis.json().mget(keys, "$.embedding")
        except Exception:
            results = [None] * len(keys)

        embeddings: List[Optional[np.ndarray]] = [None] * len(text)
        for key, result in zip(keys, results):
            if result and len(result) > 0:
                t_hash = key.split(":", 1)[1]
                idx = hash_to_idx[t_hash]
                embeddings[idx] = np.array(result[0], dtype=np.float32)

        return embeddings

    def clear(self) -> int:
        """清空全部缓存。"""
        keys = self.redis.keys(f"{self.schema.index.prefix}:*")
        if keys:
            return self.redis.delete(*keys)
        return 0


if __name__ == "__main__":
    cache = EmbeddingsCache(
        name="test_embedding_cache",
        redis_url="redis://localhost:6379",
    )

    # 模拟 Embedding 函数
    def get_embedding(text):
        return np.random.rand(768).astype(np.float32)

    print("=== EmbeddingsCache 测试 ===")
    cache.store(text="hello world", embedding=get_embedding("hello world"))
    result = cache.call(text="hello world")
    print(f"store & call: shape={result[0].shape}, dtype={result[0].dtype}")

    cache.store(
        text=["foo", "bar", "baz"],
        embedding=np.random.rand(3, 768).astype(np.float32),
    )
    results = cache.call(text=["foo", "bar", "not_exist"])
    print(f"batch call: hit={[r is not None for r in results]}")

    cache.delete(text="hello world")
    after_delete = cache.call(text="hello world")
    print(f"after delete: {after_delete}")

    cache.clear()
    print("测试完成")
