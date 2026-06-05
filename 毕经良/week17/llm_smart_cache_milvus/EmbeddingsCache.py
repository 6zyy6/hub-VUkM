"""
EmbeddingsCache - 嵌入缓存 (Milvus版本)
缓存文本到向量的转换结果，避免重复调用embedding模型
"""

import numpy as np
import redis
from typing import Union, List, Optional
import hashlib
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


class EmbeddingsCache:
    def __init__(
            self,
            name: str,
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            milvus_uri: Optional[str] = None,
            milvus_token: Optional[str] = None,
            vector_dimension: Optional[int] = None,
    ):
        """
        嵌入缓存 - 使用Milvus存储向量，Redis存储文本哈希映射

        Args:
            name: 缓存名称
            ttl: 过期时间（秒）
            redis_url: Redis主机地址
            redis_port: Redis端口
            redis_password: Redis密码
            milvus_uri: Milvus连接URI
            milvus_token: Milvus连接token
            vector_dimension: 向量维度
        """
        self.name = name
        self.ttl = ttl
        self.vector_dimension = vector_dimension

        # Redis客户端
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
            alias = f"embed_cache_{self.name}"
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

        collection_name = f"embed_cache_{self.name}".replace("-", "_")
        self._alias = f"embed_cache_{self.name}"

        if utility.has_collection(collection_name, using=self._alias):
            self._collection = Collection(collection_name, using=self._alias)
            self._collection.load()
        else:
            if self.vector_dimension is None:
                raise ValueError("vector_dimension must be specified for new collection")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text_hash", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
            ]
            schema = CollectionSchema(fields=fields, description=f"Embeddings cache for {self.name}")
            self._collection = Collection(name=collection_name, schema=schema, using=self._alias)

            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            self._collection.create_index(field_name="embedding", index_params=index_params)
            self._collection.load()

    def store(self, text: Union[List[str], str], embedding: np.ndarray) -> List:
        """
        存储文本及其嵌入向量

        Args:
            text: 文本（单条或列表）
            embedding: 对应的嵌入向量

        Returns:
            操作结果状态
        """
        if isinstance(text, str):
            text = [text]

        embedding = np.array(embedding, dtype=np.float32)

        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        if self.vector_dimension is None:
            self.vector_dimension = embedding.shape[1]

        self._ensure_collection()

        try:
            text_hashes = [hashlib.md5(t.encode()).hexdigest() for t in text]

            # 插入Milvus
            embeddings_list = embedding.tolist()
            data = [text_hashes, embeddings_list]
            result = self._collection.insert(data)
            self._collection.flush()

            # 同时存储到Redis作为快速缓存
            with self.redis.pipeline() as pipe:
                for t_hash, emb in zip(text_hashes, embedding):
                    key = f"{self.name}:hash:{t_hash}"
                    emb_bytes = np.array(emb).tobytes()
                    pipe.setex(key, self.ttl, emb_bytes)
                pipe.execute()

            return [True] * len(text)

        except Exception as e:
            print(f"Store error: {e}")
            return -1

    def call(self, text: Union[List[str], str]) -> Optional[List]:
        """
        根据文本获取缓存的嵌入向量

        Args:
            text: 文本（单条或列表）

        Returns:
            嵌入向量列表，如果没有缓存返回None
        """
        if isinstance(text, str):
            text = [text]

        self._ensure_collection()

        try:
            text_hashes = [hashlib.md5(t.encode()).hexdigest() for t in text]

            # 先通过文本哈希在Redis中查找是否有记录
            # 然后从Milvus获取embedding
            # 这里简化处理：直接查Redis中的哈希是否存在
            cached_embeddings = []
            for t in text:
                t_hash = hashlib.md5(t.encode()).hexdigest()
                key = f"{self.name}:hash:{t_hash}"
                cached = self.redis.get(key)
                if cached:
                    cached_embeddings.append(np.frombuffer(cached, dtype=np.float32))
                else:
                    cached_embeddings.append(None)

            if all(e is not None for e in cached_embeddings):
                return cached_embeddings

            # 如果Redis没有，使用Milvus查询
            # 由于Milvus不存储原始文本，这里需要用文本哈希去Milvus查找
            # 这是一个简化实现，实际应该用embedding去搜索
            return None

        except Exception as e:
            print(f"Call error: {e}")
            return None

    def delete(self, text: Union[List[str], str]) -> int:
        """
        删除指定文本的缓存

        Args:
            text: 文本（单条或列表）

        Returns:
            删除的键数量
        """
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []
            for t in text:
                t_hash = hashlib.md5(t.encode()).hexdigest()
                key_list.append(f"{self.name}:hash:{t_hash}")

            return self.redis.delete(*key_list)
        except Exception as e:
            print(f"Delete error: {e}")
            return -1

    def clear_all(self):
        """清除所有嵌入缓存"""
        # 清除Redis
        pattern = f"{self.name}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

        # 清除Milvus Collection
        if self._connected:
            collection_name = f"embed_cache_{self.name}".replace("-", "_")
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

    cache = EmbeddingsCache(
        name="test_embed",
        ttl=360,
        redis_url="localhost",
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        vector_dimension=128,
    )

    def get_embedding(text):
        return np.random.rand(128).astype(np.float32)

    text = "hello world"
    embedding = get_embedding(text)

    print("存储 embedding...")
    result = cache.store(text=text, embedding=embedding)
    print(f"store result: {result}")

    print("获取缓存...")
    cached = cache.call(text=text)
    print(f"call result: {cached}")

    print("删除缓存...")
    result = cache.delete(text=text)
    print(f"delete result: {result}")

    cache.close()
    print("测试完成!")