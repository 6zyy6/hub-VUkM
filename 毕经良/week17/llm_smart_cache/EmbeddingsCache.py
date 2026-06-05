import numpy as np
import redis
from typing import Union, List
import hashlib


class EmbeddingsCache:
    def __init__(
            self,
            name: str,
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
    ):
        """
        嵌入缓存 - 缓存文本到向量的转换结果，避免重复调用embedding模型

        Args:
            name: 缓存名称
            ttl: 过期时间（秒）
            redis_url: Redis主机地址
            redis_port: Redis端口
            redis_password: Redis密码
        """
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False
        )
        self.ttl = ttl

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

        try:
            with self.redis.pipeline() as pipe:
                for i, t in enumerate(text):
                    t_code = hashlib.md5(t.encode()).hexdigest()
                    key = f"{self.name}:{t_code}"

                    emb = embedding[i]
                    if hasattr(emb, 'tobytes'):
                        value = emb.tobytes()
                    else:
                        value = np.array(emb).tobytes()

                    pipe.setex(key, self.ttl, value)

                return pipe.execute()
        except Exception as e:
            print(f"Store error: {e}")
            return -1

    def call(self, text: Union[List[str], str]) -> List:
        """
        根据文本获取缓存的嵌入向量

        Args:
            text: 文本（单条或列表）

        Returns:
            嵌入向量列表，如果没有缓存返回None
        """
        if isinstance(text, str):
            text = [text]

        try:
            key_list = []
            for t in text:
                t_code = hashlib.md5(t.encode()).hexdigest()
                key_list.append(f"{self.name}:{t_code}")

            results = self.redis.mget(key_list)

            if not results or all(r is None for r in results):
                return None

            embeddings = []
            for result in results:
                if result is None:
                    embeddings.append(None)
                else:
                    embedding = np.frombuffer(result, dtype=np.float32)
                    embeddings.append(embedding)

            return embeddings

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
                t_code = hashlib.md5(t.encode()).hexdigest()
                key_list.append(f"{self.name}:{t_code}")

            return self.redis.delete(*key_list)
        except Exception as e:
            print(f"Delete error: {e}")
            return -1

    def clear_all(self):
        """清除所有嵌入缓存"""
        pattern = f"{self.name}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)


if __name__ == "__main__":
    embed_cache = EmbeddingsCache(
        name="embedding_cache",
        ttl=360,
        redis_url="localhost",
    )

    def get_embedding(text):
        return np.random.rand(768).astype(np.float32)

    print("store:", embed_cache.store(text="hello world", embedding=get_embedding("hello world")))
    print("call:", embed_cache.call(text="hello world"))
    print("delete:", embed_cache.delete(text="hello world"))