import numpy as np
import redis
from typing import Optional, List, Union, Any, Callable
import os


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
            index_path: Optional[str] = None
    ):
        """
        语义缓存 - 存储LLM调用的问题和回答，通过语义相似度快速获取答案

        Args:
            name: 缓存名称
            embedding_method: 文本嵌入方法
            ttl: 过期时间（秒）
            redis_url: Redis主机地址
            redis_port: Redis端口
            redis_password: Redis密码
            distance_threshold: 距离阈值，小于此阈值认为相似
            index_path: FAISS索引存储路径，默认使用name参数
        """
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        if index_path:
            self.index_file = index_path
        else:
            self.index_file = f"{name}.index"

        if os.path.exists(self.index_file):
            try:
                import faiss
                self.index = faiss.read_index(self.index_file)
            except Exception as e:
                print(f"Warning: Could not load index: {e}")
                self.index = None
        else:
            self.index = None

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]) -> int:
        """
        存储问题-回答对

        Args:
            prompt: 问题（单条或列表）
            response: 回答（单条或列表）

        Returns:
            操作结果状态
        """
        import faiss

        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        if len(prompt) != len(response):
            raise ValueError("prompt和response数量必须一致")

        embeddings = self.embedding_method(prompt)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_file)

        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    q_bytes = q.encode() if isinstance(q, str) else q
                    a_bytes = a.encode() if isinstance(a, str) else a
                    pipe.setex(self.name + ":key:" + q_bytes.hex(), self.ttl, a_bytes)
                    pipe.lpush(self.name + ":list", q_bytes)

                return pipe.execute()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: str) -> Optional[List[bytes]]:
        """
        通过语义相似度查找缓存的回答

        Args:
            prompt: 问题

        Returns:
            匹配的回答列表，如果没有匹配返回None
        """
        import faiss

        if self.index is None:
            return None

        embedding = self.embedding_method(prompt)
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        k = min(100, self.index.ntotal)
        if k == 0:
            return None

        dis, ind = self.index.search(embedding, k=k)

        if dis[0][0] > self.distance_threshold:
            return None

        filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]

        prompts = self.redis.lrange(self.name + ":list", 0, -1)

        filtered_prompts = []
        for i in filtered_ind:
            if i < len(prompts):
                p = prompts[i]
                if isinstance(p, bytes):
                    filtered_prompts.append(self.name + ":key:" + p.hex())
                else:
                    filtered_prompts.append(self.name + ":key:" + p.encode().hex())

        if not filtered_prompts:
            return None

        results = self.redis.mget(filtered_prompts)
        return results

    def check(self, prompt: str) -> Optional[str]:
        """检查缓存并返回第一个匹配结果"""
        results = self.call(prompt)
        if results and results[0]:
            result = results[0]
            return result.decode() if isinstance(result, bytes) else result
        return None

    def clear_cache(self):
        """清除所有缓存数据"""
        import faiss

        prompts = self.redis.lrange(self.name + ":list", 0, -1)
        keys_to_delete = [self.name + ":key:" + p.hex() if isinstance(p, bytes) else self.name + ":key:" + p.encode().hex() for p in prompts]

        if keys_to_delete:
            self.redis.delete(*keys_to_delete)

        self.redis.delete(self.name + ":list")

        if os.path.exists(self.index_file):
            os.unlink(self.index_file)

        self.index = None


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        return np.array([np.ones(128) for _ in text])

    embed_cache = SemanticCache(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()

    embed_cache.store(prompt="hello world", response="hello world response")
    print("check:", embed_cache.check(prompt="hello world"))

    embed_cache.store(prompt="今天天气怎么样", response="今天天气很好")
    print("check:", embed_cache.check(prompt="今天天气怎么样"))