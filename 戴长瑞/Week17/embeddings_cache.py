"""
EmbeddingsCache - 缓存文本embedding向量，避免重复生成

功能：
- 精确键匹配存储/读取embedding
- 支持TTL过期
- 支持批量操作
- 支持自定义embedding模型和Redis客户端

依赖：redisvl >= 0.2.0, redis, pydantic

使用示例：
    # 基础用法
    cache = EmbeddingsCache(name="my_embeddings", ttl=3600)
    cache.set("hello", "text-embedding-ada-002", [0.1, 0.2, ...])
    result = cache.get("hello", "text-embedding-ada-002")

    # 自定义Redis客户端和Embedding模型
    from redis import Redis
    from redisvl.utils.vectorize import OpenAIVectorizer

    client = Redis(host='localhost', port=6379, db=0)
    vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

    cache = EmbeddingsCache(
        name="my_embeddings",
        redis_client=client,
        vectorizer=vectorizer,
        ttl=7200
    )

    # 异步操作
    import asyncio
    async def demo():
        await cache.aset("query", "ada", [0.1, 0.2])
        result = await cache.aget("query", "ada")
"""

from typing import Any, Iterable

from redisvl.extensions.cache.base import BaseCache
from redisvl.extensions.cache.embeddings.schema import CacheEntry
from redisvl.redis.utils import convert_bytes, hashify
from redisvl.types import AsyncRedisClient, SyncRedisClient
from redisvl.utils.log import get_logger
from redisvl.utils.vectorize.base import BaseVectorizer

logger = get_logger(__name__)


class EmbeddingsCache(BaseCache):
    """Embeddings缓存 - 存储文本embedding向量，支持精确匹配和TTL"""

    _warning_shown: bool = False

    def __init__(
        self,
        name: str = "embedcache",
        ttl: int | None = None,
        redis_client: SyncRedisClient | None = None,
        async_redis_client: AsyncRedisClient | None = None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: dict[str, Any] = {},
        vectorizer: BaseVectorizer | None = None,
        model_name: str = "default",
        **kwargs,
    ):
        """初始化Embeddings缓存

        Args:
            name: 缓存名称，默认"embedcache"
            ttl: 过期时间（秒），默认None永不过期
            redis_client: Redis客户端实例（同步）
            async_redis_client: Redis客户端实例（异步）
            redis_url: Redis连接URL
            connection_kwargs: 连接参数
            vectorizer: 自定义向量化器（用于批处理等）
            model_name: 默认模型名称，用于生成缓存键
        """
        super().__init__(
            name=name,
            ttl=ttl,
            redis_client=redis_client,
            async_redis_client=async_redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs,
        )
        self._vectorizer = vectorizer
        self._model_name = model_name

    def _make_cache_key(self, content: bytes | str, model_name: str) -> str:
        """生成Redis键"""
        if isinstance(content, bytes):
            content = content.hex()
        entry_id = hashify(f"{content}:{model_name}")
        return self._make_key(entry_id)

    def _process_cache_data(self, data: dict[str, Any] | None) -> dict[str, Any] | None:
        """处理Redis数据为缓存条目"""
        if not data:
            return None
        cache_hit = CacheEntry(**convert_bytes(data))
        return cache_hit.model_dump(exclude_none=True)

    def _should_warn_async_only(self) -> bool:
        """检查是否仅有异步客户端"""
        return self._owns_redis_client is False and self._redis_client is None

    @property
    def vectorizer(self) -> BaseVectorizer | None:
        """获取向量化器"""
        return self._vectorizer

    @property
    def model_name(self) -> str:
        """获取默认模型名称"""
        return self._model_name

    # ==================== 同步操作 ====================

    def get(self, content: bytes | str, model_name: str | None = None) -> dict[str, Any] | None:
        """获取embedding

        Args:
            content: 文本内容
            model_name: 模型名称（可选，默认使用初始化时的模型名）

        Returns:
            缓存条目或None
        """
        model_name = model_name or self._model_name
        key = self._make_cache_key(content, model_name)
        return self.get_by_key(key)

    def get_by_key(self, key: str) -> dict[str, Any] | None:
        """通过键获取embedding"""
        if self._should_warn_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning("仅异步客户端初始化，请使用aget_by_key")
                EmbeddingsCache._warning_shown = True

        client = self._get_redis_client()
        data = client.hgetall(key)
        if data:
            self.expire(key)
        return self._process_cache_data(data)

    def mget_by_keys(self, keys: list[str]) -> list[dict[str, Any] | None]:
        """批量通过键获取embedding"""
        if not keys:
            return []

        if self._should_warn_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning("仅异步客户端初始化，请使用amget_by_keys")
                EmbeddingsCache._warning_shown = True

        client = self._get_redis_client()
        with client.pipeline(transaction=False) as pipeline:
            for key in keys:
                pipeline.hgetall(key)
            results = pipeline.execute()

        processed = []
        for i, result in enumerate(results):
            if result:
                self.expire(keys[i])
            processed.append(self._process_cache_data(result))
        return processed

    def mget(self, contents: list[bytes | str], model_name: str | None = None) -> list[dict[str, Any] | None]:
        """批量获取embedding

        Args:
            contents: 内容列表
            model_name: 模型名称（可选）

        Returns:
            缓存条目列表
        """
        model_name = model_name or self._model_name
        keys = [self._make_cache_key(c, model_name) for c in contents]
        return self.mget_by_keys(keys) if keys else []

    def set(
        self,
        content: bytes | str,
        model_name: str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """存储embedding

        Args:
            content: 文本内容
            model_name: 模型名称（可选）
            embedding: 向量（可选，若提供vectorizer可自动生成）
            metadata: 元数据
            ttl: TTL覆盖

        Returns:
            Redis键
        """
        model_name = model_name or self._model_name

        # 如果未提供embedding但有vectorizer，则自动生成
        if embedding is None and self._vectorizer is not None:
            content_str = content if isinstance(content, str) else content.decode('utf-8', errors='ignore')
            embedding = self._vectorizer.embed(content_str)

        if embedding is None:
            raise ValueError("必须提供embedding或配置vectorizer")

        if isinstance(content, bytes):
            content_hash = content.hex()
        else:
            content_hash = content

        entry_id = hashify(f"{content_hash}:{model_name}")
        key = self._make_key(entry_id)

        entry = CacheEntry(
            entry_id=entry_id,
            content=content,
            model_name=model_name,
            embedding=embedding,
            metadata=metadata,
        )

        if self._should_warn_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning("仅异步客户端初始化，请使用aset")
                EmbeddingsCache._warning_shown = True

        client = self._get_redis_client()
        client.hset(name=key, mapping=entry.to_dict())
        self.expire(key, ttl)
        return key

    def mset(
        self,
        items: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> list[str]:
        """批量存储embedding

        Args:
            items: [{"content": "...", "model_name": "...", "embedding": [...], "metadata": {...}}, ...]
            ttl: TTL覆盖

        Returns:
            Redis键列表
        """
        if not items:
            return []

        if self._should_warn_async_only():
            if not EmbeddingsCache._warning_shown:
                logger.warning("仅异步客户端初始化，请使用amset")
                EmbeddingsCache._warning_shown = True

        client = self._get_redis_client()
        keys = []

        # 自动生成embedding（如果需要）
        for item in items:
            if 'embedding' not in item and self._vectorizer is not None:
                content_str = item['content'] if isinstance(item['content'], str) else item['content'].decode('utf-8', errors='ignore')
                item['embedding'] = self._vectorizer.embed(content_str)

        with client.pipeline(transaction=False) as pipeline:
            for item in items:
                key, cache_entry = self._prepare_entry_data(**item)
                keys.append(key)
                pipeline.hset(name=key, mapping=cache_entry)
            pipeline.execute()

        for key in keys:
            self.expire(key, ttl)
        return keys

    def _prepare_entry_data(self, content, model_name, embedding, metadata=None):
        """准备存储数据"""
        model_name = model_name or self._model_name
        if isinstance(content, bytes):
            content_hash = content.hex()
        else:
            content_hash = content
        entry_id = hashify(f"{content_hash}:{model_name}")
        key = self._make_key(entry_id)
        entry = CacheEntry(
            entry_id=entry_id,
            content=content,
            model_name=model_name,
            embedding=embedding,
            metadata=metadata,
        )
        return key, entry.to_dict()

    def exists(self, content: bytes | str, model_name: str | None = None) -> bool:
        """检查embedding是否存在"""
        model_name = model_name or self._model_name
        client = self._get_redis_client()
        key = self._make_cache_key(content, model_name)
        return bool(client.exists(key))

    def mexists(self, contents: list[bytes | str], model_name: str | None = None) -> list[bool]:
        """批量检查embedding是否存在"""
        if not contents:
            return []
        model_name = model_name or self._model_name
        client = self._get_redis_client()
        keys = [self._make_cache_key(c, model_name) for c in contents]
        with client.pipeline(transaction=False) as pipeline:
            for key in keys:
                pipeline.exists(key)
            return [bool(r) for r in pipeline.execute()]

    def drop(self, content: bytes | str, model_name: str | None = None) -> None:
        """删除embedding"""
        model_name = model_name or self._model_name
        key = self._make_cache_key(content, model_name)
        self.drop_by_key(key)

    def drop_by_key(self, key: str) -> None:
        """通过键删除embedding"""
        client = self._get_redis_client()
        client.delete(key)

    def mdrop_by_keys(self, keys: list[str]) -> None:
        """批量通过键删除embedding"""
        if not keys:
            return
        client = self._get_redis_client()
        with client.pipeline(transaction=False) as pipeline:
            for key in keys:
                pipeline.delete(key)
            pipeline.execute()

    # ==================== 异步操作 ====================

    async def aget(self, content: bytes | str, model_name: str | None = None) -> dict[str, Any] | None:
        """异步获取embedding"""
        model_name = model_name or self._model_name
        key = self._make_cache_key(content, model_name)
        return await self.aget_by_key(key)

    async def aget_by_key(self, key: str) -> dict[str, Any] | None:
        """异步通过键获取embedding"""
        client = await self._get_async_redis_client()
        data = await client.hgetall(key)
        if data:
            await self.aexpire(key)
        return self._process_cache_data(data)

    async def amget_by_keys(self, keys: list[str]) -> list[dict[str, Any] | None]:
        """异步批量通过键获取embedding"""
        if not keys:
            return []
        client = await self._get_async_redis_client()
        async with client.pipeline(transaction=False) as pipeline:
            for key in keys:
                pipeline.hgetall(key)
            results = await pipeline.execute()

        processed = []
        for i, result in enumerate(results):
            if result:
                await self.aexpire(keys[i])
            processed.append(self._process_cache_data(result))
        return processed

    async def amget(self, contents: list[bytes | str], model_name: str | None = None) -> list[dict[str, Any] | None]:
        """异步批量获取embedding"""
        model_name = model_name or self._model_name
        keys = [self._make_cache_key(c, model_name) for c in contents]
        return await self.amget_by_keys(keys) if keys else []

    async def aset(
        self,
        content: bytes | str,
        model_name: str | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """异步存储embedding"""
        model_name = model_name or self._model_name

        if embedding is None and self._vectorizer is not None:
            content_str = content if isinstance(content, str) else content.decode('utf-8', errors='ignore')
            embedding = self._vectorizer.embed(content_str)

        if embedding is None:
            raise ValueError("必须提供embedding或配置vectorizer")

        if isinstance(content, bytes):
            content_hash = content.hex()
        else:
            content_hash = content

        entry_id = hashify(f"{content_hash}:{model_name}")
        key = self._make_key(entry_id)

        entry = CacheEntry(
            entry_id=entry_id,
            content=content,
            model_name=model_name,
            embedding=embedding,
            metadata=metadata,
        )

        client = await self._get_async_redis_client()
        await client.hset(name=key, mapping=entry.to_dict())
        await self.aexpire(key, ttl)
        return key

    async def amset(
        self,
        items: list[dict[str, Any]],
        ttl: int | None = None,
    ) -> list[str]:
        """异步批量存储embedding"""
        if not items:
            return []

        # 自动生成embedding（如果需要）
        for item in items:
            if 'embedding' not in item and self._vectorizer is not None:
                content_str = item['content'] if isinstance(item['content'], str) else item['content'].decode('utf-8', errors='ignore')
                item['embedding'] = self._vectorizer.embed(content_str)

        client = await self._get_async_redis_client()
        keys = []

        async with client.pipeline(transaction=False) as pipeline:
            for item in items:
                key, cache_entry = self._prepare_entry_data(**item)
                keys.append(key)
                await pipeline.hset(name=key, mapping=cache_entry)
            await pipeline.execute()

        for key in keys:
            await self.aexpire(key, ttl)
        return keys

    async def aexists(self, content: bytes | str, model_name: str | None = None) -> bool:
        """异步检查embedding是否存在"""
        model_name = model_name or self._model_name
        key = self._make_cache_key(content, model_name)
        return await self.aexists_by_key(key)

    async def aexists_by_key(self, key: str) -> bool:
        """异步通过键检查是否存在"""
        client = await self._get_async_redis_client()
        return bool(await client.exists(key))

    async def adrop(self, content: bytes | str, model_name: str | None = None) -> None:
        """异步删除embedding"""
        model_name = model_name or self._model_name
        key = self._make_cache_key(content, model_name)
        await self.adrop_by_key(key)

    async def adrop_by_key(self, key: str) -> None:
        """异步通过键删除embedding"""
        client = await self._get_async_redis_client()
        await client.delete(key)

    async def amdrop_by_keys(self, keys: list[str]) -> None:
        """异步批量通过键删除embedding"""
        if not keys:
            return
        client = await self._get_async_redis_client()
        await client.delete(*keys)


# ==================== 使用示例 ====================
"""
# 示例1: 基础用法
cache = EmbeddingsCache(name="my_embeddings", ttl=3600)
cache.set("hello world", "text-embedding-ada-002", [0.1, 0.2, 0.3, ...])
result = cache.get("hello world", "text-embedding-ada-002")
print(result)  # {'entry_id': '...', 'content': 'hello world', 'model_name': '...', 'embedding': [...], 'metadata': None}

# 示例2: 自定义Redis客户端和Embedding模型
from redis import Redis
from redisvl.utils.vectorize import OpenAIVectorizer

client = Redis(host='localhost', port=6379, db=0)
vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

cache = EmbeddingsCache(
    name="my_embeddings",
    redis_client=client,
    vectorizer=vectorizer,
    model_name="text-embedding-3-small",
    ttl=7200
)

# 自动生成embedding
cache.set("自动生成embedding", embedding=None)  # vectorizer会自动生成

# 示例3: 批量操作
cache.mset([
    {"content": "query1", "model_name": "ada", "embedding": [0.1, 0.2, ...]},
    {"content": "query2", "model_name": "ada", "embedding": [0.3, 0.4, ...]},
])

results = cache.mget(["query1", "query2"], "ada")

# 示例4: 异步操作
import asyncio

async def demo():
    cache = EmbeddingsCache(name="async_cache", redis_url="redis://localhost:6379")
    await cache.aset("async content", "ada", [0.1, 0.2, ...])
    result = await cache.aget("async content", "ada")
    print(result)

asyncio.run(demo())

# 示例5: 检查存在性
if cache.exists("hello world", "ada"):
    print("Embedding已缓存")

# 示例6: 删除
cache.drop("hello world", "ada")
"""