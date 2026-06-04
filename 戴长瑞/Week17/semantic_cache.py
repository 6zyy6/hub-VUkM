"""
SemanticCache - 语义缓存，根据用户查询的语义相似度命中缓存结果

功能：
- 基于向量相似度匹配缓存
- 可配置相似度阈值
- 支持TTL过期
- 支持自定义Redis客户端和Embedding模型

依赖：redisvl >= 0.2.0, redis, pydantic

使用示例：
    # 基础用法
    cache = SemanticCache(name="qa_cache", distance_threshold=0.1, ttl=3600)
    cache.store("What is Python?", "Python is a programming language.")
    result = cache.check("What is Python?")

    # 自定义Redis客户端和Embedding模型
    from redis import Redis
    from redisvl.utils.vectorize import OpenAIVectorizer

    client = Redis(host='localhost', port=6379, db=0)
    vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

    cache = SemanticCache(
        name="qa_cache",
        redis_client=client,
        vectorizer=vectorizer,
        distance_threshold=0.15,
        ttl=7200
    )

    # 过滤字段用法
    cache = SemanticCache(
        name="qa_cache",
        filterable_fields=[{"name": "category", "type": "tag"}]
    )
    cache.store("What is Python?", "Python is...", filters={"category": "programming"})
    hits = cache.check("Python question", filter_expression=Tag("category") == "programming")
"""

from typing import Any

from redisvl.extensions.cache.llm.base import BaseLLMCache
from redisvl.extensions.cache.llm.schema import CacheEntry, CacheHit
from redisvl.extensions.constants import (
    CACHE_VECTOR_FIELD_NAME,
    ENTRY_ID_FIELD_NAME,
    INSERTED_AT_FIELD_NAME,
    METADATA_FIELD_NAME,
    PROMPT_FIELD_NAME,
    RESPONSE_FIELD_NAME,
    UPDATED_AT_FIELD_NAME,
)
from redisvl.index import AsyncSearchIndex, SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.query.filter import FilterExpression, Tag
from redisvl.redis.utils import hashify
from redisvl.utils.log import get_logger
from redisvl.utils.utils import current_timestamp, serialize, validate_vector_dims
from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

logger = get_logger("[SemanticCache]")


class SemanticCacheIndexSchema:
    """语义缓存索引schema"""

    @classmethod
    def from_params(cls, name: str, prefix: str, vector_dims: int, dtype: str):
        from redisvl.schema import IndexSchema
        return IndexSchema(
            index={"name": name, "prefix": prefix},
            fields=[
                {"name": PROMPT_FIELD_NAME, "type": "text"},
                {"name": RESPONSE_FIELD_NAME, "type": "text"},
                {"name": INSERTED_AT_FIELD_NAME, "type": "numeric"},
                {"name": UPDATED_AT_FIELD_NAME, "type": "numeric"},
                {
                    "name": CACHE_VECTOR_FIELD_NAME,
                    "type": "vector",
                    "attrs": {
                        "dims": vector_dims,
                        "datatype": dtype,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        )


class SemanticCache(BaseLLMCache):
    """语义缓存 - 基于向量相似度匹配prompt缓存response"""

    def __init__(
        self,
        name: str = "semcache",
        distance_threshold: float = 0.1,
        ttl: int | None = None,
        vectorizer: BaseVectorizer | None = None,
        filterable_fields: list[dict[str, Any]] | None = None,
        redis_client=None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: dict[str, Any] = {},
        overwrite: bool = False,
        **kwargs,
    ):
        """初始化语义缓存

        Args:
            name: 缓存名称，默认"semcache"
            distance_threshold: 相似度阈值[0-2]，越小越严格，默认0.1
            ttl: 过期时间（秒），默认None永不过期
            vectorizer: 向量化器，默认HFTextVectorizer
                     支持：OpenAIVectorizer, CohereVectorizer, HuggingFaceVectorizer等
            filterable_fields: 可过滤字段列表
                     例如：[{"name": "category", "type": "tag"}, {"name": "version", "type": "numeric"}]
            redis_client: Redis客户端实例（同步）
            redis_url: Redis连接URL
            connection_kwargs: 连接参数
            overwrite: 是否覆盖已有索引
        """
        super().__init__(
            name=name,
            ttl=ttl,
            redis_client=redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs,
        )

        dtype = kwargs.pop("dtype", None)

        if vectorizer:
            if not isinstance(vectorizer, BaseVectorizer):
                raise TypeError("必须提供有效的vectorizer")
            if dtype and vectorizer.dtype != dtype:
                raise ValueError(f"dtype不匹配: {dtype} vs {vectorizer.dtype}")
            self._vectorizer = vectorizer
        else:
            vectorizer_kwargs = kwargs
            if dtype:
                vectorizer_kwargs["dtype"] = dtype
            self._vectorizer = HFTextVectorizer(
                model="sentence-transformers/all-mpnet-base-v2",
                **vectorizer_kwargs,
            )

        self.set_threshold(distance_threshold)

        self.return_fields = [
            ENTRY_ID_FIELD_NAME,
            PROMPT_FIELD_NAME,
            RESPONSE_FIELD_NAME,
            INSERTED_AT_FIELD_NAME,
            UPDATED_AT_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        schema = SemanticCacheIndexSchema.from_params(
            name, name, self._vectorizer.dims, self._vectorizer.dtype
        )
        schema = self._modify_schema(schema, filterable_fields)

        self._index = SearchIndex(
            schema=schema,
            redis_client=self._redis_client,
            redis_url=self.redis_kwargs["redis_url"],
            connection_kwargs=self.redis_kwargs["connection_kwargs"] or None,
        )
        self._aindex = None

        self.overwrite = overwrite
        if not self.overwrite and self._index.exists():
            existing_index = SearchIndex.from_existing(name, redis_client=self._index._redis_client)
            if existing_index.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(f"索引{name}已存在且schema不匹配，设置overwrite=True可覆盖")

        self._index.create(overwrite=self.overwrite, drop=False)

    def __repr__(self):
        return f"SemanticCache(name={self.name!r}, threshold={self.distance_threshold}, ttl={self.ttl})"

    def _modify_schema(self, schema, filterable_fields):
        """修改schema添加过滤字段"""
        if filterable_fields:
            protected = set(self.return_fields + ["id"])
            for f in filterable_fields:
                if f["name"] in protected:
                    raise ValueError(f"{f['name']}是保留字段名")
                schema.add_field(f)
                self.return_fields.append(f["name"])
        return schema

    async def _get_async_index(self) -> AsyncSearchIndex:
        """获取异步索引"""
        if self._aindex is None:
            async_client = await self._get_async_redis_client()
            self._aindex = AsyncSearchIndex(
                schema=self._index.schema,
                redis_client=async_client,
                redis_url=self.redis_kwargs["redis_url"],
                connection_kwargs=self.redis_kwargs["connection_kwargs"] or None,
            )
        return self._aindex

    @property
    def vectorizer(self) -> BaseVectorizer:
        """获取向量化器"""
        return self._vectorizer

    @property
    def distance_threshold(self) -> float:
        """获取相似度阈值"""
        return self._distance_threshold

    @property
    def index(self) -> SearchIndex:
        """获取底层SearchIndex"""
        return self._index

    def set_threshold(self, distance_threshold: float) -> None:
        """设置相似度阈值

        Args:
            distance_threshold: 阈值[0-2]，越小匹配越严格

        Raises:
            ValueError: 阈值超出范围
        """
        if not 0 <= float(distance_threshold) <= 2:
            raise ValueError(f"阈值必须在[0,2]之间，当前值: {distance_threshold}")
        self._distance_threshold = float(distance_threshold)

    def set_ttl(self, ttl: int | None) -> None:
        """设置默认TTL

        Args:
            ttl: 过期时间（秒），None表示永不过期
        """
        if ttl is not None and not isinstance(ttl, int):
            raise ValueError(f"TTL必须是整数，当前类型: {type(ttl)}")
        self._ttl = ttl

    def delete(self) -> None:
        """删除整个缓存索引"""
        self._index.delete(drop=True)

    async def adelete(self) -> None:
        """异步删除整个缓存索引"""
        aindex = await self._get_async_index()
        await aindex.delete(drop=True)

    def _check_vector_dims(self, vector: list[float]) -> None:
        """检查向量维度"""
        schema_dims = self._index.schema.fields[CACHE_VECTOR_FIELD_NAME].attrs.dims
        validate_vector_dims(len(vector), schema_dims)

    def _vectorize_prompt(self, prompt: str) -> list[float]:
        """将prompt向量化"""
        if not isinstance(prompt, str):
            raise TypeError("Prompt必须是字符串")
        return self._vectorizer.embed(prompt)

    async def _avectorize_prompt(self, prompt: str) -> list[float]:
        """异步将prompt向量化"""
        if not isinstance(prompt, str):
            raise TypeError("Prompt必须是字符串")
        return await self._vectorizer.aembed(prompt)

    def check(
        self,
        prompt: str | None = None,
        vector: list[float] | None = None,
        num_results: int = 1,
        return_fields: list[str] | None = None,
        filter_expression: FilterExpression | None = None,
        distance_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """检查语义缓存获取相似结果

        Args:
            prompt: 查询文本
            vector: 查询向量（与prompt二选一）
            num_results: 返回结果数，默认1
            return_fields: 返回字段列表，默认所有字段
            filter_expression: 过滤表达式，例如 Tag("category") == "tech"
            distance_threshold: 相似度阈值覆盖，默认使用初始化时的阈值

        Returns:
            缓存命中列表，每项包含prompt、response、vector_distance等

        示例:
            hits = cache.check("What is Python?")
            if hits:
                print(hits[0]["response"])
        """
        if not any([prompt, vector]):
            raise ValueError("必须提供prompt或vector")
        if return_fields and not isinstance(return_fields, list):
            raise TypeError("return_fields必须是列表")

        threshold = distance_threshold if distance_threshold is not None else self._distance_threshold

        if vector is None and prompt is not None:
            vector = self._vectorize_prompt(prompt)

        if vector is not None:
            self._check_vector_dims(vector)
        else:
            raise ValueError("无法生成有效向量")

        query = VectorRangeQuery(
            vector=vector,
            vector_field_name=CACHE_VECTOR_FIELD_NAME,
            return_fields=self.return_fields,
            distance_threshold=threshold,
            num_results=num_results,
            return_score=True,
            filter_expression=filter_expression,
            dtype=self._vectorizer.dtype,
        )

        results = self._index.query(query)
        redis_keys, cache_hits = self._process_results(results, return_fields)

        for key in redis_keys:
            self.expire(key)

        return cache_hits

    async def acheck(
        self,
        prompt: str | None = None,
        vector: list[float] | None = None,
        num_results: int = 1,
        return_fields: list[str] | None = None,
        filter_expression: FilterExpression | None = None,
        distance_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """异步检查语义缓存"""
        import asyncio

        aindex = await self._get_async_index()

        if not any([prompt, vector]):
            raise ValueError("必须提供prompt或vector")
        if return_fields and not isinstance(return_fields, list):
            raise TypeError("return_fields必须是列表")

        threshold = distance_threshold if distance_threshold is not None else self._distance_threshold

        if vector is None and prompt is not None:
            vector = await self._avectorize_prompt(prompt)

        if vector is not None:
            self._check_vector_dims(vector)
        else:
            raise ValueError("无法生成有效向量")

        query = VectorRangeQuery(
            vector=vector,
            vector_field_name=CACHE_VECTOR_FIELD_NAME,
            return_fields=self.return_fields,
            distance_threshold=threshold,
            num_results=num_results,
            return_score=True,
            filter_expression=filter_expression,
            dtype=self._vectorizer.dtype,
        )

        results = await aindex.query(query)
        redis_keys, cache_hits = self._process_results(results, return_fields)

        await asyncio.gather(*[self.aexpire(key) for key in redis_keys])
        return cache_hits

    def _process_results(self, search_results, return_fields):
        """处理搜索结果"""
        redis_keys = []
        cache_hits = []

        for result in search_results:
            redis_key = result.pop("id")
            redis_keys.append(redis_key)

            cache_hit = CacheHit(**result)
            hit_dict = cache_hit.to_dict()

            if isinstance(return_fields, list) and return_fields:
                hit_dict = {k: v for k, v in hit_dict.items() if k in return_fields}

            hit_dict["redis_key"] = redis_key
            cache_hits.append(hit_dict)

        return redis_keys, cache_hits

    def store(
        self,
        prompt: str,
        response: str,
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """存储prompt-response对

        Args:
            prompt: 用户查询
            response: LLM响应
            vector: 向量（可选，自动生成）
            metadata: 元数据（例如{"model": "gpt-4", "tokens": 100}）
            filters: 过滤条件（例如{"category": "tech", "version": 1}）
            ttl: TTL覆盖（秒），None使用默认TTL

        Returns:
            Redis键

        示例:
            key = cache.store(
                prompt="What is Python?",
                response="Python is a programming language.",
                metadata={"model": "gpt-4"},
                filters={"category": "programming"}
            )
        """
        vector = vector or self._vectorize_prompt(prompt)
        self._check_vector_dims(vector)

        entry_id = self._make_entry_id(prompt, filters)

        cache_entry = CacheEntry(
            entry_id=entry_id,
            prompt=prompt,
            response=response,
            prompt_vector=vector,
            metadata=metadata,
            filters=filters,
        )

        ttl = ttl if ttl is not None else self._ttl
        keys = self._index.load(
            data=[cache_entry.to_dict(self._vectorizer.dtype)],
            ttl=ttl,
            id_field=ENTRY_ID_FIELD_NAME,
        )
        return keys[0]

    async def astore(
        self,
        prompt: str,
        response: str,
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> str:
        """异步存储"""
        aindex = await self._get_async_index()

        vector = vector or await self._avectorize_prompt(prompt)
        self._check_vector_dims(vector)

        entry_id = self._make_entry_id(prompt, filters)

        cache_entry = CacheEntry(
            entry_id=entry_id,
            prompt=prompt,
            response=response,
            prompt_vector=vector,
            metadata=metadata,
            filters=filters,
        )

        ttl = ttl if ttl is not None else self._ttl
        keys = await aindex.load(
            data=[cache_entry.to_dict(self._vectorizer.dtype)],
            ttl=ttl,
            id_field=ENTRY_ID_FIELD_NAME,
        )
        return keys[0]

    def update(self, key: str, **kwargs) -> None:
        """更新缓存条目

        Args:
            key: Redis键
            **kwargs: 要更新的字段（metadata, response等）

        示例:
            cache.update(key, metadata={"hit_count": 5}, response="new response")
        """
        if kwargs:
            for k, v in kwargs.items():
                if k not in set(self._index.schema.field_names + [METADATA_FIELD_NAME]):
                    raise ValueError(f"{k}不是有效的缓存字段")

                if k == METADATA_FIELD_NAME:
                    if isinstance(v, dict):
                        kwargs[k] = serialize(v)
                    else:
                        raise TypeError("metadata必须是字典")

            kwargs[UPDATED_AT_FIELD_NAME] = current_timestamp()

            client = self._get_redis_client()
            client.hset(key, mapping=kwargs)

        self.expire(key)

    async def aupdate(self, key: str, **kwargs) -> None:
        """异步更新缓存条目"""
        if kwargs:
            for k, v in kwargs.items():
                if k not in set(self._index.schema.field_names + [METADATA_FIELD_NAME]):
                    raise ValueError(f"{k}不是有效的缓存字段")

                if k == METADATA_FIELD_NAME:
                    if isinstance(v, dict):
                        kwargs[k] = serialize(v)
                    else:
                        raise TypeError("metadata必须是字典")

            kwargs[UPDATED_AT_FIELD_NAME] = current_timestamp()

            client = await self._get_async_redis_client()
            await client.hset(key, mapping=kwargs)

        await self.aexpire(key)

    def _make_entry_id(self, prompt: str, filters: dict[str, Any] | None = None) -> str:
        """生成确定性entry_id"""
        return hashify(prompt, filters)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.adisconnect()

    def disconnect(self):
        """断开连接"""
        if hasattr(self, "_index") and self._index:
            self._index.disconnect()
        if hasattr(self, "_aindex") and self._aindex:
            self._aindex.disconnect_sync()
        super().disconnect()

    async def adisconnect(self):
        """异步断开连接"""
        if hasattr(self, "_aindex") and self._aindex:
            await self._aindex.disconnect()
            self._aindex = None
        await super().adisconnect()


# ==================== 使用示例 ====================
"""
# 示例1: 基础用法
cache = SemanticCache(name="qa_cache", distance_threshold=0.1, ttl=3600)
cache.store("What is Python?", "Python is a programming language.")
hits = cache.check("What is Python?")
if hits:
    print(hits[0]["response"])  # Python is a programming language.

# 示例2: 自定义Redis客户端和Embedding模型
from redis import Redis
from redisvl.utils.vectorize import OpenAIVectorizer

client = Redis(host='localhost', port=6379, db=0)
vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

cache = SemanticCache(
    name="qa_cache",
    redis_client=client,
    vectorizer=vectorizer,
    distance_threshold=0.15,  # 更宽松的阈值
    ttl=7200  # 2小时过期
)

# 示例3: 过滤字段用法
cache = SemanticCache(
    name="qa_cache",
    filterable_fields=[
        {"name": "category", "type": "tag"},
        {"name": "version", "type": "numeric"}
    ]
)

# 存储时带过滤条件
cache.store(
    "What is Python?",
    "Python is...",
    filters={"category": "programming", "version": 1}
)

# 查询时过滤
hits = cache.check(
    "Python question",
    filter_expression=Tag("category") == "programming"
)

# 示例4: 调整阈值
cache.set_threshold(0.2)  # 更宽松
hits = cache.check("similar query")

# 示例5: 异步操作
import asyncio

async def demo():
    cache = SemanticCache(name="async_cache")
    await cache.astore("What is AI?", "AI is artificial intelligence.")
    hits = await cache.acheck("Tell me about AI")
    print(hits)

asyncio.run(demo())

# 示例6: 更新缓存条目
key = cache.store("original prompt", "original response")
cache.update(key, metadata={"hit_count": 1}, response="updated response")

# 示例7: 多结果查询
hits = cache.check("general query", num_results=3)
for hit in hits:
    print(f"Distance: {hit['vector_distance']}, Response: {hit['response']}")

# 示例8: 上下文管理
with SemanticCache(name="temp_cache") as cache:
    cache.store("prompt", "response")
    # 自动断开连接
"""