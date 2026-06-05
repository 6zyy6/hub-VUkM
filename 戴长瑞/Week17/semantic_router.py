"""
SemanticRouter - 语义路由，根据输入文本自动路由到预设的处理函数/目标

功能：
- 基于向量相似度匹配路由
- 支持动态注册路由
- 支持批量匹配多个路由
- 支持自定义Redis客户端和Embedding模型

依赖：redisvl >= 0.2.0, redis, pydantic

使用示例：
    # 基础用法
    routes = [
        Route(name="tech", references=["Python", "JavaScript", "coding"]),
        Route(name="finance", references=["stock", "market", "investment"]),
    ]
    router = SemanticRouter(name="my_router", routes=routes)
    result = router("What is Python?")
    print(result.name, result.distance)

    # 自定义Redis客户端和Embedding模型
    from redis import Redis
    from redisvl.utils.vectorize import OpenAIVectorizer

    client = Redis(host='localhost', port=6379, db=0)
    vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

    router = SemanticRouter(
        name="intent_router",
        redis_client=client,
        vectorizer=vectorizer,
        routes=[],
        routing_config=RoutingConfig(max_k=3)
    )

    # 动态注册路由
    router.add_route(Route(name="weather", references=["weather", "rain", "sunny"]))
    router.add_route_references("tech", ["Go", "Rust", "C++"])

    # 批量匹配
    results = router.route_many("Python or Rust?", max_k=2)
"""

from typing import Any

from pydantic import BaseModel, Field

from redisvl.extensions.constants import ROUTE_VECTOR_FIELD_NAME
from redisvl.extensions.router.schema import (
    DistanceAggregationMethod,
    Route,
    RouteMatch,
    RoutingConfig,
)
from redisvl.index import SearchIndex
from redisvl.redis.utils import convert_bytes, hashify, make_dict
from redisvl.utils.log import get_logger
from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

logger = get_logger("[SemanticRouter]")


class RouterIndexSchema:
    """语义路由索引schema"""

    @classmethod
    def from_params(cls, name: str, vector_dims: int, dtype: str):
        from redisvl.schema import IndexSchema
        return IndexSchema(
            index={"name": name, "prefix": name},
            fields=[
                {"name": "reference_id", "type": "tag"},
                {"name": "route_name", "type": "tag"},
                {"name": "reference", "type": "text"},
                {
                    "name": ROUTE_VECTOR_FIELD_NAME,
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


class SemanticRouter(BaseModel):
    """语义路由器 - 将输入文本路由到预设的处理函数/目标

    支持：
    - 单路由匹配（__call__）
    - 多路由批量匹配（route_many）
    - 动态注册路由（add_route, add_route_references）
    - 自定义向量化器和Redis客户端
    """

    name: str
    routes: list[Route] = Field(default_factory=list)
    vectorizer: BaseVectorizer = Field(default_factory=HFTextVectorizer)
    routing_config: RoutingConfig = Field(default_factory=RoutingConfig)

    _index: SearchIndex | None = Field(default=None, repr=False)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        routes: list[Route] | None = None,
        vectorizer: BaseVectorizer | None = None,
        routing_config: RoutingConfig | None = None,
        redis_client=None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        connection_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        """初始化语义路由

        Args:
            name: 路由器名称
            routes: 初始路由列表
            vectorizer: 向量化器，默认HFTextVectorizer
                     支持：OpenAIVectorizer, CohereVectorizer等
            routing_config: 路由配置，默认RoutingConfig()
            redis_client: Redis客户端实例（同步）
            redis_url: Redis连接URL
            overwrite: 是否覆盖已有索引
            connection_kwargs: 连接参数
        """
        dtype = kwargs.pop("dtype", None)

        if routes is None:
            routes = []

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

        if routing_config is None:
            routing_config = RoutingConfig()

        super().__init__(
            name=name,
            routes=routes or [],
            vectorizer=self._vectorizer,
            routing_config=routing_config,
        )

        self._initialize_index(
            redis_client,
            redis_url,
            overwrite,
            connection_kwargs=connection_kwargs or None,
        )

        self._index.client.json().set(f"{self.name}:route_config", ".", self.to_dict())

    def _initialize_index(
        self,
        redis_client=None,
        redis_url: str = "redis://localhost:6379",
        overwrite: bool = False,
        connection_kwargs: dict[str, Any] | None = None,
    ):
        """初始化搜索索引"""
        schema = RouterIndexSchema.from_params(
            self.name, self._vectorizer.dims, self._vectorizer.dtype
        )

        self._index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs,
        )

        existed = self._index.exists()
        if not overwrite and existed:
            existing = SearchIndex.from_existing(self.name, redis_client=self._index.client)
            if existing.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(f"索引{self.name}已存在且schema不匹配，设置overwrite=True可覆盖")

        self._index.create(overwrite=overwrite, drop=False)

        if not existed or overwrite:
            self._add_routes(self.routes)

    def __repr__(self) -> str:
        return f"SemanticRouter(name={self.name!r}, routes={len(self.routes)})"

    @property
    def vectorizer(self) -> BaseVectorizer:
        """获取向量化器"""
        return self._vectorizer

    @property
    def route_names(self) -> list[str]:
        """获取所有路由名称"""
        return [route.name for route in self.routes]

    @property
    def route_thresholds(self) -> dict[str, float | None]:
        """获取所有路由的阈值"""
        return {route.name: route.distance_threshold for route in self.routes}

    def _route_ref_key(self, route_name: str, reference_hash: str) -> str:
        """生成路由引用键"""
        sep = self._index.key_separator
        prefix = self._index.prefix.rstrip(sep) if sep and self._index.prefix else self._index.prefix
        if prefix:
            return f"{prefix}{sep}{route_name}{sep}{reference_hash}"
        return f"{route_name}{sep}{reference_hash}"

    def _route_pattern(self, route_name: str) -> str:
        """生成路由搜索模式"""
        sep = self._index.key_separator
        prefix = self._index.prefix.rstrip(sep) if sep and self._index.prefix else self._index.prefix
        if prefix:
            return f"{prefix}{sep}{route_name}{sep}*"
        return f"{route_name}{sep}*"

    def _add_routes(self, routes: list[Route]) -> None:
        """添加路由到索引"""
        route_references = []
        keys = []

        for route in routes:
            if not route.references:
                continue

            reference_vectors = self._vectorizer.embed_many(
                [ref for ref in route.references],
                as_buffer=True,
            )

            for i, reference in enumerate(route.references):
                ref_hash = hashify(reference)
                route_references.append({
                    "reference_id": ref_hash,
                    "route_name": route.name,
                    "reference": reference,
                    "vector": reference_vectors[i],
                })
                keys.append(self._route_ref_key(route.name, ref_hash))

            if not self.get(route.name):
                self.routes.append(route)

        if route_references:
            self._index.load(route_references, keys=keys)

    def get(self, route_name: str) -> Route | None:
        """获取路由

        Args:
            route_name: 路由名称

        Returns:
            Route对象或None
        """
        return next((r for r in self.routes if r.name == route_name), None)

    def _process_route_result(self, result: dict[str, Any]) -> RouteMatch:
        """处理路由结果"""
        route_dict = make_dict(convert_bytes(result))
        return RouteMatch(
            name=route_dict["route_name"],
            distance=float(route_dict["distance"]),
        )

    def _distance_threshold_filter(self) -> str:
        """生成分距离阈值的过滤表达式"""
        if not self.routes:
            return ""

        filter_parts = []
        for route in self.routes:
            threshold = route.distance_threshold
            if threshold is not None:
                filter_parts.append(f"(@route_name == '{route.name}' && @distance < {threshold})")
            else:
                filter_parts.append(f"@route_name == '{route.name}'")

        return " || ".join(filter_parts)

    def _get_route_matches(
        self,
        vector: list[float],
        aggregation_method: DistanceAggregationMethod,
        max_k: int = 1,
    ) -> list[RouteMatch]:
        """获取匹配的路由"""
        if not self.routes:
            return []

        # 使用最大阈值作为初始过滤
        max_threshold = max(
            (r.distance_threshold for r in self.routes if r.distance_threshold is not None),
            default=2.0
        )

        from redisvl.query import VectorRangeQuery, FilterQuery
        from redisvl.query.filter import Tag

        # 先用VectorRangeQuery找到候选
        range_query = VectorRangeQuery(
            vector=vector,
            vector_field_name=ROUTE_VECTOR_FIELD_NAME,
            distance_threshold=float(max_threshold),
            return_fields=["route_name", "distance"],
            num_results=max_k * len(self.routes),
            return_score=True,
            dtype=self._vectorizer.dtype,
        )

        # 获取所有候选路由
        candidates = self._index.query(range_query)

        # 应用各路由的独立阈值
        route_matches = []
        for candidate in candidates:
            route_name = candidate.get("route_name")
            distance = float(candidate.get("distance", float('inf')))

            route = self.get(route_name)
            if route and route.distance_threshold is not None:
                if distance < route.distance_threshold:
                    route_matches.append(RouteMatch(name=route_name, distance=distance))
            elif route:
                route_matches.append(RouteMatch(name=route_name, distance=distance))

        # 按距离排序
        route_matches.sort(key=lambda x: x.distance)

        return route_matches[:max_k]

    def _classify_route(
        self,
        vector: list[float],
        aggregation_method: DistanceAggregationMethod,
    ) -> RouteMatch:
        """分类到单个路由"""
        route_matches = self._get_route_matches(vector, aggregation_method)

        if not route_matches:
            return RouteMatch()

        top = route_matches[0]
        if top.name is not None:
            return top

        raise ValueError(f"{top.name}不是支持的路由")

    def __call__(
        self,
        statement: str | None = None,
        vector: list[float] | None = None,
        aggregation_method: DistanceAggregationMethod | None = None,
        distance_threshold: float | None = None,
    ) -> RouteMatch:
        """路由输入文本/向量

        Args:
            statement: 输入文本
            vector: 输入向量
            aggregation_method: 聚合方法（min/sum/avg）
            distance_threshold: 全局阈值覆盖

        Returns:
            RouteMatch，包含匹配的路由名称和距离

        示例:
            result = router("What is Python?")
            if result.name:
                print(f"路由到: {result.name}, 距离: {result.distance}")
        """
        if not vector:
            if not statement:
                raise ValueError("必须提供statement或vector")
            vector = self._vectorizer.embed(statement)

        agg = aggregation_method or self.routing_config.aggregation_method

        top_match = self._classify_route(vector, agg)

        if distance_threshold and top_match.distance > distance_threshold:
            return RouteMatch()

        return top_match

    def route_many(
        self,
        statement: str | None = None,
        vector: list[float] | None = None,
        max_k: int | None = None,
        aggregation_method: DistanceAggregationMethod | None = None,
        distance_threshold: float | None = None,
    ) -> list[RouteMatch]:
        """路由到多个匹配

        Args:
            statement: 输入文本
            vector: 输入向量
            max_k: 返回的最多路由数，默认使用routing_config.max_k
            aggregation_method: 聚合方法
            distance_threshold: 全局阈值覆盖

        Returns:
            RouteMatch列表，按距离排序

        示例:
            results = router.route_many("Python or JavaScript?", max_k=3)
            for r in results:
                print(f"{r.name}: {r.distance}")
        """
        if not vector:
            if not statement:
                raise ValueError("必须提供statement或vector")
            vector = self._vectorizer.embed(statement)

        max_k = max_k or self.routing_config.max_k
        agg = aggregation_method or self.routing_config.aggregation_method

        route_matches = self._get_route_matches(vector, agg, max_k)

        if distance_threshold is not None:
            route_matches = [r for r in route_matches if r.distance <= distance_threshold]

        return route_matches

    def add_route(self, route: Route) -> None:
        """动态添加新路由

        Args:
            route: Route对象

        示例:
            new_route = Route(
                name="sports",
                references=["football", "basketball", "soccer"],
                distance_threshold=0.5
            )
            router.add_route(new_route)
        """
        if self.get(route.name):
            logger.warning(f"路由{route.name}已存在，将更新")
            self.remove_route(route.name)

        self._add_routes([route])
        self._update_router_state()

    def add_routes(self, routes: list[Route]) -> None:
        """批量添加路由

        Args:
            routes: Route对象列表

        示例:
            router.add_routes([
                Route(name="tech", references=["Python", "Java"]),
                Route(name="sports", references=["football", "basketball"]),
            ])
        """
        for route in routes:
            if self.get(route.name):
                logger.warning(f"路由{route.name}已存在，跳过")
                continue
            self._add_routes([route])

        self._update_router_state()

    def add_route_references(self, route_name: str, references: str | list[str]) -> list[str]:
        """为已有路由添加引用

        Args:
            route_name: 路由名称
            references: 引用文本或列表

        Returns:
            添加的引用对应的Redis键

        示例:
            router.add_route_references("tech", ["Go", "Rust", "C++"])
            router.add_route_references("tech", "TypeScript")
        """
        if isinstance(references, str):
            references = [references]

        route = self.get(route_name)
        if not route:
            raise ValueError(f"路由{route_name}不存在")

        ref_vectors = self._vectorizer.embed_many(references, as_buffer=True)
        route_refs = []
        keys = []

        for i, ref in enumerate(references):
            ref_hash = hashify(ref)
            route_refs.append({
                "reference_id": ref_hash,
                "route_name": route_name,
                "reference": ref,
                "vector": ref_vectors[i],
            })
            keys.append(self._route_ref_key(route_name, ref_hash))

        keys = self._index.load(route_refs, keys=keys)
        route.references.extend(references)
        self._update_router_state()
        return keys

    def update_route_threshold(self, route_name: str, distance_threshold: float | None) -> None:
        """更新路由的阈值

        Args:
            route_name: 路由名称
            distance_threshold: 新阈值，None表示无限制

        示例:
            router.update_route_threshold("tech", 0.3)
        """
        route = self.get(route_name)
        if not route:
            raise ValueError(f"路由{route_name}不存在")

        route.distance_threshold = distance_threshold
        self._update_router_state()

    def get_route_references(
        self,
        route_name: str = "",
        reference_ids: list[str] = [],
        keys: list[str] = [],
    ) -> list[dict[str, Any]]:
        """获取路由引用

        Args:
            route_name: 路由名称
            reference_ids: 引用ID列表
            keys: Redis键列表

        Returns:
            引用详情列表
        """
        from redisvl.utils.utils import scan_by_pattern

        if reference_ids:
            queries = self._make_filter_queries(reference_ids)
        elif route_name:
            if not keys:
                pattern = self._route_pattern(route_name)
                keys = scan_by_pattern(self._index.client, pattern)

            if keys:
                sep = self._index.key_separator
                queries = self._make_filter_queries([key.split(sep)[-1] for key in convert_bytes(keys)])
            else:
                queries = []
        else:
            raise ValueError("必须提供route_name, reference_ids或keys")

        if not queries:
            return []

        res = self._index.batch_query(queries)
        return [r[0] for r in res if len(r) > 0]

    def _make_filter_queries(self, ids: list[str]) -> list:
        """创建过滤查询"""
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag

        queries = []
        for id in ids:
            fe = Tag("reference_id") == id
            fq = FilterQuery(
                return_fields=["reference_id", "route_name", "reference"],
                filter_expression=fe,
            )
            queries.append(fq)
        return queries

    def delete_route_references(
        self,
        route_name: str = "",
        reference_ids: list[str] = [],
        keys: list[str] = [],
    ) -> int:
        """删除路由引用

        Args:
            route_name: 路由名称
            reference_ids: 引用ID列表
            keys: Redis键列表

        Returns:
            删除的引用数量
        """
        from redisvl.utils.utils import scan_by_pattern

        if reference_ids and not keys:
            queries = self._make_filter_queries(reference_ids)
            res = self._index.batch_query(queries)
            keys = [r[0]["id"] for r in res if len(r) > 0]
        elif not keys:
            pattern = self._route_pattern(route_name)
            keys = scan_by_pattern(self._index.client, pattern)

        if not keys:
            return 0

        deleted = self._index.drop_keys(keys)

        for key in keys:
            route_name_from_key = key.split(":")[-2]
            route = self.get(route_name_from_key)
            if route:
                ref_key = key.split(":")[-1]
                route.references = [r for r in route.references if hashify(r) != ref_key]

        self._update_router_state()
        return deleted

    def remove_route(self, route_name: str) -> None:
        """移除路由及其所有引用

        Args:
            route_name: 路由名称
        """
        route = self.get(route_name)
        if route is None:
            logger.warning(f"路由{route_name}不存在")
            return

        # 删除所有引用
        self._index.drop_keys([
            self._route_ref_key(route.name, hashify(ref))
            for ref in route.references
        ])

        # 从列表中移除
        self.routes = [r for r in self.routes if r.name != route_name]
        self._update_router_state()

    def delete(self) -> None:
        """删除整个路由器"""
        self._index.delete(drop=True)

    def clear(self) -> None:
        """清空所有路由"""
        self._index.clear()
        self.routes = []
        self._update_router_state()

    def _update_router_state(self) -> None:
        """更新路由配置到Redis"""
        self._index.client.json().set(f"{self.name}:route_config", ".", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        from redisvl.utils.utils import model_to_dict
        return {
            "name": self.name,
            "routes": [model_to_dict(route) for route in self.routes],
            "vectorizer": {
                "type": self.vectorizer.type,
                "model": self.vectorizer.model,
            },
            "routing_config": model_to_dict(self.routing_config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs) -> "SemanticRouter":
        """从字典创建

        Args:
            data: 路由器配置字典
            **kwargs: 传递给构造器的额外参数

        返回:
            SemanticRouter实例

        示例:
            config = {
                "name": "my_router",
                "routes": [{"name": "tech", "references": ["Python"]}],
                "vectorizer": {"type": "huggingface", "model": "sentence-transformers/all-mpnet-base-v2"},
                "routing_config": {"max_k": 3}
            }
            router = SemanticRouter.from_dict(config, redis_url="redis://localhost:6379")
        """
        from redisvl.utils.vectorize import vectorizer_from_dict

        try:
            name = data["name"]
            routes_data = data.get("routes", [])
            vectorizer_data = data.get("vectorizer", {})
            routing_config_data = data.get("routing_config", {})
        except KeyError as e:
            raise ValueError(f"缺少必需字段: {str(e)}")

        vectorizer = vectorizer_from_dict(vectorizer_data) if vectorizer_data else None
        if not vectorizer:
            raise ValueError(f"无法加载vectorizer: {vectorizer_data}")

        routes = [Route(**r) for r in routes_data]
        routing_config = RoutingConfig(**routing_config_data)

        return cls(
            name=name,
            routes=routes,
            vectorizer=vectorizer,
            routing_config=routing_config,
            **kwargs,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def disconnect(self):
        """断开连接"""
        if self._index:
            self._index.disconnect()


# ==================== 使用示例 ====================
"""
# 示例1: 基础用法
routes = [
    Route(name="tech", references=["Python", "JavaScript", "coding"]),
    Route(name="finance", references=["stock", "market", "investment"]),
    Route(name="sports", references=["football", "basketball", "soccer"]),
]

router = SemanticRouter(name="my_router", routes=routes)

# 单路由匹配
result = router("What is Python?")
print(f"路由到: {result.name}, 距离: {result.distance}")

# 示例2: 自定义Redis客户端和Embedding模型
from redis import Redis
from redisvl.utils.vectorize import OpenAIVectorizer

client = Redis(host='localhost', port=6379, db=0)
vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

router = SemanticRouter(
    name="intent_router",
    redis_client=client,
    vectorizer=vectorizer,
    routes=[],
    routing_config=RoutingConfig(max_k=3, aggregation_method=DistanceAggregationMethod.min)
)

# 示例3: 动态注册路由
router.add_route(Route(
    name="weather",
    references=["weather", "rain", "sunny", "temperature"],
    distance_threshold=0.5
))

# 为已有路由添加引用
router.add_route_references("tech", ["Go", "Rust", "C++", "TypeScript"])
router.add_route_references("tech", "JavaScript")

# 示例4: 批量匹配多个路由
results = router.route_many("I want to learn Python and JavaScript", max_k=3)
for r in results:
    print(f"{r.name}: distance={r.distance}")

# 示例5: 更新路由阈值
router.update_route_threshold("tech", 0.3)
router.update_route_threshold("finance", 0.4)

# 示例6: 获取路由引用
refs = router.get_route_references(route_name="tech")
for ref in refs:
    print(f"Reference: {ref['reference']}")

# 示例7: 删除路由引用
deleted = router.delete_route_references(route_name="tech", reference_ids=["hash1", "hash2"])
print(f"删除了 {deleted} 个引用")

# 示例8: 移除路由
router.remove_route("sports")

# 示例9: 从字典加载
config = {
    "name": "saved_router",
    "routes": [
        {"name": "greeting", "references": ["hello", "hi", "hey"], "distance_threshold": 0.4},
    ],
    "vectorizer": {"type": "huggingface", "model": "sentence-transformers/all-mpnet-base-v2"},
    "routing_config": {"max_k": 5}
}

router2 = SemanticRouter.from_dict(config, redis_url="redis://localhost:6379")

# 示例10: 上下文管理
with SemanticRouter(name="temp_router", routes=[]) as router:
    router.add_route(Route(name="temp", references=["temporary"]))
    result = router("temporary query")
    # 自动断开连接

# 示例11: 无匹配处理
result = router("完全不相关的查询")
if not result.name:
    print("没有匹配的路由")

# 示例12: 批量添加路由
router.add_routes([
    Route(name="music", references=["song", "music", "play"]),
    Route(name="movie", references=["film", "movie", "cinema"]),
])
"""