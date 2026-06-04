r"""
SemanticRouter.py

一个可复用的语义路由底层组件。

核心能力：
1. 基于 embedding 语义相似度，把用户问题路由到指定 target。
2. 支持与 EmbeddingsCache.py 中的 CachedEmbeddingService / SentenceBertEmbedder 组合使用。
3. 支持注入任意 embedding_service、embedder 或 embed_fn，方便接入企业内部模型服务。
4. 支持多路由、多示例问题、独立阈值、top-k 候选、批量路由和可解释结果。
5. 支持路由配置保存与加载，方便在 RAG、Agent、客服机器人、工具调用等场景复用。
6. 内置轻量 HashingTextEmbedder，仅用于本地离线测试；生产环境建议接入真实 embedding 模型。

依赖安装：
    pip install numpy

可选依赖：
    如果要使用真实语义向量，建议配合 EmbeddingsCache.py：
    pip install redis sentence-transformers

典型用法：
    # 生产环境：推荐传入 CachedEmbeddingService
    router = SemanticRouter(embedding_service=embedding_service)

    router.add_route(
        questions=["如何退货", "我要申请退款", "商品不满意可以退吗"],
        target="refund",
        threshold=0.72,
    )

    decision = router.route("我买错了，想退掉")
    print(decision.target, decision.score)
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


TextInput = Union[str, Sequence[str]]
EmbeddingArray = np.ndarray
EmbedFn = Callable[..., EmbeddingArray]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteCandidate:
    """单个候选路由的匹配结果。"""

    target: str
    route_name: str
    score: float
    threshold: float
    matched_question: str
    accepted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def margin_to_threshold(self) -> float:
        """候选分数距离阈值的差值。"""
        return self.score - self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """转为普通字典，便于日志记录或 API 返回。"""
        return asdict(self)


@dataclass(frozen=True)
class RouteDecision:
    """一次语义路由的最终决策。"""

    question: str
    target: Optional[str]
    route_name: Optional[str]
    score: float
    threshold: float
    matched_question: Optional[str]
    accepted: bool
    candidates: List[RouteCandidate]
    latency_ms: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """转为普通字典，便于日志记录或 API 返回。"""
        return {
            "question": self.question,
            "target": self.target,
            "route_name": self.route_name,
            "score": self.score,
            "threshold": self.threshold,
            "matched_question": self.matched_question,
            "accepted": self.accepted,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
            "candidates": [item.to_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class RouterStats:
    """语义路由器运行期统计信息。"""

    total: int
    accepted: int
    rejected: int
    total_latency_ms: float

    @property
    def accept_rate(self) -> float:
        """路由接受率。"""
        if self.total == 0:
            return 0.0
        return self.accepted / self.total

    @property
    def avg_latency_ms(self) -> float:
        """平均路由耗时。"""
        if self.total == 0:
            return 0.0
        return self.total_latency_ms / self.total


@dataclass
class _RouteRecord:
    """内部路由记录。"""

    route_name: str
    target: str
    questions: List[str]
    embeddings: EmbeddingArray
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))

    def to_config(self) -> Dict[str, Any]:
        """导出可持久化的路由配置，不直接导出 embedding。"""
        return {
            "route_name": self.route_name,
            "target": self.target,
            "questions": list(self.questions),
            "threshold": self.threshold,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class HashingTextEmbedder:
    """
    轻量级文本 hashing embedder。

    说明：
    - 这个类不是真正的语义模型，只用于本地开发、单元测试或无模型环境下的 smoke test。
    - 生产环境建议使用 Sentence-BERT / BGE / 企业内部 embedding 服务。
    - 接口设计与 SentenceBertEmbedder 类似，方便被 SemanticRouter 直接调用。
    """

    def __init__(
        self,
        dim: int = 384,
        lowercase: bool = True,
        normalize_embeddings: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim 必须大于 0")

        self.dim = int(dim)
        self.lowercase = lowercase
        self.normalize_embeddings = normalize_embeddings

    @staticmethod
    def _tokens(text: str, lowercase: bool = True) -> List[str]:
        """把中英文文本切成适合 hashing 的 token。"""
        if lowercase:
            text = text.lower()

        # 英文/数字连续片段 + 单个 CJK 字符。
        base_tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
        if not base_tokens:
            return [text.strip()] if text.strip() else []

        # 加入 char bigram，提升短句匹配稳定性。
        compact = "".join(base_tokens)
        char_bigrams = [compact[idx : idx + 2] for idx in range(max(0, len(compact) - 1))]
        return base_tokens + char_bigrams

    def _encode_one(self, text: str) -> EmbeddingArray:
        """生成单条文本的 hashing 向量。"""
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokens(text, lowercase=self.lowercase)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big", signed=False) % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign

        if self.normalize_embeddings:
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm

        return vec.astype(np.float32)

    def encode(self, text: TextInput, is_query: bool = False) -> EmbeddingArray:
        """生成 embedding。单条输入返回一维数组，批量输入返回二维数组。"""
        texts, single_input = SemanticRouter._ensure_text_list(text)
        if not texts:
            raise ValueError("text 不能为空")

        embeddings = np.vstack([self._encode_one(item) for item in texts]).astype(np.float32)
        return embeddings[0] if single_input else embeddings


class SemanticRouter:
    """
    企业级 AI 应用通用语义路由组件。

    这个类只负责“基于语义相似度做路由决策”，不绑定具体 embedding 模型。
    你可以按需注入：
    - embedding_service：推荐传入 EmbeddingsCache.py 中的 CachedEmbeddingService；
    - embedder：例如 SentenceBertEmbedder，要求提供 encode(text, is_query=...)；
    - embed_fn：企业内部 embedding RPC / HTTP 客户端函数；
    - 都不传时，使用 HashingTextEmbedder 作为本地测试兜底。

    Parameters
    ----------
    name:
        路由器名称，便于日志与配置管理。
    embedding_service:
        可选的高层 embedding 服务，优先调用其 embed 方法。
    embedder:
        可选的 embedding 模型对象，优先调用其 encode 方法，其次调用 embed 方法。
    embed_fn:
        可选的 embedding 函数，建议签名为 embed_fn(texts, is_query=False)。
    default_threshold:
        默认接受阈值。候选路由分数低于该值时会被拒绝。
    min_score_margin:
        第一名与第二名之间的最小分差。用于避免两个路由过于接近导致误判。
    aggregation:
        多示例问题的路由聚合方式：max / mean / centroid。
    top_k:
        默认返回候选数量。
    fallback_target:
        拒绝路由时返回的兜底 target。None 表示不返回业务 target。
    normalize_embeddings:
        是否在路由层再次归一化 embedding。推荐 True。
    use_cache:
        调用 embedding_service 时是否启用缓存。
    raise_on_error:
        True 时异常直接抛出；False 时记录日志并返回拒绝路由结果。
    """

    SUPPORTED_AGGREGATIONS = {"max", "mean", "centroid"}

    def __init__(
        self,
        name: str = "semantic_router",
        embedding_service: Optional[object] = None,
        embedder: Optional[object] = None,
        embed_fn: Optional[EmbedFn] = None,
        default_threshold: float = 0.65,
        min_score_margin: float = 0.0,
        aggregation: str = "max",
        top_k: int = 3,
        fallback_target: Optional[str] = None,
        normalize_embeddings: bool = True,
        use_cache: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("name 不能为空")
        if not 0 <= default_threshold <= 1:
            raise ValueError("default_threshold 必须在 [0, 1] 范围内")
        if min_score_margin < 0:
            raise ValueError("min_score_margin 不能小于 0")
        if aggregation not in self.SUPPORTED_AGGREGATIONS:
            raise ValueError(
                "aggregation 只支持："
                f"{sorted(self.SUPPORTED_AGGREGATIONS)}"
            )
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        self.name = name.strip()
        self.embedding_service = embedding_service
        self.embedder = embedder
        self.embed_fn = embed_fn
        self.default_threshold = float(default_threshold)
        self.min_score_margin = float(min_score_margin)
        self.aggregation = aggregation
        self.top_k = int(top_k)
        self.fallback_target = fallback_target
        self.normalize_embeddings = normalize_embeddings
        self.use_cache = use_cache
        self.raise_on_error = raise_on_error

        # 没有注入真实模型时，使用轻量测试 embedder，避免组件无法本地运行。
        if self.embedding_service is None and self.embedder is None and self.embed_fn is None:
            self.embedder = HashingTextEmbedder(normalize_embeddings=normalize_embeddings)

        self._routes: Dict[str, _RouteRecord] = {}
        self._lock = RLock()

        self._total = 0
        self._accepted = 0
        self._rejected = 0
        self._total_latency_ms = 0.0

    def _handle_error(self, message: str, exc: Exception, default):
        """统一处理异常。"""
        logger.exception("%s: %s", message, exc)
        if self.raise_on_error:
            raise exc
        return default

    @staticmethod
    def _ensure_text_list(text: TextInput) -> Tuple[List[str], bool]:
        """
        统一把输入转成 List[str]。

        Returns
        -------
        texts:
            文本列表。
        single_input:
            原始输入是否为单条字符串。
        """
        single_input = isinstance(text, str)
        texts = [text] if single_input else list(text)

        for item in texts:
            if not isinstance(item, str):
                raise TypeError("text 中的每一项都必须是字符串")

        return texts, single_input

    @staticmethod
    def _safe_route_name(value: str, max_len: int = 128) -> str:
        """生成安全的路由名称。"""
        safe = []
        for ch in value:
            if ch.isalnum() or ch in {"_", "-", ".", ":"}:
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)[:max_len] or "default"

    @staticmethod
    def _validate_threshold(threshold: Optional[float], field_name: str = "threshold") -> Optional[float]:
        """校验阈值。"""
        if threshold is None:
            return None
        if not 0 <= threshold <= 1:
            raise ValueError(f"{field_name} 必须在 [0, 1] 范围内")
        return float(threshold)

    @staticmethod
    def _call_with_supported_kwargs(func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        只传入被函数签名支持的关键字参数。

        这样可以兼容：
        - embed(texts)
        - encode(texts, is_query=False)
        - embed(texts, is_query=False, use_cache=True)
        """
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            return func(*args, **kwargs)

        parameters = signature.parameters
        has_var_keyword = any(
            item.kind == inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        )
        if has_var_keyword:
            return func(*args, **kwargs)

        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters
        }
        return func(*args, **supported_kwargs)

    def _embed(
        self,
        text: TextInput,
        is_query: bool = False,
        use_cache: Optional[bool] = None,
    ) -> EmbeddingArray:
        """调用外部 embedding 能力，并把输出统一成 numpy 数组。"""
        texts, single_input = self._ensure_text_list(text)
        if not texts:
            raise ValueError("text 不能为空")

        actual_use_cache = self.use_cache if use_cache is None else use_cache

        try:
            if self.embedding_service is not None:
                if not hasattr(self.embedding_service, "embed"):
                    raise TypeError("embedding_service 必须提供 embed 方法")
                embeddings = self._call_with_supported_kwargs(
                    self.embedding_service.embed,
                    texts,
                    is_query=is_query,
                    use_cache=actual_use_cache,
                    return_cache_stats=False,
                )
            elif self.embedder is not None:
                if hasattr(self.embedder, "encode"):
                    embeddings = self._call_with_supported_kwargs(
                        self.embedder.encode,
                        texts,
                        is_query=is_query,
                    )
                elif hasattr(self.embedder, "embed"):
                    embeddings = self._call_with_supported_kwargs(
                        self.embedder.embed,
                        texts,
                        is_query=is_query,
                    )
                else:
                    raise TypeError("embedder 必须提供 encode 或 embed 方法")
            elif self.embed_fn is not None:
                embeddings = self._call_with_supported_kwargs(
                    self.embed_fn,
                    texts,
                    is_query=is_query,
                )
            else:  # pragma: no cover - __init__ 已经兜底
                raise RuntimeError("未配置 embedding 生成方式")

            # 兼容部分服务返回 (embeddings, stats) 的情况。
            if isinstance(embeddings, tuple) and embeddings:
                embeddings = embeddings[0]

            arr = self._normalize_embeddings(len(texts), embeddings)
            return arr[0] if single_input else arr
        except Exception as exc:
            return self._handle_error("Generate embedding failed", exc, default=np.empty((0, 0), dtype=np.float32))

    def _normalize_embeddings(
        self,
        text_count: int,
        embedding: EmbeddingArray,
    ) -> EmbeddingArray:
        """把 embedding 统一成二维 float32 数组，并按需归一化。"""
        arr = np.asarray(embedding, dtype=np.float32)

        if text_count == 1 and arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError("embedding 必须是一维或二维数组")
        if arr.shape[0] != text_count:
            raise ValueError(
                "text 数量和 embedding 数量不一致："
                f"text={text_count}, embedding={arr.shape[0]}"
            )
        if arr.shape[1] <= 0:
            raise ValueError("embedding 维度必须大于 0")

        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if self.normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms

        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _cosine_similarity(query_embedding: EmbeddingArray, route_embeddings: EmbeddingArray) -> EmbeddingArray:
        """计算 query 与一组 route example embedding 的余弦相似度。"""
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        routes = np.asarray(route_embeddings, dtype=np.float32)

        if routes.ndim != 2:
            raise ValueError("route_embeddings 必须是二维数组")
        if query.shape[0] != routes.shape[1]:
            raise ValueError(
                "query embedding 维度与 route embedding 维度不一致："
                f"query={query.shape[0]}, route={routes.shape[1]}"
            )

        query_norm = float(np.linalg.norm(query))
        route_norms = np.linalg.norm(routes, axis=1)
        denom = np.where(route_norms == 0, 1.0, route_norms) * (query_norm if query_norm > 0 else 1.0)
        return np.dot(routes, query) / denom

    def _route_score(
        self,
        query_embedding: EmbeddingArray,
        record: _RouteRecord,
    ) -> Tuple[float, str]:
        """计算某一路由的最终分数，并返回最相似的示例问题。"""
        similarities = self._cosine_similarity(query_embedding, record.embeddings)
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])
        matched_question = record.questions[best_index]

        if self.aggregation == "max":
            route_score = best_score
        elif self.aggregation == "mean":
            route_score = float(np.mean(similarities))
        elif self.aggregation == "centroid":
            centroid = np.mean(record.embeddings, axis=0)
            route_score = float(self._cosine_similarity(query_embedding, centroid.reshape(1, -1))[0])
        else:  # pragma: no cover - __init__ 已校验
            raise ValueError(f"不支持的 aggregation: {self.aggregation}")

        return route_score, matched_question

    def add_route(
        self,
        questions: List[str],
        target: str,
        route_name: Optional[str] = None,
        threshold: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        append: bool = True,
        use_cache: Optional[bool] = None,
    ) -> None:
        """
        添加或更新一个语义路由。

        Parameters
        ----------
        questions:
            该路由的代表性问题列表。建议每个 target 至少提供 3-10 条表达。
        target:
            命中的业务目标，例如：refund、greeting、search_docs、call_tool_x。
        route_name:
            路由名称。默认使用 target。多个 route 可以指向同一个 target。
        threshold:
            当前路由独立阈值。None 表示使用 default_threshold。
        metadata:
            业务元数据，例如工具名、权限、租户、领域等。
        append:
            True 表示同名 route 已存在时追加问题；False 表示覆盖。
        use_cache:
            调用 embedding_service 时是否启用缓存。None 表示使用路由器默认配置。
        """
        if not isinstance(questions, list):
            raise TypeError("questions 必须是 List[str]")
        if not questions:
            raise ValueError("questions 不能为空")
        if not target or not target.strip():
            raise ValueError("target 不能为空")

        clean_questions = []
        seen = set()
        for item in questions:
            if not isinstance(item, str):
                raise TypeError("questions 中的每一项都必须是字符串")
            clean_item = item.strip()
            if not clean_item:
                raise ValueError("questions 中不能包含空字符串")
            if clean_item not in seen:
                clean_questions.append(clean_item)
                seen.add(clean_item)

        clean_target = target.strip()
        clean_route_name = self._safe_route_name(route_name or clean_target)
        clean_threshold = self._validate_threshold(threshold)
        clean_metadata = dict(metadata or {})

        with self._lock:
            if append and clean_route_name in self._routes:
                existing = self._routes[clean_route_name]
                merged_questions = list(existing.questions)
                for item in clean_questions:
                    if item not in merged_questions:
                        merged_questions.append(item)

                final_threshold = clean_threshold if clean_threshold is not None else existing.threshold
                final_metadata = dict(existing.metadata)
                final_metadata.update(clean_metadata)
                created_at = existing.created_at
            else:
                merged_questions = clean_questions
                final_threshold = clean_threshold
                final_metadata = clean_metadata
                created_at = int(time.time())

            embeddings = self._embed(merged_questions, is_query=False, use_cache=use_cache)
            embeddings = self._normalize_embeddings(len(merged_questions), embeddings)

            self._routes[clean_route_name] = _RouteRecord(
                route_name=clean_route_name,
                target=clean_target,
                questions=merged_questions,
                embeddings=embeddings,
                threshold=final_threshold,
                metadata=final_metadata,
                created_at=created_at,
                updated_at=int(time.time()),
            )

    def remove_route(self, route_name: str) -> bool:
        """删除指定路由。"""
        if not route_name or not route_name.strip():
            raise ValueError("route_name 不能为空")

        clean_route_name = self._safe_route_name(route_name)
        with self._lock:
            return self._routes.pop(clean_route_name, None) is not None

    def clear(self) -> None:
        """清空所有路由。"""
        with self._lock:
            self._routes.clear()

    def list_routes(self) -> List[Dict[str, Any]]:
        """列出当前路由配置，不包含 embedding。"""
        with self._lock:
            return [record.to_config() for record in self._routes.values()]

    def has_route(self, route_name: str) -> bool:
        """判断指定路由是否存在。"""
        if not route_name or not route_name.strip():
            return False
        return self._safe_route_name(route_name) in self._routes

    def route(
        self,
        question: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        fallback_target: Optional[str] = None,
        return_candidates: bool = True,
        use_cache: Optional[bool] = None,
    ) -> RouteDecision:
        """
        对单条问题进行语义路由。

        Returns
        -------
        RouteDecision
            包含最终 target、分数、阈值、命中示例、候选列表、耗时和拒绝原因。
        """
        start = time.perf_counter()

        if not isinstance(question, str):
            raise TypeError("question 必须是字符串")
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question 不能为空")

        clean_threshold = self._validate_threshold(threshold, field_name="threshold")
        actual_top_k = self.top_k if top_k is None else int(top_k)
        if actual_top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        actual_fallback = self.fallback_target if fallback_target is None else fallback_target

        try:
            with self._lock:
                route_records = list(self._routes.values())

            if not route_records:
                latency_ms = (time.perf_counter() - start) * 1000
                decision = RouteDecision(
                    question=clean_question,
                    target=actual_fallback,
                    route_name=None,
                    score=0.0,
                    threshold=clean_threshold if clean_threshold is not None else self.default_threshold,
                    matched_question=None,
                    accepted=False,
                    candidates=[],
                    latency_ms=latency_ms,
                    reason="no_routes",
                )
                self._update_stats(decision)
                return decision

            query_embedding = self._embed(clean_question, is_query=True, use_cache=use_cache)
            query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

            candidates: List[RouteCandidate] = []
            for record in route_records:
                route_threshold = (
                    clean_threshold
                    if clean_threshold is not None
                    else record.threshold
                    if record.threshold is not None
                    else self.default_threshold
                )
                score, matched_question = self._route_score(query_embedding, record)
                candidates.append(
                    RouteCandidate(
                        target=record.target,
                        route_name=record.route_name,
                        score=score,
                        threshold=route_threshold,
                        matched_question=matched_question,
                        accepted=score >= route_threshold,
                        metadata=dict(record.metadata),
                    )
                )

            candidates.sort(key=lambda item: item.score, reverse=True)
            top_candidates = candidates[:actual_top_k]
            best = candidates[0]
            second = candidates[1] if len(candidates) > 1 else None
            score_margin = best.score - second.score if second is not None else float("inf")

            if best.score < best.threshold:
                accepted = False
                reason = "below_threshold"
            elif score_margin < self.min_score_margin:
                accepted = False
                reason = "ambiguous"
            else:
                accepted = True
                reason = "accepted"

            latency_ms = (time.perf_counter() - start) * 1000
            decision = RouteDecision(
                question=clean_question,
                target=best.target if accepted else actual_fallback,
                route_name=best.route_name if accepted else None,
                score=best.score,
                threshold=best.threshold,
                matched_question=best.matched_question,
                accepted=accepted,
                candidates=top_candidates if return_candidates else [],
                latency_ms=latency_ms,
                reason=reason,
            )
            self._update_stats(decision)
            return decision
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            default_decision = RouteDecision(
                question=clean_question,
                target=actual_fallback,
                route_name=None,
                score=0.0,
                threshold=clean_threshold if clean_threshold is not None else self.default_threshold,
                matched_question=None,
                accepted=False,
                candidates=[],
                latency_ms=latency_ms,
                reason="error",
            )
            return self._handle_error("Semantic route failed", exc, default=default_decision)

    def route_many(
        self,
        questions: List[str],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        fallback_target: Optional[str] = None,
        return_candidates: bool = True,
        use_cache: Optional[bool] = None,
    ) -> List[RouteDecision]:
        """批量路由。为了保持清晰与稳定，这里复用单条 route 逻辑。"""
        if not isinstance(questions, list):
            raise TypeError("questions 必须是 List[str]")
        return [
            self.route(
                question=item,
                threshold=threshold,
                top_k=top_k,
                fallback_target=fallback_target,
                return_candidates=return_candidates,
                use_cache=use_cache,
            )
            for item in questions
        ]

    def predict(self, question: str, **kwargs) -> Optional[str]:
        """只返回 target，适合在业务代码中快速使用。"""
        return self.route(question, **kwargs).target

    def explain(self, question: str, top_k: Optional[int] = None, **kwargs) -> RouteDecision:
        """返回带候选详情的路由解释结果。"""
        return self.route(question, top_k=top_k, return_candidates=True, **kwargs)

    def _update_stats(self, decision: RouteDecision) -> None:
        """更新运行期统计信息。"""
        with self._lock:
            self._total += 1
            self._total_latency_ms += decision.latency_ms
            if decision.accepted:
                self._accepted += 1
            else:
                self._rejected += 1

    def stats(self) -> RouterStats:
        """读取当前统计信息。"""
        with self._lock:
            return RouterStats(
                total=self._total,
                accepted=self._accepted,
                rejected=self._rejected,
                total_latency_ms=self._total_latency_ms,
            )

    def reset_stats(self) -> None:
        """重置统计信息。"""
        with self._lock:
            self._total = 0
            self._accepted = 0
            self._rejected = 0
            self._total_latency_ms = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """导出路由器配置，不包含 embedding。"""
        with self._lock:
            return {
                "version": "1.0",
                "name": self.name,
                "default_threshold": self.default_threshold,
                "min_score_margin": self.min_score_margin,
                "aggregation": self.aggregation,
                "top_k": self.top_k,
                "fallback_target": self.fallback_target,
                "normalize_embeddings": self.normalize_embeddings,
                "routes": [record.to_config() for record in self._routes.values()],
            }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        embedding_service: Optional[object] = None,
        embedder: Optional[object] = None,
        embed_fn: Optional[EmbedFn] = None,
        use_cache: bool = True,
        raise_on_error: bool = True,
    ) -> "SemanticRouter":
        """从配置字典恢复路由器，并重新生成 route embedding。"""
        router = cls(
            name=str(payload.get("name") or "semantic_router"),
            embedding_service=embedding_service,
            embedder=embedder,
            embed_fn=embed_fn,
            default_threshold=float(payload.get("default_threshold", 0.65)),
            min_score_margin=float(payload.get("min_score_margin", 0.0)),
            aggregation=str(payload.get("aggregation", "max")),
            top_k=int(payload.get("top_k", 3)),
            fallback_target=payload.get("fallback_target"),
            normalize_embeddings=bool(payload.get("normalize_embeddings", True)),
            use_cache=use_cache,
            raise_on_error=raise_on_error,
        )

        for route_config in payload.get("routes", []):
            router.add_route(
                questions=list(route_config["questions"]),
                target=str(route_config["target"]),
                route_name=str(route_config.get("route_name") or route_config["target"]),
                threshold=route_config.get("threshold"),
                metadata=dict(route_config.get("metadata") or {}),
                append=False,
                use_cache=use_cache,
            )

            # 保留原始时间戳，方便配置审计。
            route_name = router._safe_route_name(str(route_config.get("route_name") or route_config["target"]))
            if route_name in router._routes:
                router._routes[route_name].created_at = int(route_config.get("created_at") or router._routes[route_name].created_at)
                router._routes[route_name].updated_at = int(route_config.get("updated_at") or router._routes[route_name].updated_at)

        return router

    def save(self, path: Union[str, Path]) -> None:
        """保存路由配置到 JSON 文件，不保存 embedding。"""
        path_obj = Path(path).expanduser()
        if path_obj.parent and not path_obj.parent.exists():
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        payload = self.to_dict()
        path_obj.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        embedding_service: Optional[object] = None,
        embedder: Optional[object] = None,
        embed_fn: Optional[EmbedFn] = None,
        use_cache: bool = True,
        raise_on_error: bool = True,
    ) -> "SemanticRouter":
        """从 JSON 文件加载路由配置，并重新生成 route embedding。"""
        path_obj = Path(path).expanduser()
        if not path_obj.exists():
            raise FileNotFoundError(f"路由配置文件不存在：{path_obj}")

        payload = json.loads(path_obj.read_text(encoding="utf-8"))
        return cls.from_dict(
            payload,
            embedding_service=embedding_service,
            embedder=embedder,
            embed_fn=embed_fn,
            use_cache=use_cache,
            raise_on_error=raise_on_error,
        )

    def __len__(self) -> int:
        """返回路由数量。"""
        return len(self._routes)

    def __contains__(self, route_name: str) -> bool:
        """支持：'refund' in router。"""
        return self.has_route(route_name)

    def __call__(self, question: str, **kwargs) -> RouteDecision:
        """让 router(question) 等价于 router.route(question)。"""
        return self.route(question, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # 本地 demo 使用 HashingTextEmbedder，不依赖真实模型。
    # 生产环境建议传入 EmbeddingsCache.py 中的 CachedEmbeddingService。
    router = SemanticRouter(
        name="demo_semantic_router",
        default_threshold=0.25,
        top_k=3,
        fallback_target="unknown",
    )

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "Hello", "你好"],
        target="greeting",
    )
    router.add_route(
        questions=["如何退货", "我要退款", "商品不满意可以退吗", "怎么申请售后"],
        target="refund",
    )

    decision = router("Hi, good morning")
    print("路由结果：", decision.to_dict())

    decision = router.explain("我想申请退货退款", top_k=2)
    print("解释结果：", decision.to_dict())

    print("只返回 target：", router.predict("你好"))
    print("统计信息：", router.stats())
