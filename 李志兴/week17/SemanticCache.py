r"""
SemanticCache.py

一个可复用的语义缓存工具。

核心能力：
1. 使用 Redis 保存 prompt、response、元数据和过期时间。
2. 使用 FAISS 保存 prompt embedding，并通过统一 ID 与 Redis 数据对应。
3. 支持单条写入、批量写入、相似 prompt 查询和缓存命中元数据返回。
4. 支持 cosine / l2 两种相似度度量。
5. 缓存 key 中包含业务命名空间、模型标识和度量方式，避免不同模型或不同用途的数据混用。
6. 提供本地 Sentence-BERT / BGE 模型封装，方便直接组合成语义缓存服务。

依赖安装：
    pip install numpy redis faiss-cpu sentence-transformers

运行前请确保：
1. Redis 服务已启动。
2. 本地模型路径存在。

Windows 本地测试模型路径：
    D:\桌面\typora文件\八斗AI\models\sentence_bert\bge-small-zh-v1.5
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


PromptInput = Union[str, Sequence[str]]
ResponseInput = Union[Any, Sequence[Any]]
EmbeddingArray = np.ndarray

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = r"D:\桌面\typora文件\八斗AI\models\sentence_bert\bge-small-zh-v1.5"


@dataclass(frozen=True)
class CacheHit:
    """一次语义缓存命中的详细信息。"""

    id: int
    prompt: str
    response: Any
    score: float
    metric: str  # 向量相似度度量方式


class SemanticCache:
    """
    基于 Redis + FAISS 的语义缓存层。

    这个类只负责“存储、读取、删除语义缓存”，不负责生成 response。
    embedding_method 只负责把 prompt 转为 embedding；response 可以来自 LLM、规则系统或任意业务函数。

    Parameters
    ----------
    name:
        业务命名空间，例如："faq_bot"、"rag_answer_cache"。
    embedding_method:
        外部 embedding 函数。输入 List[str]，返回二维 numpy 数组或可转成 numpy 的对象。
    model_id:
        当前 embedding 模型的唯一标识。建议传入模型路径或模型名。
        它会参与 Redis key 和 FAISS index 文件名计算，防止不同模型的数据混用。
    ttl:
        Redis 缓存过期时间，单位秒。None 或 0 表示不过期。
    metric:
        相似度度量方式。"cosine" 表示越大越相似；"l2" 表示越小越相似。
    threshold:
        命中阈值。cosine 默认 0.85，l2 默认 0.1。
    redis_host / redis_port / redis_db / redis_password:
        Redis 连接信息。
    redis_client:
        可选的 Redis 兼容客户端，方便测试或接入已有连接池。
    key_prefix:
        Redis key 前缀。
    index_path:
        FAISS index 文件路径。None 时根据 name、model_id、metric 自动生成。
    raise_on_error:
        True 时 Redis / FAISS 异常直接抛出；False 时记录日志并返回安全默认值。
    """

    def __init__(
        self,
        name: str,
        embedding_method: Callable[[List[str]], Any],
        model_id: str = "default-model",
        ttl: Optional[int] = 3600 * 24,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        redis_client: Optional[object] = None,
        key_prefix: str = "semantic_cache",
        index_path: Optional[str] = None,
        metric: str = "cosine",
        threshold: Optional[float] = None,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        use_redis_lock: bool = True,
        redis_lock_timeout: int = 300,
        redis_lock_blocking_timeout: int = 30,
        raise_on_error: bool = True,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("name 不能为空")
        if not model_id or not str(model_id).strip():
            raise ValueError("model_id 不能为空")
        if embedding_method is None or not callable(embedding_method):
            raise ValueError("embedding_method 必须是可调用对象")

        metric = metric.lower().strip()
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric 只支持 'cosine' 或 'l2'")

        self.name = name.strip()
        self.model_id = str(model_id).strip()
        self.embedding_method = embedding_method
        self.ttl = ttl
        self.metric = metric
        self.threshold = self._default_threshold(metric) if threshold is None else float(threshold)
        self.key_prefix = key_prefix.strip() or "semantic_cache"
        self.raise_on_error = raise_on_error
        self.use_redis_lock = use_redis_lock
        self.redis_lock_timeout = redis_lock_timeout
        self.redis_lock_blocking_timeout = redis_lock_blocking_timeout

        self._thread_lock = threading.RLock()

        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "缺少 faiss 依赖，请先安装：pip install faiss-cpu"
            ) from exc
        self.faiss = faiss

        if redis_client is not None:
            self.redis = redis_client
        else:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "缺少 redis 依赖，请先安装：pip install redis"
                ) from exc

            # decode_responses 必须为 False，否则二进制/字节数据可能被错误解码。
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=30,
            )

        self.readable_name = self._safe_part(self.name)
        self.model_hash = self._sha256(self.model_id)[:16]
        self.readable_metric = self._safe_part(self.metric)
        self.prefix = (
            f"{self.key_prefix}:{self.readable_name}:{self.model_hash}:{self.readable_metric}"
        )

        self.ids_key = f"{self.prefix}:ids" # 保存所有已经存在的ids集合
        self.next_id_key = f"{self.prefix}:next_id" # 保存写一个可用id
        self.meta_key = f"{self.prefix}:meta" # 缓存系统的元信息

        self.index_path = index_path or self._default_index_path()
        self.index: Optional[object] = None

        try:
            self._load_index_from_disk()
            self._write_meta()
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis / FAISS 服务
            self._handle_error("Initialize semantic cache failed", exc, default=None)

    def ping(self) -> bool:
        """检查 Redis 是否可用。"""
        try:
            return bool(self.redis.ping())
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Redis ping failed", exc, default=False)

    def _handle_error(self, message: str, exc: Exception, default):
        """统一处理 Redis / FAISS 异常。"""
        logger.exception("%s: %s", message, exc)
        if self.raise_on_error:
            raise exc
        return default

    @staticmethod
    def _default_threshold(metric: str) -> float:
        """根据度量方式给出默认命中阈值。"""
        return 0.85 if metric == "cosine" else 0.1

    @staticmethod
    def _safe_part(value: str, max_len: int = 48) -> str:
        """把 key / 文件名中的可读部分限制在安全字符范围内。"""
        safe = []
        for ch in value:
            if ch.isalnum() or ch in {"_", "-"}:
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)[:max_len] or "default"

    @staticmethod
    def _sha256(value: str) -> str:
        """计算文本的 SHA256 哈希。"""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _ensure_prompt_list(prompt: PromptInput) -> Tuple[List[str], bool]:
        """
        统一把输入转成 List[str]。

        Returns
        -------
        prompts:
            prompt 列表。
        single_input:
            原始输入是否为单条字符串。
        """
        single_input = isinstance(prompt, str)
        prompts = [prompt] if single_input else list(prompt)

        for item in prompts:
            if not isinstance(item, str):
                raise TypeError("prompt 中的每一项都必须是字符串")
            if not item.strip():
                raise ValueError("prompt 不能为空字符串")

        return prompts, single_input

    def _normalize_store_inputs(
        self,
        prompt: PromptInput,
        response: ResponseInput,
    ) -> Tuple[List[str], List[Any]]:
        """统一处理单条写入和批量写入。"""
        prompts, single_input = self._ensure_prompt_list(prompt)

        if single_input:
            responses = [response]
        else:
            if isinstance(response, (str, bytes, bytearray, dict)):
                raise ValueError("当 prompt 是列表时，response 也必须是等长列表")
            try:
                responses = list(response)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError("当 prompt 是列表时，response 也必须是等长列表") from exc

        if len(prompts) != len(responses):
            raise ValueError(
                "prompt 和 response 数量必须一致："
                f"prompt={len(prompts)}, response={len(responses)}"
            )

        for item in responses:
            self._json_dumps(item)

        return prompts, responses

    def _prompt_hash(self, prompt: str) -> str:
        """计算 prompt 在当前 namespace / model / metric 下的哈希。"""
        return self._sha256(
            json.dumps(
                {
                    "name": self.name,
                    "model_id": self.model_id,
                    "metric": self.metric,
                    "prompt": prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def _entry_key(self, cache_id: int) -> str:
        """生成指定缓存 ID 的 Redis entry key。"""
        return f"{self.prefix}:entry:{cache_id}"

    def _prompt_map_key(self, prompt_hash: str) -> str:
        """生成 prompt hash 到缓存 ID 的 Redis 映射 key。"""
        return f"{self.prefix}:prompt:{prompt_hash}"

    def _default_index_path(self) -> str:
        """根据 name / model_id / metric 生成默认 FAISS index 文件名。"""
        filename = f"{self.readable_name}_{self.model_hash}_{self.readable_metric}.index"
        return str(Path(filename))

    def _write_meta(self) -> None:
        """写入当前语义缓存的元数据，便于排查问题。"""
        self.redis.hset( # 写入一个hash表数据结构
            self.meta_key,
            mapping={
                b"name": self.name.encode("utf-8"),
                b"model_id": self.model_id.encode("utf-8"),
                b"metric": self.metric.encode("utf-8"),
                b"threshold": str(self.threshold).encode("utf-8"),
                b"index_path": self.index_path.encode("utf-8"),
                b"updated_at": str(int(time.time())).encode("utf-8"),
            },
        )

    def _next_id(self) -> int:
        """获取新的语义缓存 ID。"""
        return int(self.redis.incr(self.next_id_key))

    def _redis_set(self, pipe, key: str, value: bytes) -> None:
        """根据 ttl 选择 set 或 setex。"""
        if self.ttl and self.ttl > 0:
            pipe.setex(key, int(self.ttl), value)
        else:
            pipe.set(key, value)

    def _serialize_payload(
        self,
        cache_id: int,
        prompt: str,
        response: Any,
        prompt_hash: str,
    ) -> bytes:
        """序列化单条语义缓存数据。"""
        payload = {
            "id": cache_id,
            "prompt": prompt,
            "response": response,
            "created_at": int(time.time()),
            "metric": self.metric,
            "model_id": self.model_id,
            "prompt_hash": prompt_hash,
        }
        return self._json_dumps(payload).encode("utf-8")

    @staticmethod
    def _deserialize_payload(value: Optional[bytes]) -> Optional[Dict[str, Any]]:
        """反序列化单条语义缓存数据。"""
        if value is None:
            return None

        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return json.loads(value)
        except Exception as exc:
            raise ValueError("反序列化语义缓存数据失败") from exc

    def _get_payload(self, cache_id: int) -> Optional[Dict[str, Any]]:
        """从 Redis 中读取单条语义缓存数据。"""
        return self._deserialize_payload(self.redis.get(self._entry_key(cache_id)))

    def _embed(self, prompts: List[str]) -> EmbeddingArray:
        """调用外部 embedding_method，并统一转换为 FAISS 可用的 float32 二维矩阵。"""
        vectors = self.embedding_method(prompts)
        arr = np.asarray(vectors, dtype=np.float32)

        if len(prompts) == 1 and arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError("embedding_method 必须返回二维数组，shape 应为 [n, dim]")
        if arr.shape[0] != len(prompts):
            raise ValueError(
                "embedding 返回数量错误："
                f"期望 {len(prompts)} 条，实际 {arr.shape[0]} 条"
            )
        if arr.shape[1] <= 0:
            raise ValueError("embedding 维度必须大于 0")
        if not np.all(np.isfinite(arr)):
            raise ValueError("embedding 中包含 NaN 或 Inf")

        arr = np.ascontiguousarray(arr, dtype=np.float32)

        if self.metric == "cosine":
            norms = np.linalg.norm(arr, axis=1)
            if np.any(norms == 0):
                raise ValueError("cosine 模式下 embedding 不能是零向量")
            self.faiss.normalize_L2(arr)

        return arr

    def _create_index(self, dim: int):
        """创建可指定向量 ID 的 FAISS index。"""
        if self.metric == "cosine":
            base_index = self.faiss.IndexFlatIP(dim)
        else:
            base_index = self.faiss.IndexFlatL2(dim)
        return self.faiss.IndexIDMap2(base_index)

    def _ensure_index(self, dim: int) -> None:
        """确保 FAISS index 存在，并且 embedding 维度一致。"""
        if self.index is None:
            self.index = self._create_index(dim)
            return

        if int(self.index.d) != dim:
            raise ValueError(
                "embedding 维度和已有 FAISS index 不一致："
                f"已有维度 {self.index.d}，当前维度 {dim}。"
                "如果更换了 embedding 模型，请使用新的 cache name / model_id，或执行 clear_cache。"
            )

    def _load_index_from_disk(self) -> None:
        """从磁盘加载 FAISS index。"""
        if os.path.exists(self.index_path):
            self.index = self.faiss.read_index(self.index_path)
        else:
            self.index = None

    def _save_index_to_disk(self) -> None:
        """原子写入 FAISS index 文件。"""
        if self.index is None:
            return

        index_dir = os.path.dirname(os.path.abspath(self.index_path))
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)

        tmp_path = f"{self.index_path}.tmp.{os.getpid()}"
        self.faiss.write_index(self.index, tmp_path)
        os.replace(tmp_path, self.index_path)

    def _remove_ids_from_index_locked(self, ids: List[int]) -> None:
        """在已持有锁的情况下，从 FAISS index 和 ids 集合中删除指定 ID。"""
        if not ids:
            return

        self._load_index_from_disk()
        if self.index is not None and self.index.ntotal > 0:
            id_array = np.asarray(ids, dtype=np.int64)
            self.index.remove_ids(id_array)
            self._save_index_to_disk()

        pipe = self.redis.pipeline(transaction=False)
        for cache_id in ids:
            pipe.srem(self.ids_key, str(cache_id).encode("utf-8"))
        pipe.execute()

    def _remove_ids_from_index(self, ids: List[int]) -> None:
        """从 FAISS index 和 ids 集合中删除指定 ID。"""
        if not ids:
            return

        try:
            with self._lock():
                self._remove_ids_from_index_locked(ids)
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis / FAISS 服务
            self._handle_error("Remove ids from FAISS index failed", exc, default=None)

    def _pass_threshold(self, score: float) -> bool:
        """判断搜索分数是否达到命中阈值。"""
        if self.metric == "cosine":
            return score >= float(self.threshold)
        return score <= float(self.threshold)

    def _json_dumps(self, obj: Any) -> str:
        """统一 JSON 序列化，保证中文不转义，并尽量减少 Redis 体积。"""
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def _lock(self):
        """返回本地线程锁，或本地线程锁 + Redis 分布式锁。"""
        if not self.use_redis_lock:
            return self._thread_lock

        return _CombinedLock(
            thread_lock=self._thread_lock,
            redis_lock=self.redis.lock(
                f"{self.prefix}:lock",
                timeout=self.redis_lock_timeout,
                blocking_timeout=self.redis_lock_blocking_timeout,
            ),
        )

    def store(
        self,
        prompt: PromptInput,
        response: ResponseInput,
    ) -> List[int]:
        """
        写入语义缓存。

        Parameters
        ----------
        prompt:
            单条 prompt 或 prompt 列表。
        response:
            单条 response 或与 prompt 等长的 response 列表。

        Returns
        -------
        List[int]
            写入或更新后的缓存 ID 列表。
        """
        prompts, responses = self._normalize_store_inputs(prompt, response)
        if not prompts:
            return []

        try:
            embeddings = self._embed(prompts)

            ids: List[int] = []
            new_vectors: List[EmbeddingArray] = []
            new_ids: List[int] = []

            with self._lock():
                self._load_index_from_disk()
                self._ensure_index(dim=embeddings.shape[1])

                with self.redis.pipeline(transaction=False) as pipe:
                    for idx, (item_prompt, item_response) in enumerate(zip(prompts, responses)):
                        prompt_hash = self._prompt_hash(item_prompt)
                        prompt_map_key = self._prompt_map_key(prompt_hash)

                        existing_id = self.redis.get(prompt_map_key)
                        cache_id: Optional[int] = None

                        if existing_id is not None:
                            try:
                                cache_id = int(existing_id)
                            except (TypeError, ValueError):
                                cache_id = None

                        if cache_id is None or not self.redis.exists(self._entry_key(cache_id)):
                            cache_id = self._next_id()
                            new_vectors.append(embeddings[idx])
                            new_ids.append(cache_id)

                        payload_bytes = self._serialize_payload(
                            cache_id=cache_id,
                            prompt=item_prompt,
                            response=item_response,
                            prompt_hash=prompt_hash,
                        )

                        self._redis_set(pipe, self._entry_key(cache_id), payload_bytes)
                        self._redis_set(pipe, prompt_map_key, str(cache_id).encode("utf-8"))
                        pipe.sadd(self.ids_key, str(cache_id).encode("utf-8"))

                        ids.append(cache_id)

                    pipe.execute()

                if new_vectors:
                    vector_array = np.vstack(new_vectors).astype(np.float32)
                    id_array = np.asarray(new_ids, dtype=np.int64)
                    self.index.add_with_ids(vector_array, id_array)
                    self._save_index_to_disk()

            return ids
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis / FAISS 服务
            return self._handle_error("Store semantic cache failed", exc, default=[])

    # set 是 store 的别名，更符合缓存组件的直觉用法。
    set = store

    def search(
        self,
        prompt: str,
        top_k: int = 5,
        oversample: int = 5,
    ) -> List[CacheHit]:
        """搜索相似 prompt，返回命中列表。"""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if oversample <= 0:
            raise ValueError("oversample 必须大于 0")

        try:
            with self._thread_lock:
                self._load_index_from_disk()
                if self.index is None or self.index.ntotal == 0:
                    return []

                query_vector = self._embed([prompt])
                search_k = min(max(top_k * oversample, top_k), self.index.ntotal)
                scores, ids = self.index.search(query_vector, search_k)

                hits: List[CacheHit] = []
                expired_ids: List[int] = []

                for raw_score, raw_id in zip(scores[0], ids[0]):
                    cache_id = int(raw_id)
                    if cache_id < 0:
                        continue

                    score = float(raw_score)
                    if not self._pass_threshold(score):
                        continue

                    payload = self._get_payload(cache_id)
                    if payload is None:
                        expired_ids.append(cache_id)
                        continue

                    hits.append(
                        CacheHit(
                            id=cache_id,
                            prompt=payload["prompt"],
                            response=payload["response"],
                            score=score,
                            metric=self.metric,
                        )
                    )

                    if len(hits) >= top_k:
                        break

            if expired_ids:
                self._remove_ids_from_index(expired_ids)

            return hits
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis / FAISS 服务
            return self._handle_error("Search semantic cache failed", exc, default=[])

    def call(
        self,
        prompt: str,
        top_k: int = 1,
        return_metadata: bool = False,
    ) -> Optional[Union[Any, CacheHit, List[CacheHit]]]:
        """读取语义缓存。默认只返回最佳命中的 response。"""
        hits = self.search(prompt, top_k=top_k)
        if not hits:
            return None

        if return_metadata:
            return hits[0] if top_k == 1 else hits

        return hits[0].response

    # get 是 call 的别名，更符合缓存组件的直觉用法。
    get = call

    def delete(self, prompt: PromptInput) -> int:
        """删除指定 prompt 对应的语义缓存。"""
        prompts, _ = self._ensure_prompt_list(prompt)
        if not prompts:
            return 0

        try:
            deleted_ids: List[int] = []

            with self._lock():
                with self.redis.pipeline(transaction=False) as pipe:
                    for item_prompt in prompts:
                        prompt_hash = self._prompt_hash(item_prompt)
                        prompt_map_key = self._prompt_map_key(prompt_hash)
                        raw_id = self.redis.get(prompt_map_key)
                        if raw_id is None:
                            pipe.delete(prompt_map_key)
                            continue

                        try:
                            cache_id = int(raw_id)
                        except (TypeError, ValueError):
                            pipe.delete(prompt_map_key)
                            continue

                        pipe.delete(self._entry_key(cache_id))
                        pipe.delete(prompt_map_key)
                        deleted_ids.append(cache_id)

                    pipe.execute()

                self._remove_ids_from_index_locked(deleted_ids)

            return len(deleted_ids)
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis / FAISS 服务
            return self._handle_error("Delete semantic cache failed", exc, default=0)

    def get_or_generate(
        self,
        prompt: str,
        generate_fn: Callable[[str], Any],
        top_k: int = 1,
        return_metadata: bool = False,
    ) -> Union[Any, CacheHit, List[CacheHit]]:
        """
        缓存优先读取；未命中时调用 generate_fn 生成 response 并写回缓存。
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串")
        if generate_fn is None or not callable(generate_fn):
            raise ValueError("generate_fn 必须是可调用对象")

        cached = self.call(prompt, top_k=top_k, return_metadata=return_metadata)
        if cached is not None:
            return cached

        response = generate_fn(prompt)
        self.store(prompt, response)

        if return_metadata:
            hits = self.search(prompt, top_k=top_k)
            return hits[0] if top_k == 1 else hits

        return response

    def clear_cache(self) -> None:
        """清空当前 namespace 下的 Redis 数据和本地 FAISS index 文件。"""
        try:
            with self._lock():
                cursor = 0
                pattern = f"{self.prefix}:*"

                while True:
                    cursor, keys = self.redis.scan(cursor=cursor, match=pattern, count=500)
                    if keys:
                        self.redis.delete(*keys)
                    if cursor == 0:
                        break

                if os.path.exists(self.index_path):
                    os.remove(self.index_path)

                self.index = None
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis / FAISS 服务
            self._handle_error("Clear semantic cache failed", exc, default=None)

    def rebuild_index(self) -> int:
        """根据 Redis 中仍未过期的缓存数据重建 FAISS index。"""
        try:
            with self._lock():
                raw_ids = self.redis.smembers(self.ids_key)
                if not raw_ids:
                    self.index = None
                    if os.path.exists(self.index_path):
                        os.remove(self.index_path)
                    return 0

                alive_ids: List[int] = []
                prompts: List[str] = []
                stale_ids: List[bytes] = []

                for raw_id in raw_ids:
                    try:
                        cache_id = int(raw_id)
                    except (TypeError, ValueError):
                        stale_ids.append(raw_id)
                        continue

                    payload = self._get_payload(cache_id)
                    if payload is None:
                        stale_ids.append(raw_id)
                        continue

                    alive_ids.append(cache_id)
                    prompts.append(payload["prompt"])

                if stale_ids:
                    self.redis.srem(self.ids_key, *stale_ids)

                if not alive_ids:
                    self.index = None
                    if os.path.exists(self.index_path):
                        os.remove(self.index_path)
                    return 0

                embeddings = self._embed(prompts)
                self.index = self._create_index(dim=embeddings.shape[1])
                self.index.add_with_ids(
                    embeddings.astype(np.float32),
                    np.asarray(alive_ids, dtype=np.int64),
                )
                self._save_index_to_disk()

                return len(alive_ids)
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis / FAISS 服务
            return self._handle_error("Rebuild FAISS index failed", exc, default=0)


class _CombinedLock:
    """同时使用本地线程锁和 Redis 分布式锁。"""

    def __init__(self, thread_lock: threading.RLock, redis_lock) -> None:
        self.thread_lock = thread_lock
        self.redis_lock = redis_lock
        self._thread_acquired = False
        self._redis_acquired = False

    def __enter__(self):
        self.thread_lock.acquire()
        self._thread_acquired = True

        try:
            self._redis_acquired = bool(self.redis_lock.acquire())
        except Exception:
            self.thread_lock.release()
            self._thread_acquired = False
            raise

        if not self._redis_acquired:
            self.thread_lock.release()
            self._thread_acquired = False
            raise TimeoutError("获取 Redis 分布式锁失败，请稍后重试")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._redis_acquired:
                try:
                    self.redis_lock.release()
                except Exception as exc:
                    # Redis 锁过期或已被释放时，release 可能抛错；此处不影响主流程。
                    if exc.__class__.__name__ != "LockNotOwnedError":
                        logger.warning("Release Redis lock failed: %s", exc)
        finally:
            if self._thread_acquired:
                self.thread_lock.release()

        return False


class SentenceBertEmbedder:
    """
    本地 SentenceTransformer / BGE 模型封装。

    这个类只负责“把文本转为 embedding”，不关心语义缓存。
    与 SemanticCache 组合后，就可以成为可复用的语义缓存底层服务。
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        query_instruction: Optional[str] = "为这个句子生成表示以用于检索相关文章：",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.query_instruction = query_instruction

        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

        # 如果传入的是本地路径，提前检查路径是否存在，避免 SentenceTransformer 报错不直观。
        # 注意：Windows 路径请使用 raw string，例如 r"D:\xxx\model"。
        path_obj = Path(model_path).expanduser()
        looks_like_local_path = (
            path_obj.is_absolute()
            or model_path.startswith((".", "~"))
            or ":" in model_path
            or "\\" in model_path
        )
        if looks_like_local_path and not path_obj.exists():
            raise FileNotFoundError(
                "本地模型路径不存在，请检查 model_path："
                f"{model_path}"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "缺少 sentence-transformers 依赖，请先安装："
                "pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_path, device=device)

    def _prepare_texts(self, texts: List[str], is_query: bool) -> List[str]:
        """
        根据用途处理文本。

        BGE 类模型做检索时，query 侧通常可以加中文检索指令；
        document / cache prompt 侧通常直接使用原文。
        """
        if is_query and self.query_instruction:
            return [self.query_instruction + item for item in texts]
        return texts

    def encode(self, text: PromptInput, is_query: bool = False) -> EmbeddingArray:
        """生成 embedding。单条输入返回一维数组，批量输入返回二维数组。"""
        texts, single_input = SemanticCache._ensure_prompt_list(text)
        if not texts:
            raise ValueError("text 不能为空")

        encoded_texts = self._prepare_texts(texts, is_query=is_query)
        embeddings = self.model.encode(
            encoded_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
        ).astype(np.float32)

        if single_input and embeddings.ndim == 2:
            return embeddings[0]
        return embeddings

    def __call__(self, text: PromptInput, is_query: bool = False) -> EmbeddingArray:
        """调用 encode；当传入列表时返回二维数组，适合直接作为 SemanticCache.embedding_method。"""
        return self.encode(text, is_query=is_query)


class CachedSemanticService:
    """
    面向 AI 应用的高层语义缓存服务。

    用法：
        service.generate("用户问题")

    内部流程：
        1. 先查语义缓存；
        2. 缓存命中的直接返回；
        3. 缓存未命中的调用外部 generate_fn 生成 response；
        4. 新生成的 response 写回 Redis + FAISS；
        5. 返回 response 或命中元数据。
    """

    def __init__(
        self,
        cache: SemanticCache,
        generate_fn: Callable[[str], Any],
    ) -> None:
        if generate_fn is None or not callable(generate_fn):
            raise ValueError("generate_fn 必须是可调用对象")
        self.cache = cache
        self.generate_fn = generate_fn

    def generate(
        self,
        prompt: str,
        use_cache: bool = True,
        top_k: int = 1,
        return_metadata: bool = False,
    ) -> Union[Any, CacheHit, List[CacheHit]]:
        """获取 response。默认启用语义缓存。"""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 必须是非空字符串")

        if not use_cache:
            response = self.generate_fn(prompt)
            if return_metadata:
                return CacheHit(
                    id=-1,
                    prompt=prompt,
                    response=response,
                    score=0.0,
                    metric=self.cache.metric,
                )
            return response

        return self.cache.get_or_generate(
            prompt=prompt,
            generate_fn=self.generate_fn,
            top_k=top_k,
            return_metadata=return_metadata,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # 你指定的本地 BGE/Sentence-BERT 模型路径。
    # 注意：这里使用 raw string，避免 Windows 路径中的反斜杠被当作转义字符。
    MODEL_PATH = r"D:\桌面\typora文件\八斗AI\models\sentence_bert\bge-small-zh-v1.5"

    # 1. 初始化本地 embedding 模型。
    embedder = SentenceBertEmbedder(
        model_path=MODEL_PATH,
        device=None,  # 可改成 "cuda" 或 "cpu"；None 表示由 sentence-transformers 自动选择。
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # 2. 初始化 Redis + FAISS 语义缓存。
    semantic_cache = SemanticCache(
        name="demo_semantic_cache",
        model_id=MODEL_PATH,
        embedding_method=embedder,
        ttl=3600 * 24,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        metric="cosine",
        threshold=0.80,
        raise_on_error=True,
    )

    # 可选：提前检查 Redis 是否可用。
    semantic_cache.ping()

    # 3. 准备业务生成函数。真实项目中这里通常是 LLM 调用。
    def fake_llm(prompt: str) -> Dict[str, str]:
        return {"answer": f"这是针对『{prompt}』生成的回答。"}

    semantic_service = CachedSemanticService(
        cache=semantic_cache,
        generate_fn=fake_llm,
    )

    # 4. 写入一些 FAQ 语义缓存。
    semantic_cache.clear_cache()
    semantic_cache.store("如何申请退货？", {"answer": "您可以在订单详情页申请退货。"})
    semantic_cache.store("怎么修改手机号？", {"answer": "您可以在个人资料页面修改手机号。"})
    semantic_cache.store("订单什么时候发货？", {"answer": "正常情况下订单会在 48 小时内发货。"})

    # 5. 查询相似问题。通常可以命中“如何申请退货？”。
    hit = semantic_cache.get(
        "我想退货怎么办？",
        return_metadata=True,
    )
    print("语义缓存命中：", hit)

    # 6. 高层服务示例：命中则读缓存，未命中则调用 fake_llm 并写回缓存。
    response = semantic_service.generate("会员积分在哪里查看？")
    print("服务返回：", response)

    # 7. 删除缓存示例。
    deleted_count = semantic_cache.delete("如何申请退货？")
    print("删除语义缓存数量：", deleted_count)

    semantic_cache.clear_cache()
