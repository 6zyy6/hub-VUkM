r"""
EmbeddingsCache.py

一个可复用的 Embedding 缓存与生成工具。

核心能力：
1. 使用 Redis 缓存 embedding，避免重复计算。
2. 使用本地 Sentence-BERT / BGE 模型生成真实 embedding。
3. 支持单条文本与批量文本。
4. 支持缓存命中统计，方便集成到 RAG、Agent、语义搜索等 AI 应用中。
5. 缓存 key 中包含模型标识和向量用途，避免不同模型或不同用途的 embedding 混用。

依赖安装：
    pip install numpy redis sentence-transformers

运行前请确保：
1. Redis 服务已启动。
2. 本地模型路径存在。

Windows 本地测试模型路径：
    D:\桌面\typora文件\八斗AI\models\sentence_bert\bge-small-zh-v1.5
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np


TextInput = Union[str, Sequence[str]]
EmbeddingArray = np.ndarray

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = r"D:\桌面\typora文件\八斗AI\models\sentence_bert\bge-small-zh-v1.5"


@dataclass(frozen=True) # 对象创建后不能修改，只读
class CacheStats:
    """一次 embedding 调用的缓存统计信息。"""

    total: int
    hit: int
    miss: int

    @property  # 定义了一个只读属性
    def hit_rate(self) -> float:
        """缓存命中率。"""
        if self.total == 0:
            return 0.0
        return self.hit / self.total


class EmbeddingsCache:
    """
    基于 Redis 的 embedding 缓存层。

    这个类只负责“存储、读取、删除 embedding”，不负责生成 embedding。
    这样可以把缓存能力作为底层组件，复用于 RAG、语义搜索、推荐、Agent 记忆等场景。

    Parameters
    ----------
    name:
        业务命名空间，例如："rag_knowledge_base"、"chat_memory"。
    model_id:
        当前 embedding 模型的唯一标识。建议传入模型路径或模型名。
        它会参与缓存 key 计算，防止不同模型的向量混用。
    ttl:
        缓存过期时间，单位秒。None 或 0 表示不过期。
    redis_host / redis_port / redis_db / redis_password:
        Redis 连接信息。
    redis_client:
        可选的 Redis 兼容客户端，方便测试或接入已有连接池。
    key_prefix:
        Redis key 前缀。
    raise_on_error:
        True 时 Redis 异常直接抛出；False 时记录日志并返回安全默认值。
    """

    def __init__(
        self,
        name: str,
        model_id: str = "default-model",
        ttl: Optional[int] = 3600 * 24,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0, # Redis 的逻辑数据库编号，默认为0
        redis_password: Optional[str] = None,
        redis_client: Optional[object] = None, # 允许外部传入一个已经创建好的 Redis 客户端对象，让 EmbeddingsCache 直接复用它
        key_prefix: str = "embedding_cache",
        socket_timeout: float = 5.0, # 连接redis数据库后命令超时时间
        socket_connect_timeout: float = 5.0, # 连接redis超时时间
        raise_on_error: bool = True,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("name 不能为空")
        if not model_id or not str(model_id).strip():
            raise ValueError("model_id 不能为空")

        self.name = name.strip()
        self.model_id = str(model_id).strip()
        self.ttl = ttl
        self.key_prefix = key_prefix.strip() or "embedding_cache"
        self.raise_on_error = raise_on_error

        if redis_client is not None:
            self.redis = redis_client
        else:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "缺少 redis 依赖，请先安装：pip install redis"
                ) from exc

            # decode_responses 必须为 False，否则二进制 embedding 会被错误解码成字符串。
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

    def ping(self) -> bool:
        """检查 Redis 是否可用。"""
        try:
            return bool(self.redis.ping())
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Redis ping failed", exc, default=False)

    def _handle_error(self, message: str, exc: Exception, default):
        """统一处理 Redis 异常。"""
        logger.exception("%s: %s", message, exc)
        if self.raise_on_error:
            raise exc
        return default

    # 静态方法既不接收隐式参数 self 或 cls，也无法访问实例属性或类属性，本质是封装在类内的独立函数。
    @staticmethod
    def _safe_part(value: str, max_len: int = 48) -> str:
        """把 key 的可读部分限制在安全字符范围内。"""
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

    def _key(self, text: str, variant: str = "document") -> str:
        """
        生成 Redis key。

        variant 用于区分同一段文本在不同场景下的向量，例如：
        - document: 文档向量
        - query: 查询向量

        对 BGE 这类模型来说，query 可能会额外加检索指令，因此 query/document 不能共用缓存。
        """
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")

        variant = variant or "document"
        model_hash = self._sha256(self.model_id)[:16]
        text_hash = self._sha256(
            json.dumps(
                {
                    "name": self.name,
                    "model_id": self.model_id,
                    "variant": variant,
                    "text": text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        readable_name = self._safe_part(self.name)
        readable_variant = self._safe_part(variant)
        return f"{self.key_prefix}:{readable_name}:{model_hash}:{readable_variant}:{text_hash}"

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
    def _normalize_embeddings_for_store(
        text_count: int,
        embedding: EmbeddingArray,
    ) -> EmbeddingArray:
        """把待存储的 embedding 统一成二维 float32 数组。"""
        arr = np.asarray(embedding, dtype=np.float32)

        # 单条文本 + 一维向量时，转成形状为 (1, dim) 的二维数组。
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

        return np.ascontiguousarray(arr, dtype=np.float32)

    @staticmethod
    def _serialize_embedding(embedding: EmbeddingArray) -> bytes:
        """
        序列化单条 embedding。

        使用 JSON + base64，而不是直接存裸 bytes：
        - 可记录 dtype、shape、创建时间，便于排查问题；
        - 后续可以平滑扩展元数据；
        - 反序列化时能做一致性检查。
        """
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("单条 embedding 必须是一维数组")

        payload = {
            "dtype": "float32",
            "shape": list(arr.shape),
            "created_at": int(time.time()),
            "embedding_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _deserialize_embedding(value: Optional[bytes]) -> Optional[EmbeddingArray]:
        """
        反序列化单条 embedding。

        """
        if value is None:
            return None

        try:
            payload = json.loads(value.decode("utf-8"))
            if payload.get("dtype") != "float32":
                raise ValueError(f"不支持的 embedding dtype: {payload.get('dtype')}")

            raw = base64.b64decode(payload["embedding_b64"])
            shape = tuple(payload["shape"])
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            return arr.copy()
        except Exception as exc:
            raise ValueError("反序列化 embedding 失败") from exc


    def store(
        self,
        text: TextInput,
        embedding: EmbeddingArray,
        variant: str = "document",
    ) -> List[bool]:
        """
        存储 embedding。

        Parameters
        ----------
        text:
            单条文本或文本列表。
        embedding:
            单条 embedding 或批量 embedding。
        variant:
            向量用途，默认 document。

        Returns
        -------
        List[bool]
            Redis pipeline 每条 set/setex 命令的执行结果。
        """
        texts, _ = self._ensure_text_list(text)
        embeddings = self._normalize_embeddings_for_store(len(texts), embedding)

        try:
            # transaction=false表示不启用事物，Redis 事务的核心保证是：事务里的命令会被顺序执行，执行过程中不会被其他客户端的命令插进来。
            with self.redis.pipeline(transaction=False) as pipe:
                for item_text, item_embedding in zip(texts, embeddings):
                    key = self._key(item_text, variant=variant)
                    value = self._serialize_embedding(item_embedding)

                    if self.ttl and self.ttl > 0:
                        pipe.setex(key, int(self.ttl), value)
                    else:
                        pipe.set(key, value)

                return list(pipe.execute())
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Store embedding failed", exc, default=[])

    # set 是 store 的别名，更符合缓存组件的直觉用法。
    set = store

    def call(
        self,
        text: TextInput,
        variant: str = "document",
    ) -> Union[Optional[EmbeddingArray], List[Optional[EmbeddingArray]]]:
        """
        读取 embedding。

        """
        texts, single_input = self._ensure_text_list(text)

        try:
            keys = [self._key(item_text, variant=variant) for item_text in texts]
            values = self.redis.mget(keys)  # muti get
            embeddings = [self._deserialize_embedding(value) for value in values]
            return embeddings[0] if single_input else embeddings
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Get embedding failed", exc, default=None if single_input else [])

    # get 是 call 的别名，更符合缓存组件的直觉用法。
    get = call

    def delete(self, text: TextInput, variant: str = "document") -> int:
        """删除指定文本对应的缓存。"""
        texts, _ = self._ensure_text_list(text)
        if not texts:
            return 0

        try:
            keys = [self._key(item_text, variant=variant) for item_text in texts]
            return int(self.redis.delete(*keys))
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Delete embedding failed", exc, default=0)

    def get_or_embed(
        self,
        text: TextInput,
        embed_fn: Callable[[List[str]], EmbeddingArray],
        variant: str = "document",
        return_cache_stats: bool = False,
    ):
        """
        缓存优先读取；未命中时调用 embed_fn 生成并写回缓存。

        """
        texts, single_input = self._ensure_text_list(text)
        if not texts:
            raise ValueError("text 不能为空")

        cached = self.get(texts, variant=variant)
        assert isinstance(cached, list)

        result: List[Optional[EmbeddingArray]] = list(cached)
        miss_indices = [idx for idx, item in enumerate(result) if item is None]

        if miss_indices:
            miss_texts = [texts[idx] for idx in miss_indices]
            new_embeddings = self._normalize_embeddings_for_store(
                len(miss_texts),
                embed_fn(miss_texts),
            )
            self.store(miss_texts, new_embeddings, variant=variant)

            for idx, embedding in zip(miss_indices, new_embeddings):
                result[idx] = embedding

        # 到这里所有 result 都应该有值。
        final_embeddings = np.vstack([item for item in result if item is not None]).astype(np.float32)
        stats = CacheStats(
            total=len(texts),
            hit=len(texts) - len(miss_indices),
            miss=len(miss_indices),
        )

        output = final_embeddings[0] if single_input else final_embeddings
        return (output, stats) if return_cache_stats else output


class SentenceBertEmbedder:
    """
    本地 SentenceTransformer / BGE 模型封装。

    这个类只负责“把文本转为 embedding”，不关心缓存。
    与 EmbeddingsCache 组合后，就可以成为可复用的底层 embedding 服务。
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

        BGE 类模型做检索时，query 侧通常可以加中文检索指令；# https://huggingface.co/BAAI/bge-small-zh-v1.5
        document 侧通常直接使用原文。
        """
        if is_query and self.query_instruction:
            return [self.query_instruction + item for item in texts]
        return texts

    def encode(self, text: TextInput, is_query: bool = False) -> EmbeddingArray:
        """生成 embedding。单条输入返回一维数组，批量输入返回二维数组。"""
        texts, single_input = EmbeddingsCache._ensure_text_list(text)
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


class CachedEmbeddingService:
    """
    面向 AI 应用的高层 embedding 服务。

    用法：
        service.embed("文本")
        service.embed(["文本1", "文本2"])

    内部流程：
        1. 先查 Redis 缓存；
        2. 缓存命中的直接返回；
        3. 缓存未命中的调用本地模型生成；
        4. 新生成的向量写回 Redis；
        5. 按输入顺序返回 embedding。
    """

    def __init__(
        self,
        cache: EmbeddingsCache,
        embedder: SentenceBertEmbedder,
    ) -> None:
        self.cache = cache
        self.embedder = embedder

    def embed(
        self,
        text: TextInput,
        is_query: bool = False,
        use_cache: bool = True,
        return_cache_stats: bool = False,
    ):
        """
        获取 embedding。

        Parameters
        ----------
        text:
            单条文本或文本列表。
        is_query:
            True 表示查询向量；False 表示文档向量。
            query 和 document 会使用不同缓存 variant，避免混用。
        use_cache:
            是否启用 Redis 缓存。
        return_cache_stats:
            是否同时返回缓存统计信息。
        """
        texts, single_input = EmbeddingsCache._ensure_text_list(text)
        if not texts:
            raise ValueError("text 不能为空")

        variant = "query" if is_query else "document"

        if not use_cache:
            embeddings = self.embedder.encode(texts, is_query=is_query)
            if single_input and embeddings.ndim == 2:
                embeddings = embeddings[0]

            stats = CacheStats(total=len(texts), hit=0, miss=len(texts))
            return (embeddings, stats) if return_cache_stats else embeddings

        return self.cache.get_or_embed(
            text=text,
            # get_or_embed()会将missing_texts传给lambda函数
            embed_fn=lambda missing_texts: self.embedder.encode(missing_texts, is_query=is_query),
            variant=variant,
            return_cache_stats=return_cache_stats,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # 你指定的本地 BGE/Sentence-BERT 模型路径。
    # 注意：这里使用 raw string，避免 Windows 路径中的反斜杠被当作转义字符。
    MODEL_PATH = r"D:\桌面\typora文件\八斗AI\models\sentence_bert\bge-small-zh-v1.5"

    # 1. 初始化 Redis 缓存。
    embedding_cache = EmbeddingsCache(
        name="demo_embedding_cache",
        model_id=MODEL_PATH,
        ttl=3600 * 24 * 7,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        raise_on_error=True,
    )

    # 可选：提前检查 Redis 是否可用。
    embedding_cache.ping()

    # 2. 初始化本地模型。加载本机的 bge-small-zh-v1.5 模型。
    embedder = SentenceBertEmbedder(
        model_path=MODEL_PATH,
        device=None,  # 可改成 "cuda" 或 "cpu"；None 表示由 sentence-transformers 自动选择。
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # 3. 组合成可复用的底层 embedding 服务。
    embedding_service = CachedEmbeddingService(
        cache=embedding_cache,
        embedder=embedder,
    )

    test_texts = [
        "你好，世界。",
        "这是一个用于测试 embedding 缓存的句子。",
        "Embedding 缓存可以减少重复计算，提升 AI 应用性能。",
    ]

    # 第一次调用：通常全部缓存未命中，会调用本地模型生成 embedding 并写入 Redis。
    embeddings, stats = embedding_service.embed(
        test_texts,
        is_query=False,
        return_cache_stats=True,
    )
    print("第一次调用：", stats)
    print("embedding shape:", embeddings.shape)
    print("第一条向量前 5 维:", embeddings[0][:5])

    # 第二次调用：同样的文本和模型，应该优先从 Redis 命中。
    cached_embeddings, cached_stats = embedding_service.embed(
        test_texts,
        is_query=False,
        return_cache_stats=True,
    )
    print("第二次调用：", cached_stats)
    print("缓存向量是否一致:", np.allclose(embeddings, cached_embeddings))

    # 单条 query 向量测试。query 会使用独立缓存 variant，不会和 document 缓存混用。
    query_text = "如何提升 AI 应用中的 embedding 计算性能？"
    query_embedding, query_stats = embedding_service.embed(
        query_text,
        is_query=True,
        return_cache_stats=True,
    )
    print("query 调用：", query_stats)
    print("query embedding shape:", query_embedding.shape)


    # 删除缓存示例。
    deleted_count = embedding_cache.delete(test_texts, variant="document")
    deleted_count+= embedding_cache.delete(query_text, variant="query")
    print("删除 document 缓存数量:", deleted_count)
