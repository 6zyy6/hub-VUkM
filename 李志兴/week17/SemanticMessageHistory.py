r"""
SemanticMessageHistory.py

一个可复用的 Redis 对话历史管理工具。

核心能力：
1. 使用 Redis 保存多轮对话历史。
2. 支持 session 级别隔离，不同 name 对应不同会话。
3. 支持 TTL 自动过期。
4. 支持单条消息与批量消息写入。
5. 支持按角色读取最近消息。
6. 支持基于关键词与 Levenshtein 字符串相似度的相关历史检索。
7. 支持 Redis 异常统一处理，方便测试与工程集成。
8. 支持外部传入 redis_client，便于复用连接池或单元测试。

依赖安装：
    pip install redis python-Levenshtein

说明：
    当前 get_relevant() 是“字符串相关检索”，不是严格意义上的 embedding 语义检索。
    如果需要真正的语义检索，应结合 EmbeddingsCache、Sentence-BERT/BGE 和向量索引使用。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


try:
    import Levenshtein
except ImportError:  # pragma: no cover - 取决于外部依赖
    Levenshtein = None


ChatMessage = Dict[str, Any]
MessageInput = Union[ChatMessage, Sequence[ChatMessage]]
RoleInput = Optional[Union[str, Sequence[str]]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoryStats:
    """一次对话历史读取或检索的统计信息。"""

    total: int
    selected: int

    @property
    def selected_rate(self) -> float:
        """被选中消息占总消息数的比例。"""
        if self.total == 0:
            return 0.0
        return self.selected / self.total


@dataclass(frozen=True)
class RelevantMessage:
    """相关历史检索结果。"""

    index: int
    score: float
    message: ChatMessage


class SemanticMessageHistory:
    """
    基于 Redis 的对话历史管理组件。

    这个类只负责“存储、读取、裁剪、清空、相关检索对话历史”，不负责生成 embedding。
    如果需要真正的语义记忆，可以与 EmbeddingsCache、FAISS、Redis Vector 等组件组合。

    Parameters
    ----------
    name:
        会话命名空间，类似 session id，例如："user_001"、"my-session"。
    ttl:
        对话历史过期时间，单位秒。None 或 0 表示不过期。
    redis_host / redis_port / redis_db / redis_password:
        Redis 连接信息。
    redis_client:
        可选的 Redis 兼容客户端，方便测试或接入已有连接池。
    key_prefix:
        Redis key 前缀。
    auto_add_created_at:
        是否在写入消息时自动补充 created_at 字段。
    raise_on_error:
        True 时 Redis 异常直接抛出；False 时记录日志并返回安全默认值。
    """

    def __init__(
        self,
        name: str,
        ttl: Optional[int] = 3600 * 24,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        redis_client: Optional[object] = None,
        key_prefix: str = "semantic_history",
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        auto_add_created_at: bool = False,
        raise_on_error: bool = True,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("name 不能为空")

        self.name = name.strip()
        self.ttl = ttl
        self.key_prefix = key_prefix.strip() or "semantic_history"
        self.auto_add_created_at = auto_add_created_at
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

            # decode_responses=False，保持与 EmbeddingsCache.py 风格一致。
            # 这里虽然存 JSON，但用 bytes 更稳，避免 Redis 客户端自动解码带来的兼容问题。
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

    @staticmethod
    def _safe_part(value: str, max_len: int = 48) -> str:
        """把 Redis key 的可读部分限制在安全字符范围内。"""
        safe = []
        for ch in value:
            if ch.isalnum() or ch in {"_", "-"}:
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)[:max_len] or "default"

    def _key(self) -> str:
        """生成当前会话的 Redis key。"""
        readable_name = self._safe_part(self.name)
        return f"{self.key_prefix}:{readable_name}:messages"

    @staticmethod
    def _ensure_message_list(message: MessageInput) -> List[ChatMessage]:
        """
        统一把输入消息转为 List[Dict[str, Any]]。

        支持：
            {"role": "user", "content": "你好"}
        或：
            [
                {"role": "user", "content": "你好"},
                {"role": "llm", "content": "你好，我是大模型。"}
            ]
        """
        if isinstance(message, dict):
            messages = [message]
        else:
            messages = list(message)

        if not messages:
            raise ValueError("message 不能为空")

        normalized_messages: List[ChatMessage] = []

        for item in messages:
            if not isinstance(item, dict):
                raise TypeError("message 中的每一项都必须是 dict")

            role = item.get("role")
            content = item.get("content")

            if not isinstance(role, str) or not role.strip():
                raise ValueError("每条 message 必须包含非空字符串字段 role")

            if not isinstance(content, str):
                raise ValueError("每条 message 必须包含字符串字段 content")

            normalized_messages.append(dict(item))

        return normalized_messages

    @staticmethod
    def _normalize_roles(role: RoleInput) -> Optional[Set[str]]:
        """统一处理 role 参数，支持单个角色或多个角色。"""
        if role is None:
            return None

        if isinstance(role, str):
            if not role.strip():
                return None
            return {role.strip()}

        roles = {str(item).strip() for item in role if str(item).strip()}
        return roles or None

    @staticmethod
    def _serialize_history(messages: List[ChatMessage]) -> bytes:
        """
        序列化对话历史。

        使用带 version 的 JSON 结构，而不是直接存 list：
        - 方便以后扩展元数据；
        - 方便排查创建时间、更新时间；
        - 兼容性更好。
        """
        payload = {
            "version": 1,
            "updated_at": int(time.time()),
            "messages": messages,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _deserialize_history(value: Optional[bytes]) -> List[ChatMessage]:
        """
        反序列化对话历史。

        同时兼容两种格式：
        1. 新格式：{"version": 1, "messages": [...]}
        2. 旧格式：[{"role": "...", "content": "..."}]
        """
        if value is None:
            return []

        try:
            if isinstance(value, bytes):
                payload = json.loads(value.decode("utf-8"))
            else:
                payload = json.loads(value)

            if isinstance(payload, list):
                return payload

            if isinstance(payload, dict):
                messages = payload.get("messages", [])
                if not isinstance(messages, list):
                    raise ValueError("messages 字段必须是 list")
                return messages

            raise ValueError("Redis 中的 history 格式不合法")
        except Exception as exc:
            raise ValueError("反序列化对话历史失败") from exc

    def _store_history(self, messages: List[ChatMessage]) -> bool:
        """把完整历史写回 Redis。"""
        key = self._key()
        value = self._serialize_history(messages)

        if self.ttl and self.ttl > 0:
            return bool(self.redis.setex(key, int(self.ttl), value))

        return bool(self.redis.set(key, value))

    def add(self, message: MessageInput) -> int:
        """
        添加单条或多条消息。

        Returns
        -------
        int
            实际添加的消息数量。
        """
        messages = self._ensure_message_list(message)

        if self.auto_add_created_at:
            now = int(time.time())
            for item in messages:
                item.setdefault("created_at", now)

        try:
            history = self.get_history()
            history.extend(messages)
            self._store_history(history)
            return len(messages)
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Add message failed", exc, default=0)

    # 兼容原来的方法名。
    add_message = add

    def get_history(self) -> List[ChatMessage]:
        """读取完整对话历史。"""
        try:
            value = self.redis.get(self._key())
            return self._deserialize_history(value)
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Get history failed", exc, default=[])

    # get / call 是 get_history 的别名，更符合缓存组件的直觉用法。
    get = get_history
    call = get_history

    def count(self) -> int:
        """返回当前会话的消息数量。"""
        return len(self.get_history())

    def get_recent(
        self,
        role: RoleInput = None,
        top_k: Optional[int] = 10,
        return_stats: bool = False,
    ):
        """
        获取最近若干条消息。

        Parameters
        ----------
        role:
            可选。支持单个角色或多个角色。
            例如：
                role="user"
                role=["user", "llm"]
        top_k:
            返回最近多少条。None 表示不限制。
        return_stats:
            是否返回统计信息。
        """
        history = self.get_history()
        roles = self._normalize_roles(role)

        if roles is None:
            selected = history
        else:
            selected = [
                item for item in history
                if item.get("role", "") in roles
            ]

        if top_k is not None:
            if top_k <= 0:
                selected = []
            else:
                selected = selected[-top_k:]

        stats = HistoryStats(total=len(history), selected=len(selected))
        return (selected, stats) if return_stats else selected

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        """计算两个字符串的相似度。优先使用 Levenshtein，缺失时退化为 SequenceMatcher。"""
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0

        if Levenshtein is not None:
            return float(Levenshtein.ratio(left, right))

        return float(SequenceMatcher(None, left, right).ratio())

    def get_relevant(
        self,
        content: str,
        top_k: Optional[int] = 10,
        role: RoleInput = None,
        min_score: float = 0.2,
        substring_bonus: float = 0.35,
        return_score: bool = False,
    ):
        """
        获取与输入内容相关的历史消息。

        注意：
            这里不是 embedding 语义检索，而是：
            1. 字符串相似度；
            2. 子串命中奖励；
            3. 按得分排序。

        Parameters
        ----------
        content:
            查询内容。
        top_k:
            返回前多少条。None 表示不限制。
        role:
            可选角色过滤。
        min_score:
            最低相关性得分，小于该值的结果会被过滤。
        substring_bonus:
            当查询内容是历史内容子串时，额外增加的得分。
        return_score:
            True 时返回 RelevantMessage，False 时只返回 message。
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content 必须是非空字符串")

        history = self.get_history()
        roles = self._normalize_roles(role)
        query = content.strip().lower()

        results: List[RelevantMessage] = []

        for index, message in enumerate(history):
            if roles is not None and message.get("role", "") not in roles:
                continue

            message_content = message.get("content", "")
            if not isinstance(message_content, str):
                continue

            text = message_content.lower()
            score = self._similarity(text, query)

            if query in text:
                score = min(1.0, score + substring_bonus)

            if score >= min_score:
                results.append(
                    RelevantMessage(
                        index=index,
                        score=score,
                        message=message,
                    )
                )

        results.sort(key=lambda item: (item.score, item.index), reverse=True)

        if top_k is not None:
            if top_k <= 0:
                results = []
            else:
                results = results[:top_k]

        if return_score:
            return results

        return [item.message for item in results]

    def truncate(self, keep_last: int = 10) -> int:
        """
        裁剪历史，只保留最近 keep_last 条。

        Returns
        -------
        int
            被删除的历史消息数量。
        """
        if keep_last < 0:
            raise ValueError("keep_last 不能小于 0")

        try:
            history = self.get_history()
            original_count = len(history)

            if keep_last == 0:
                self.clear()
                return original_count

            new_history = history[-keep_last:]
            self._store_history(new_history)

            return original_count - len(new_history)
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Truncate history failed", exc, default=0)

    # 兼容原来的 delete_history 命名。
    # 但语义上它不是“清空历史”，而是“只保留最近 top_k 条”。
    def delete_history(self, top_k: int = 10) -> int:
        """兼容旧接口：只保留最近 top_k 条历史。"""
        return self.truncate(keep_last=top_k)

    def clear(self) -> int:
        """清空当前会话历史。"""
        try:
            return int(self.redis.delete(self._key()))
        except Exception as exc:  # pragma: no cover - 取决于外部 Redis 服务
            return self._handle_error("Clear history failed", exc, default=0)

    # 兼容原来的 clear_history 方法名。
    clear_history = clear


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    history = SemanticMessageHistory(
        name="my-session",
        ttl=3600 * 24,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        auto_add_created_at=True,
        raise_on_error=True,
    )

    # 可选：检查 Redis 是否可用。
    history.ping()

    # 清空旧历史，避免影响测试。
    history.clear()

    # 1. 添加多条消息。
    history.add([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "what is the weather going to be today?"},
    ])

    # 2. 添加单条消息。
    history.add({
        "role": "llm",
        "content": "You can check a weather API for real-time weather.",
        "metadata": {"model": "gpt-4"},
    })

    print("消息总数:", history.count())

    print("\n完整历史:")
    print(history.get_history())

    print("\n最近 1 条消息:")
    print(history.get_recent(top_k=1))

    print("\n最近 2 条 user 消息:")
    print(history.get_recent(role="user", top_k=2))

    print("\n最近 3 条 user 或 llm 消息:")
    print(history.get_recent(role=["user", "llm"], top_k=3))

    print("\n相关历史：today")
    print(history.get_relevant("today", top_k=2))

    print("\n相关历史：thanks，带分数")
    relevant_results = history.get_relevant("thanks", top_k=2, return_score=True)
    for item in relevant_results:
        print(item)

    print("\n只保留最近 3 条，删除数量:")
    print(history.truncate(keep_last=3))

    print("\n裁剪后的历史:")
    print(history.get_history())

    print("\n清空历史:")
    print(history.clear())