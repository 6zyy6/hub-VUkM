"""
SemanticMessageHistory — 基于 RedisVL SemanticSessionManager 的对话历史管理

对照课程原始实现：
- 原始用 Redis String (JSON) 存储历史 + Levenshtein 编辑距离排序
- 升级为 RedisVL SemanticSessionManager，用向量语义搜索替代编辑距离

功能：
- add_messages(messages)            — 追加对话消息
- get_recent(role, top_k)           — 获取最近 N 条消息（可按角色过滤）
- get_relevant(content, top_k)      — 语义检索最相关的历史消息
- delete_history(top_k)             — 删除最旧消息，保留最近 N 条
- clear_history()                   — 清空全部历史
"""

from typing import Optional, List, Union, Dict, Any
from redisvl.extensions.session_manager import SemanticSessionManager
from redisvl.utils.vectorize import HFTextVectorizer


class SemanticMessageHistory:
    """基于 RedisVL SemanticSessionManager 的对话历史管理。

    核心改进：
    - get_relevant 从 Levenshtein 编辑距离 → 向量语义相似度
    - 历史消息自动向量化存储，无需手动管理
    - 支持异步操作
    """

    def __init__(
        self,
        name: str = "default_session",
        ttl: int = 86400,
        redis_url: str = "redis://localhost:6379",
        redis_password: str = None,
        distance_threshold: float = 0.7,
        vectorizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.name = name
        self.ttl = ttl

        conn_kwargs = {"redis_url": redis_url}
        if redis_password:
            conn_kwargs["connection_args"] = {"password": redis_password}

        self.vectorizer = HFTextVectorizer(model=vectorizer_model)

        self.session = SemanticSessionManager(
            name=name,
            ttl=ttl,
            distance_threshold=distance_threshold,
            vectorizer=self.vectorizer,
            **conn_kwargs,
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """获取全部对话历史。"""
        return self.session.get_recent(top_k=-1)

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
        """追加对话消息。

        Args:
            messages: 消息列表，每条为 {"role": "user/assistant", "content": "..."}
        """
        self.session.add_messages(messages)

    def get_recent(
        self,
        role: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """获取最近 N 条消息。

        Args:
            role:   可选，按角色过滤（"user" / "assistant"）
            top_k:  返回的最大条数，-1 表示全部
        """
        all_messages = self.session.get_recent(top_k=-1)
        if role:
            all_messages = [m for m in all_messages if m.get("role") == role]
        if top_k > 0:
            all_messages = all_messages[-top_k:]
        return all_messages

    def get_relevant(self, content: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """语义检索与 content 最相关的历史消息。

        RedisVL 用向量相似度替代原始实现的 Levenshtein 编辑距离，
        对语义相近但措辞不同的查询更准确。

        Args:
            content: 搜索内容
            top_k:   返回的最大条数
        """
        return self.session.get_relevant(content, top_k=top_k)

    def delete_history(self, top_k: int = 10) -> None:
        """删除旧消息，仅保留最近 top_k 条。"""
        messages = self.get_history()
        if len(messages) > top_k:
            # 逐条删除旧消息
            to_delete = messages[:-top_k]
            for msg in to_delete:
                self.session.delete(message=msg)

    def clear_history(self) -> None:
        """清空全部对话历史。"""
        self.session.clear()


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="test_session",
        redis_url="redis://localhost:6379",
    )
    history.clear_history()

    print("=== SemanticMessageHistory 测试 ===")

    history.add_messages([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "assistant", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "assistant", "content": "I don't know"},
        {"role": "user", "content": "tell me about machine learning"},
    ])

    print(f"get_recent top_k=2: {history.get_recent(top_k=2)}")
    print(f"get_recent role=user top_k=1: {history.get_recent(role='user', top_k=1)}")
    print(f"get_relevant 'weather': {history.get_relevant('weather', top_k=1)}")
    print(f"get_relevant 'AI learning': {history.get_relevant('AI learning', top_k=1)}")

    history.clear_history()
    print("测试完成")
