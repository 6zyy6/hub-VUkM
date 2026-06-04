import json
import numpy as np
import redis
from typing import Optional, Union, List, Dict, Any
import Levenshtein


class SemanticMessageHistory:
    def __init__(
            self,
            name: str,
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
    ):
        """
        对话历史管理 - 存储和检索对话历史，支持按角色筛选和语义检索

        Args:
            name: 对话名称（类似session id）
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
        self._history_key = f"semantic_history:{name}"

    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取完整的对话历史

        Returns:
            对话历史列表
        """
        history = self.redis.get(self._history_key)
        if not history:
            return []
        else:
            if isinstance(history, bytes):
                return json.loads(history.decode())
            return json.loads(history)

    def add_message(self, message: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        添加消息到对话历史

        Args:
            message: 消息字典或消息列表
        """
        history = self.get_history()

        if isinstance(message, dict):
            message = [message]

        history.extend(message)

        history_str = json.dumps(history, ensure_ascii=False)
        if isinstance(history_str, str):
            history_str = history_str.encode()

        self.redis.setex(self._history_key, self.ttl, history_str)

    def add_messages(self, messages: List[Dict[str, Any]]):
        """添加多条消息（add_message的别名）"""
        self.add_message(messages)

    def get_recent(self, role: Optional[Union[str, List[str]]] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的消息

        Args:
            role: 角色过滤（可选），支持单个角色或角色列表
            top_k: 返回最近k条消息

        Returns:
            筛选后的消息列表
        """
        history = self.get_history()

        if role:
            if isinstance(role, str):
                role = [role]
            history = [msg for msg in history if msg.get("role", "") in role]

        if top_k:
            history = history[-top_k:]

        return history

    def get_relevant(self, content: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        根据内容关键词检索相关消息

        Args:
            content: 关键词
            top_k: 返回最多k条

        Returns:
            相关消息列表
        """
        history = self.get_history()

        selected_history = [
            msg for msg in history
            if content.lower() in msg.get("content", "").lower()
        ]

        if not selected_history:
            return []

        selected_history = sorted(
            selected_history,
            key=lambda msg: Levenshtein.ratio(msg.get("content", ""), content),
            reverse=True
        )

        if top_k:
            selected_history = selected_history[:top_k]

        return selected_history

    def delete_history(self, top_k: Optional[int] = None):
        """
        删除最近的消息

        Args:
            top_k: 删除最近k条消息，如果为None则删除全部
        """
        if top_k is None:
            self.clear_history()
            return

        history = self.get_history()
        history = history[:-top_k] if top_k < len(history) else []

        if history:
            history_str = json.dumps(history, ensure_ascii=False)
            if isinstance(history_str, str):
                history_str = history_str.encode()
            self.redis.setex(self._history_key, self.ttl, history_str)
        else:
            self.redis.delete(self._history_key)

    def clear_history(self):
        """清除所有对话历史"""
        self.redis.delete(self._history_key)

    def search_by_embedding(self, query: str, embedding_method, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        根据语义相似度搜索消息（需要embedding_method）

        Args:
            query: 查询文本
            embedding_method: 嵌入方法
            top_k: 返回最多k条
            threshold: 相似度阈值

        Returns:
            语义相关消息列表
        """
        history = self.get_history()
        if not history:
            return []

        contents = [msg.get("content", "") for msg in history]
        if not contents:
            return []

        try:
            query_emb = embedding_method(query)
            content_embs = embedding_method(contents)

            if isinstance(query_emb, list):
                query_emb = np.array(query_emb)
            if isinstance(content_embs, list):
                content_embs = np.array(content_embs)

            similarities = np.dot(content_embs, query_emb) / (
                np.linalg.norm(content_embs, axis=1) * np.linalg.norm(query_emb)
            )

            scored = [(i, sim) for i, sim in enumerate(similarities) if sim >= threshold]
            scored.sort(key=lambda x: x[1], reverse=True)

            return [history[i] for i, _ in scored[:top_k]]
        except Exception as e:
            print(f"Search error: {e}")
            return []


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="localhost",
    )

    history.clear_history()

    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
    ])

    print("get_history:", history.get_history())
    print("get_recent top_k=1:", history.get_recent(top_k=1))
    print("get_recent role=user:", history.get_recent(role="user", top_k=1))
    print("get_relevant 'today':", history.get_relevant("today", top_k=1))
    print("get_relevant 'thanks':", history.get_relevant("thanks", top_k=1))