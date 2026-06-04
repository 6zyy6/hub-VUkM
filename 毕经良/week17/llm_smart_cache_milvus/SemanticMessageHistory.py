"""
SemanticMessageHistory - 对话历史管理 (Milvus版本)
存储和检索对话历史，支持按角色筛选和语义检索
"""

import json
import numpy as np
import redis
from typing import Optional, Union, List, Dict, Any, Callable
import Levenshtein
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


class SemanticMessageHistory:
    def __init__(
            self,
            name: str,
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            milvus_uri: Optional[str] = None,
            milvus_token: Optional[str] = None,
            vector_dimension: Optional[int] = None,
    ):
        """
        对话历史管理 - 使用Milvus做语义搜索，Redis存储消息

        Args:
            name: 对话名称（类似session id）
            ttl: 过期时间（秒）
            redis_url: Redis主机地址
            redis_port: Redis端口
            redis_password: Redis密码
            milvus_uri: Milvus连接URI
            milvus_token: Milvus连接token
            vector_dimension: 向量维度
        """
        self.name = name
        self.ttl = ttl
        self.vector_dimension = vector_dimension

        # Redis客户端
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
            decode_responses=False
        )
        self._history_key = f"semantic_history:{name}"

        # Milvus连接
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self._collection = None
        self._connected = False
        self._embedding_method = None

    def set_embedding_method(self, embedding_method: Callable):
        """设置嵌入方法（用于语义搜索）"""
        self._embedding_method = embedding_method

    def _connect(self):
        """连接Milvus"""
        if self._connected:
            return

        if self.milvus_uri and self.milvus_token:
            alias = f"msg_history_{self.name}"
            connections.connect(
                alias=alias,
                uri=self.milvus_uri,
                token=self.milvus_token
            )
            self._alias = alias
            self._connected = True
        else:
            raise ValueError("Milvus URI and token are required")

    def _ensure_collection(self):
        """确保Collection存在"""
        self._connect()

        collection_name = f"msg_history_{self.name}".replace("-", "_")
        self._alias = f"msg_history_{self.name}"

        if utility.has_collection(collection_name, using=self._alias):
            self._collection = Collection(collection_name, using=self._alias)
            self._collection.load()
        else:
            if self.vector_dimension is None:
                raise ValueError("vector_dimension must be specified for new collection")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
            ]
            schema = CollectionSchema(fields=fields, description=f"Message history for {self.name}")
            self._collection = Collection(name=collection_name, schema=schema, using=self._alias)

            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            self._collection.create_index(field_name="embedding", index_params=index_params)
            self._collection.load()

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

        # 同时存入Milvus（如果提供了embedding方法）
        if self._embedding_method:
            self._add_to_milvus(message)

    def _add_to_milvus(self, messages: List[Dict[str, Any]]):
        """将消息添加到Milvus"""
        contents = [msg.get("content", "") for msg in messages]
        roles = [msg.get("role", "") for msg in messages]

        if self.vector_dimension is None and self._embedding_method:
            emb = self._embedding_method(contents[0] if contents else "")
            self.vector_dimension = len(emb)

        self._ensure_collection()

        try:
            embeddings = self._embedding_method(contents)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)

            data = [contents, roles, embeddings.tolist()]
            self._collection.insert(data)
            self._collection.flush()
        except Exception as e:
            print(f"Milvus insert error: {e}")

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

        # 清除Milvus Collection
        if self._connected:
            collection_name = f"msg_history_{self.name}".replace("-", "_")
            try:
                utility.drop_collection(collection_name, using=self._alias)
            except:
                pass
            self._collection = None

    def search_by_embedding(self, query: str, top_k: int = 5, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """
        根据语义相似度搜索消息

        Args:
            query: 查询文本
            top_k: 返回最多k条
            threshold: 距离阈值

        Returns:
            语义相关消息列表
        """
        if not self._embedding_method:
            return []

        self._ensure_collection()

        try:
            query_emb = self._embedding_method(query)
            if isinstance(query_emb, list):
                query_emb = np.array(query_emb, dtype=np.float32)
            query_emb = query_emb.reshape(1, -1)

            search_params = {"params": {"nprobe": 10}, "metric_type": "L2", "offset": 0}
            results = self._collection.search(
                data=query_emb.tolist(),
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["content", "role"]
            )

            if not results or not results[0]:
                return []

            messages = []
            for result in results[0]:
                if result.distance <= threshold:
                    messages.append({
                        "content": result.entity.get("content", ""),
                        "role": result.entity.get("role", "")
                    })

            return messages

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def close(self):
        """关闭连接"""
        if self._connected:
            connections.disconnect(alias=self._alias)
            self._connected = False


if __name__ == "__main__":
    MILVUS_URI = "https://in03-6fc9fda7586c8a5.serverless.aws-eu-central-1.cloud.zilliz.com"
    MILVUS_TOKEN = "319f97861036cbada2e4af735478028c1dda6e728b875e7d698472763eed54c46927310d70760cad623df9071587e2cb19f48637"

    def get_embedding(text):
        return np.random.rand(128).astype(np.float32)

    history = SemanticMessageHistory(
        name="test-session",
        redis_url="localhost",
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        vector_dimension=128,
    )
    history.set_embedding_method(get_embedding)

    history.clear_history()

    print("添加消息...")
    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather today?"},
        {"role": "llm", "content": "It's sunny."},
    ])

    print("获取完整历史:")
    hist = history.get_history()
    print(f"  history count: {len(hist)}")

    print("获取最近1条:")
    recent = history.get_recent(top_k=1)
    print(f"  recent: {recent}")

    print("获取user角色最近2条:")
    user_msgs = history.get_recent(role="user", top_k=2)
    print(f"  user messages: {user_msgs}")

    print("关键词搜索'thanks':")
    relevant = history.get_relevant("thanks", top_k=1)
    print(f"  relevant: {relevant}")

    history.close()
    print("测试完成!")