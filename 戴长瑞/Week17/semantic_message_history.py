"""
SemanticMessageHistory - 语义化对话历史存储与检索

功能：
- 基于向量召回历史消息
- 支持session会话管理
- 支持角色过滤（user/llm/tool/system）
- 支持LLM长记忆
- 可选兼容LangChain ChatMessage接口

依赖：redisvl >= 0.2.0, redis, pydantic

使用示例：
    # 基础用法
    history = SemanticMessageHistory(name="chat_history")
    history.store("Hello", "Hi there!")
    relevant = history.get_relevant("Greetings", top_k=5)
    recent = history.get_recent(top_k=10)

    # 自定义Redis客户端和Embedding模型
    from redis import Redis
    from redisvl.utils.vectorize import OpenAIVectorizer

    client = Redis(host='localhost', port=6379, db=0)
    vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

    history = SemanticMessageHistory(
        name="chat_history",
        redis_client=client,
        vectorizer=vectorizer,
        session_tag="user_123_session",
        distance_threshold=0.2
    )

    # LangChain兼容用法
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    history.add_messages([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi, how can I help you?"),
    ])
"""

from typing import Any, Sequence

from redisvl.extensions.message_history.base_history import BaseMessageHistory
from redisvl.extensions.message_history.schema import ChatMessage
from redisvl.extensions.constants import (
    CONTENT_FIELD_NAME,
    ID_FIELD_NAME,
    MESSAGE_VECTOR_FIELD_NAME,
    METADATA_FIELD_NAME,
    ROLE_FIELD_NAME,
    SESSION_FIELD_NAME,
    TIMESTAMP_FIELD_NAME,
    TOOL_FIELD_NAME,
)
from redisvl.index import SearchIndex
from redisvl.query import CountQuery, FilterQuery, RangeQuery
from redisvl.query.filter import Tag
from redisvl.redis.utils import serialize, validate_vector_dims
from redisvl.utils.log import get_logger
from redisvl.utils.vectorize.base import BaseVectorizer
from redisvl.utils.vectorize.text.huggingface import HFTextVectorizer

logger = get_logger("[SemanticMessageHistory]")


class MessageHistorySchema:
    """消息历史索引schema"""

    @classmethod
    def from_params(cls, name: str, prefix: str, vector_dims: int, dtype: str):
        from redisvl.schema import IndexSchema
        return IndexSchema(
            index={"name": name, "prefix": prefix},
            fields=[
                {"name": ROLE_FIELD_NAME, "type": "tag"},
                {"name": SESSION_FIELD_NAME, "type": "tag"},
                {"name": CONTENT_FIELD_NAME, "type": "text"},
                {"name": TIMESTAMP_FIELD_NAME, "type": "numeric"},
                {"name": TOOL_FIELD_NAME, "type": "text"},
                {"name": METADATA_FIELD_NAME, "type": "text"},
                {
                    "name": MESSAGE_VECTOR_FIELD_NAME,
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


class SemanticMessageHistory(BaseMessageHistory):
    """语义化对话历史 - 基于向量检索历史消息

    支持：
    - 语义搜索相关消息
    - 最近消息检索
    - Session会话管理
    - 角色过滤
    - LangChain消息接口兼容（可选）
    """

    # LangChain消息类型映射
    LANGCHAIN_TYPE_MAP = {
        "human": "user",
        "ai": "llm",
        "system": "system",
        "tool": "tool",
    }

    def __init__(
        self,
        name: str = "msg_history",
        session_tag: str | None = None,
        prefix: str | None = None,
        vectorizer: BaseVectorizer | None = None,
        distance_threshold: float = 0.3,
        redis_client=None,
        redis_url: str = "redis://localhost:6379",
        connection_kwargs: dict[str, Any] = {},
        overwrite: bool = False,
        langchain_compatible: bool = False,
        **kwargs,
    ):
        """初始化消息历史

        Args:
            name: 索引名称
            session_tag: 会话标签，用于关联特定对话
            prefix: 键前缀，默认与name相同
            vectorizer: 向量化器，默认HFTextVectorizer
                     支持：OpenAIVectorizer, CohereVectorizer等
            distance_threshold: 相似度阈值，默认0.3
            redis_client: Redis客户端实例（同步）
            redis_url: Redis连接URL
            connection_kwargs: 连接参数
            overwrite: 是否覆盖已有索引
            langchain_compatible: 是否启用LangChain兼容模式
        """
        super().__init__(name, session_tag)

        prefix = prefix or name
        dtype = kwargs.pop("dtype", None)
        self._langchain_compatible = langchain_compatible

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

        self.set_distance_threshold(distance_threshold)

        schema = MessageHistorySchema.from_params(
            name, prefix, self._vectorizer.dims, self._vectorizer.dtype
        )

        self._index = SearchIndex(
            schema=schema,
            redis_client=redis_client,
            redis_url=redis_url,
            connection_kwargs=connection_kwargs or None,
        )

        if not overwrite and self._index.exists():
            existing = SearchIndex.from_existing(name, redis_client=self._index.client)
            if existing.schema.to_dict() != self._index.schema.to_dict():
                raise ValueError(f"索引{name}已存在且schema不匹配，设置overwrite=True可覆盖")

        self._index.create(overwrite=overwrite, drop=False)
        self._default_session_filter = Tag(SESSION_FIELD_NAME) == self._session_tag

    def __repr__(self):
        return f"SemanticMessageHistory(name={self.name!r}, session={self._session_tag!r}, threshold={self.distance_threshold})"

    @property
    def vectorizer(self) -> BaseVectorizer:
        """获取向量化器"""
        return self._vectorizer

    @property
    def distance_threshold(self) -> float:
        """获取相似度阈值"""
        return self._distance_threshold

    @property
    def session_tag(self) -> str | None:
        """获取当前会话标签"""
        return self._session_tag

    def set_distance_threshold(self, threshold: float) -> None:
        """设置相似度阈值"""
        self._distance_threshold = threshold

    def clear(self) -> None:
        """清空消息历史"""
        self._index.clear()

    def delete(self) -> None:
        """删除整个索引"""
        self._index.delete(drop=True)

    def drop(self, id: str | None = None) -> None:
        """删除特定消息

        Args:
            id: 消息ID，None时删除最近一条
        """
        if id is None:
            id = self.get_recent(top_k=1, raw=True)[0][ID_FIELD_NAME]
        self._index.client.delete(self._index.key(id))

    def count(self, session_tag: str | None = None) -> int:
        """统计消息数量

        Args:
            session_tag: 会话标签过滤，None使用默认会话

        Returns:
            消息数量
        """
        query = CountQuery(
            filter_expression=(
                Tag(SESSION_FIELD_NAME) == session_tag
                if session_tag
                else self._default_session_filter
            )
        )
        return self._index.query(query)

    def _validate_roles(self, role: str | list[str] | None) -> list[str] | None:
        """验证角色参数"""
        if role is None:
            return None

        valid_roles = {"system", "user", "llm", "tool"}
        if isinstance(role, str):
            role = [role]

        for r in role:
            if r not in valid_roles:
                raise ValueError(f"无效角色: {r}，有效值: {valid_roles}")

        return role

    def _format_context(self, messages: list[dict], as_text: bool) -> list[str] | list[dict]:
        """格式化消息为上下文"""
        if as_text:
            return [f"{m.get(ROLE_FIELD_NAME, 'unknown')}: {m.get(CONTENT_FIELD_NAME, '')}" for m in messages]
        return messages

    def _convert_langchain_messages(self, messages: list) -> list[dict[str, str]]:
        """将LangChain消息转换为内部格式

        Args:
            messages: LangChain消息列表

        Returns:
            内部格式的消息列表
        """
        converted = []
        for msg in messages:
            # 获取消息类型
            msg_type = msg.type if hasattr(msg, "type") else str(type(msg).__name__).lower()

            # 映射到内部角色
            role = self.LANGCHAIN_TYPE_MAP.get(msg_type, "user")

            # 获取内容
            content = msg.content if hasattr(msg, "content") else str(msg)

            converted.append({
                ROLE_FIELD_NAME: role,
                CONTENT_FIELD_NAME: content,
            })

            # 处理额外属性
            if hasattr(msg, "additional_kwargs"):
                metadata = {k: v for k, v in msg.additional_kwargs.items()}
                if metadata:
                    converted[-1][METADATA_FIELD_NAME] = serialize(metadata)

        return converted

    def messages(
        self,
        session_tag: str | None = None,
        role: str | list[str] | None = None,
        as_text: bool = False,
    ) -> list[str] | list[dict[str, str]]:
        """获取完整消息历史

        Args:
            session_tag: 会话标签过滤
            role: 角色过滤（"user", "llm", "system", "tool"）
            as_text: 是否返回文本格式

        Returns:
            消息列表
        """
        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            TOOL_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        session_filter = (
            Tag(SESSION_FIELD_NAME) == session_tag
            if session_tag
            else self._default_session_filter
        )

        filter_expr = session_filter
        roles = self._validate_roles(role)
        if roles:
            if len(roles) == 1:
                role_filter = Tag(ROLE_FIELD_NAME) == roles[0]
            else:
                role_filters = [Tag(ROLE_FIELD_NAME) == r for r in roles]
                role_filter = role_filters[0]
                for rf in role_filters[1:]:
                    role_filter = role_filter | rf
            filter_expr = session_filter & role_filter

        query = FilterQuery(
            filter_expression=filter_expr,
            return_fields=return_fields,
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=True)
        messages = self._index.query(query)

        return self._format_context(messages, as_text)

    def get_relevant(
        self,
        prompt: str,
        as_text: bool = False,
        top_k: int = 5,
        fall_back: bool = False,
        session_tag: str | None = None,
        raw: bool = False,
        distance_threshold: float | None = None,
        role: str | list[str] | None = None,
    ) -> list[str] | list[dict[str, str]]:
        """语义搜索相关消息

        Args:
            prompt: 查询文本
            as_text: 是否返回文本格式 ["role: content", ...]
            top_k: 返回数量
            fall_back: 无结果时是否回退到最近消息
            session_tag: 会话标签过滤
            raw: 是否返回原始数据
            distance_threshold: 阈值覆盖
            role: 角色过滤（"user", "llm", "system", "tool"）

        Returns:
            相关消息列表

        示例:
            relevant = history.get_relevant("Hello", top_k=5)
            relevant_text = history.get_relevant("Hello", top_k=5, as_text=True)
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k必须是>=0的整数")

        roles = self._validate_roles(role)
        threshold = distance_threshold or self._distance_threshold

        return_fields = [
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            TOOL_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        session_filter = (
            Tag(SESSION_FIELD_NAME) == session_tag
            if session_tag
            else self._default_session_filter
        )

        filter_expr = session_filter
        if roles:
            if len(roles) == 1:
                role_filter = Tag(ROLE_FIELD_NAME) == roles[0]
            else:
                role_filters = [Tag(ROLE_FIELD_NAME) == r for r in roles]
                role_filter = role_filters[0]
                for rf in role_filters[1:]:
                    role_filter = role_filter | rf
            filter_expr = session_filter & role_filter

        query = RangeQuery(
            vector=self._vectorizer.embed(prompt),
            vector_field_name=MESSAGE_VECTOR_FIELD_NAME,
            return_fields=return_fields,
            distance_threshold=threshold,
            num_results=top_k,
            return_score=True,
            filter_expression=filter_expr,
            dtype=self._vectorizer.dtype,
        )

        messages = self._index.query(query)

        if not messages and fall_back:
            return self.get_recent(as_text=as_text, top_k=top_k, raw=raw, role=role)

        if raw:
            return messages

        return self._format_context(messages, as_text)

    def get_recent(
        self,
        top_k: int = 5,
        as_text: bool = False,
        raw: bool = False,
        session_tag: str | None = None,
        role: str | list[str] | None = None,
    ) -> list[str] | list[dict[str, str]]:
        """获取最近消息

        Args:
            top_k: 返回数量
            as_text: 是否返回文本格式
            raw: 是否返回原始数据
            session_tag: 会话标签过滤
            role: 角色过滤

        Returns:
            最近消息列表
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k必须是>=0的整数")

        roles = self._validate_roles(role)

        return_fields = [
            ID_FIELD_NAME,
            SESSION_FIELD_NAME,
            ROLE_FIELD_NAME,
            CONTENT_FIELD_NAME,
            TIMESTAMP_FIELD_NAME,
            TOOL_FIELD_NAME,
            METADATA_FIELD_NAME,
        ]

        session_filter = (
            Tag(SESSION_FIELD_NAME) == session_tag
            if session_tag
            else self._default_session_filter
        )

        filter_expr = session_filter
        if roles:
            if len(roles) == 1:
                role_filter = Tag(ROLE_FIELD_NAME) == roles[0]
            else:
                role_filters = [Tag(ROLE_FIELD_NAME) == r for r in roles]
                role_filter = role_filters[0]
                for rf in role_filters[1:]:
                    role_filter = role_filter | rf
            filter_expr = session_filter & role_filter

        query = FilterQuery(
            filter_expression=filter_expr,
            return_fields=return_fields,
            num_results=top_k,
        )
        query.sort_by(TIMESTAMP_FIELD_NAME, asc=False)
        messages = self._index.query(query)

        if raw:
            return messages[::-1]

        return self._format_context(messages[::-1], as_text)

    def store(
        self,
        prompt: str,
        response: str,
        session_tag: str | None = None,
    ) -> None:
        """存储对话消息对（用户prompt + LLM response）

        Args:
            prompt: 用户消息
            response: LLM响应
            session_tag: 会话标签
        """
        self.add_messages(
            [
                {ROLE_FIELD_NAME: "user", CONTENT_FIELD_NAME: prompt},
                {ROLE_FIELD_NAME: "llm", CONTENT_FIELD_NAME: response},
            ],
            session_tag,
        )

    def add_messages(
        self,
        messages: list[dict[str, str]] | list[Any],
        session_tag: str | None = None,
    ) -> None:
        """批量添加消息

        Args:
            messages: 消息列表
                  格式1: [{"role": "user", "content": "..."}, ...]
                  格式2 (LangChain): [HumanMessage, AIMessage, SystemMessage, ...]
            session_tag: 会话标签

        示例:
            # 内部格式
            history.add_messages([
                {"role": "user", "content": "Hello"},
                {"role": "llm", "content": "Hi!"},
            ])

            # LangChain格式
            from langchain.schema import HumanMessage, AIMessage
            history.add_messages([
                HumanMessage(content="Hello"),
                AIMessage(content="Hi!"),
            ])
        """
        # 处理LangChain消息兼容
        if self._langchain_compatible and messages:
            first_msg = messages[0]
            if hasattr(first_msg, "type") or hasattr(first_msg, "content"):
                messages = self._convert_langchain_messages(messages)

        session_tag = session_tag or self._session_tag
        chat_messages = []

        for message in messages:
            content_vector = self._vectorizer.embed(message[CONTENT_FIELD_NAME])
            validate_vector_dims(
                len(content_vector),
                self._index.schema.fields[MESSAGE_VECTOR_FIELD_NAME].attrs.dims,
            )

            chat_message = ChatMessage(
                role=message[ROLE_FIELD_NAME],
                content=message[CONTENT_FIELD_NAME],
                session_tag=session_tag,
                vector_field=content_vector,
            )

            if TOOL_FIELD_NAME in message:
                chat_message.tool_call_id = message[TOOL_FIELD_NAME]
            if METADATA_FIELD_NAME in message:
                meta = message[METADATA_FIELD_NAME]
                if isinstance(meta, dict):
                    chat_message.metadata = serialize(meta)

            chat_messages.append(chat_message.to_dict(dtype=self._vectorizer.dtype))

        self._index.load(data=chat_messages, id_field=ID_FIELD_NAME)

    def add_message(
        self,
        message: dict[str, str] | Any,
        session_tag: str | None = None,
    ) -> None:
        """添加单条消息

        Args:
            message: 消息（内部格式或LangChain格式）
            session_tag: 会话标签
        """
        self.add_messages([message], session_tag)

    # ==================== LangChain兼容接口 ====================

    def add_langchain_messages(self, messages: list) -> None:
        """添加LangChain消息（显式调用）

        Args:
            messages: LangChain消息列表 [HumanMessage, AIMessage, ...]

        示例:
            from langchain.schema import HumanMessage, AIMessage
            history.add_langchain_messages([
                HumanMessage(content="Hello"),
                AIMessage(content="Hi!"),
            ])
        """
        converted = self._convert_langchain_messages(messages)
        self.add_messages(converted)

    def get_langchain_messages(
        self,
        session_tag: str | None = None,
        role: str | list[str] | None = None,
    ) -> list:
        """获取LangChain格式的消息

        Args:
            session_tag: 会话标签
            role: 角色过滤

        Returns:
            LangChain消息列表

        示例:
            from langchain.schema import HumanMessage, AIMessage
            lc_messages = history.get_langchain_messages()
            for msg in lc_messages:
                print(f"{msg.type}: {msg.content}")
        """
        try:
            from langchain.schema import BaseMessage
            has_langchain = True
        except ImportError:
            has_langchain = False

        if not has_langchain:
            logger.warning("LangChain未安装，请运行: pip install langchain")
            return []

        messages = self.messages(session_tag=session_tag, role=role, as_text=False)

        role_to_lc = {v: k for k, v in self.LANGCHAIN_TYPE_MAP.items()}

        lc_messages = []
        for msg in messages:
            lc_type = role_to_lc.get(msg.get(ROLE_FIELD_NAME), "human")
            content = msg.get(CONTENT_FIELD_NAME, "")

            # 创建对应类型的LangChain消息
            if lc_type == "human":
                lc_messages.append({"type": "human", "content": content})
            elif lc_type == "ai":
                lc_messages.append({"type": "ai", "content": content})
            elif lc_type == "system":
                lc_messages.append({"type": "system", "content": content})
            elif lc_type == "tool":
                lc_messages.append({"type": "tool", "content": content})

        return lc_messages

    def clear_session(self, session_tag: str | None = None) -> None:
        """清空特定会话的消息

        Args:
            session_tag: 会话标签，None使用当前会话
        """
        tag = session_tag or self._session_tag
        if tag:
            from redisvl.query import FilterQuery
            from redisvl.query.filter import Tag

            query = FilterQuery(
                filter_expression=Tag(SESSION_FIELD_NAME) == tag,
                return_fields=[ID_FIELD_NAME],
            )
            results = self._index.query(query)

            for msg in results:
                self._index.client.delete(self._index.key(msg[ID_FIELD_NAME]))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


# ==================== 使用示例 ====================
"""
# 示例1: 基础用法
history = SemanticMessageHistory(name="chat_history")
history.store("Hello", "Hi there!")
history.store("How are you?", "I'm fine, thank you!")

# 语义检索
relevant = history.get_relevant("Greetings", top_k=5)
print(relevant)

# 最近消息
recent = history.get_recent(top_k=10)
print(recent)

# 示例2: 自定义Redis客户端和Embedding模型
from redis import Redis
from redisvl.utils.vectorize import OpenAIVectorizer

client = Redis(host='localhost', port=6379, db=0)
vectorizer = OpenAIVectorizer(model="text-embedding-3-small")

history = SemanticMessageHistory(
    name="chat_history",
    redis_client=client,
    vectorizer=vectorizer,
    session_tag="user_123_session",
    distance_threshold=0.2
)

# 示例3: 角色过滤
history.store("Tell me about Python", "Python is a programming language.")
history.store("What is Java?", "Java is another programming language.")

# 只获取用户消息
user_msgs = history.get_recent(role="user")
# 只获取AI回复
ai_msgs = history.get_recent(role="llm")
# 获取特定角色
specific = history.get_relevant("programming", role=["user", "llm"])

# 示例4: 多会话管理
history1 = SemanticMessageHistory(name="session_1", session_tag="user_1")
history2 = SemanticMessageHistory(name="session_2", session_tag="user_2")

history1.store("Message from user 1", "Response to user 1")
history2.store("Message from user 2", "Response to user 2")

# 获取特定会话消息
msgs = history1.messages()
msgs2 = history2.messages()

# 示例5: LangChain兼容用法
from langchain.schema import HumanMessage, AIMessage, SystemMessage

history = SemanticMessageHistory(
    name="chat_history",
    langchain_compatible=True  # 启用LangChain兼容
)

# 添加LangChain消息
history.add_langchain_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi, how can I help you?"),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language."),
])

# 获取LangChain格式消息
lc_msgs = history.get_langchain_messages()
for msg in lc_msgs:
    print(f"{msg['type']}: {msg['content']}")

# 示例6: 清空会话
history.clear_session()  # 清空当前会话
history.clear_session("user_123")  # 清空指定会话

# 示例7: 消息统计
count = history.count()
print(f"Total messages: {count}")

# 示例8: 上下文管理
with SemanticMessageHistory(name="temp_history") as history:
    history.store("Hello", "Hi!")
    # 自动断开连接

# 示例9: 回退到最近消息
relevant = history.get_relevant(
    "完全不相关的话题",
    top_k=3,
    fall_back=True  # 无语义匹配时回退
)
# 如果没有语义匹配，会返回最近的3条消息

# 示例10: 原始数据获取
raw_msgs = history.get_recent(top_k=5, raw=True)
for msg in raw_msgs:
    print(f"ID: {msg['id']}, Role: {msg['role']}, Content: {msg['content']}")
"""