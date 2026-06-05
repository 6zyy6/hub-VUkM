"""
Week 17 homework: RedisVL SemanticMessageHistory demo.

The demo stores a short conversation, reads recent turns, and retrieves
semantically relevant memory for a new user question.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, List


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SESSION_TAG = "week17-demo-session"
TOKEN_PATTERN = re.compile(r"[a-z0-9_+-]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        chunk = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            tokens.extend(chunk[i : i + 2] for i in range(max(1, len(chunk) - 1)))
        else:
            tokens.append(chunk)
    return tokens or [text.lower().strip() or "<empty>"]


def embed_text(text: str, dims: int = 16) -> List[float]:
    vector = [0.0] * dims
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def build_vectorizer() -> Any:
    try:
        from redisvl.utils.vectorize import CustomVectorizer
    except ImportError:
        from redisvl.utils.vectorize.text import CustomTextVectorizer as CustomVectorizer

    def embed_many(values: list[str], **_: Any) -> list[list[float]]:
        return [embed_text(value) for value in values]

    return CustomVectorizer(embed=embed_text, embed_many=embed_many)


def main() -> None:
    try:
        from redisvl.extensions.message_history import SemanticMessageHistory
    except ImportError:
        print("Please install RedisVL first: pip install redisvl")
        return

    history = SemanticMessageHistory(
        name="zyy_week17_message_history",
        session_tag=SESSION_TAG,
        redis_url=REDIS_URL,
        vectorizer=build_vectorizer(),
        distance_threshold=0.55,
        overwrite=True,
    )

    try:
        history.clear()
        history.store(
            prompt="How can I reduce repeated LLM calls?",
            response="Use SemanticCache to store answers and search by vector similarity.",
        )
        history.store(
            prompt="How can an agent remember useful context?",
            response="Use SemanticMessageHistory to retrieve recent and semantically relevant messages.",
        )
        history.add_message(
            {
                "role": "user",
                "content": "Route billing questions to the finance tool and search questions to RAG.",
            }
        )

        print(f"message count: {history.count(session_tag=SESSION_TAG)}")
        print("-" * 60)
        print("recent history:")
        print(history.get_recent(top_k=4, as_text=True, session_tag=SESSION_TAG))

        question = "Which memory should I use for similar previous chat context?"
        print("-" * 60)
        print(f"relevant to: {question}")
        relevant = history.get_relevant(
            prompt=question,
            top_k=2,
            as_text=True,
            fall_back=True,
            session_tag=SESSION_TAG,
        )
        print(relevant)
    except Exception as exc:
        print("RedisVL semantic message history demo failed.")
        print("Make sure Redis is running on REDIS_URL and supports vector search.")
        print(f"error: {exc}")
    finally:
        try:
            history.delete()
        except Exception:
            pass


if __name__ == "__main__":
    main()
