"""
Week 17 homework: RedisVL SemanticRouter demo.

The router classifies a user statement into the best route by comparing the
statement embedding with route reference embeddings stored in Redis.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, List


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
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
        from redisvl.extensions.router import Route, RoutingConfig, SemanticRouter
    except ImportError:
        print("Please install RedisVL first: pip install redisvl")
        return

    routes = [
        Route(
            name="rag_search",
            references=[
                "search the knowledge base",
                "find documents about redis vector search",
                "retrieve relevant chunks for a question",
            ],
            metadata={"tool": "retriever"},
            distance_threshold=0.8,
        ),
        Route(
            name="llm_cache",
            references=[
                "reuse an existing answer",
                "check semantic cache before calling the model",
                "avoid repeated LLM requests",
            ],
            metadata={"tool": "semantic_cache"},
            distance_threshold=0.8,
        ),
        Route(
            name="chat_memory",
            references=[
                "remember previous conversation context",
                "retrieve useful message history",
                "what did the user ask before",
            ],
            metadata={"tool": "message_history"},
            distance_threshold=0.8,
        ),
    ]

    router = SemanticRouter(
        name="zyy_week17_router",
        routes=routes,
        redis_url=REDIS_URL,
        vectorizer=build_vectorizer(),
        routing_config=RoutingConfig(max_k=2),
        overwrite=True,
    )

    try:
        statements = [
            "Can we answer this from cached responses?",
            "Look up RedisVL docs and retrieve useful chunks.",
            "Use earlier conversation context before replying.",
        ]

        print(f"available routes: {router.route_names}")
        for statement in statements:
            print("-" * 60)
            print(f"statement: {statement}")
            matches = router.route_many(statement=statement, max_k=2)
            if not matches:
                print("no route matched")
                continue
            for match in matches:
                print(f"route: {match.name}, distance: {match.distance}")

        print("-" * 60)
        added = router.add_route_references(
            "rag_search",
            ["use retrieval augmented generation for this question"],
        )
        print(f"added references: {len(added)}")
        print(f"rag_search references: {len(router.get_route_references('rag_search'))}")
    except Exception as exc:
        print("RedisVL semantic router demo failed.")
        print("Make sure Redis is running on REDIS_URL and supports vector search.")
        print(f"error: {exc}")
    finally:
        try:
            router.delete()
        except Exception:
            pass


if __name__ == "__main__":
    main()
