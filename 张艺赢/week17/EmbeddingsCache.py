"""
Week 17 homework: RedisVL EmbeddingsCache demo.

Run:
    pip install redisvl
    docker run -d --name redis-vl -p 6379:6379 redis:latest
    python EmbeddingsCache.py
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Iterable, List


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_NAME = "local-hashing-vectorizer-v1"
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


def print_embedding_result(label: str, result: dict | None) -> None:
    if result is None:
        print(f"{label}: MISS")
        return

    content = result.get("content") or result.get("text")
    embedding = result.get("embedding", [])
    metadata = result.get("metadata", {})
    print(f"{label}: HIT")
    print(f"  content: {content}")
    print(f"  dims: {len(embedding)}")
    print(f"  metadata: {metadata}")


def build_items(texts: Iterable[str]) -> list[dict]:
    return [
        {
            "content": text,
            "model_name": MODEL_NAME,
            "embedding": embed_text(text),
            "metadata": {"source": "week17", "vectorizer": "hashing"},
        }
        for text in texts
    ]


def main() -> None:
    try:
        from redisvl.extensions.cache.embeddings import EmbeddingsCache
    except ImportError:
        print("Please install RedisVL first: pip install redisvl")
        return

    cache = EmbeddingsCache(
        name="zyy_week17_embeddings",
        redis_url=REDIS_URL,
        ttl=3600,
    )

    try:
        content = "RedisVL can cache embeddings for repeated text."
        key = cache.set(
            content=content,
            model_name=MODEL_NAME,
            embedding=embed_text(content),
            metadata={"case": "single_set_get"},
        )
        print(f"stored key: {key}")
        print(f"exists: {cache.exists(content, MODEL_NAME)}")
        print_embedding_result("single get", cache.get(content, MODEL_NAME))
        print_embedding_result("get by key", cache.get_by_key(key))

        batch_texts = [
            "semantic cache reduces repeated LLM calls",
            "message history stores useful conversation memory",
            "semantic router sends a question to the right tool",
        ]
        keys = cache.mset(build_items(batch_texts))
        print(f"batch stored: {len(keys)} keys")
        for text, result in zip(batch_texts, cache.mget(batch_texts, MODEL_NAME)):
            print_embedding_result(f"batch get: {text[:28]}", result)

        cache.drop(content, MODEL_NAME)
        print(f"exists after drop: {cache.exists(content, MODEL_NAME)}")
    except Exception as exc:
        print("RedisVL demo failed.")
        print("Make sure Redis is running on REDIS_URL and supports vector search.")
        print(f"error: {exc}")
    finally:
        try:
            cache.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
