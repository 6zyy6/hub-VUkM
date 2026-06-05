"""
Week 17 homework: RedisVL SemanticCache demo.

The script uses a deterministic local vectorizer so it does not need an
external embedding API. RedisVL still stores and searches the semantic cache.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Callable, List


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


def fake_llm(prompt: str) -> str:
    answers: dict[str, str] = {
        "what is redisvl?": "RedisVL is a Python client for building AI apps on Redis vector search.",
        "how does semantic cache help?": "It reuses answers for semantically similar prompts, reducing latency and token cost.",
    }
    return answers.get(prompt.lower(), f"Generated answer for: {prompt}")


def read_through(cache: Any, prompt: str, answer_fn: Callable[[str], str]) -> str:
    hits = cache.check(prompt=prompt, num_results=1)
    if hits:
        hit = hits[0]
        print(f"CACHE HIT: {prompt}")
        print(f"  matched prompt: {hit.get('prompt')}")
        print(f"  distance: {hit.get('vector_distance')}")
        return hit.get("response", "")

    print(f"CACHE MISS: {prompt}")
    response = answer_fn(prompt)
    key = cache.store(
        prompt=prompt,
        response=response,
        metadata={"source": "fake_llm", "homework": "week17"},
        filters={"app": "redisvl_demo"},
    )
    print(f"  stored key: {key}")
    return response


def main() -> None:
    try:
        from redisvl.extensions.cache.llm import SemanticCache
    except ImportError:
        print("Please install RedisVL first: pip install redisvl")
        return

    cache = SemanticCache(
        name="zyy_week17_semantic_cache",
        redis_url=REDIS_URL,
        vectorizer=build_vectorizer(),
        distance_threshold=0.45,
        ttl=3600,
        overwrite=True,
    )

    try:
        prompts = [
            "What is RedisVL?",
            "Can you explain RedisVL?",
            "How does semantic cache help?",
            "Why is semantic caching useful?",
        ]
        for prompt in prompts:
            print("-" * 60)
            print(read_through(cache, prompt, fake_llm))

        cache.set_threshold(0.35)
        print("-" * 60)
        print(f"updated distance threshold: {cache.distance_threshold}")
    except Exception as exc:
        print("RedisVL semantic cache demo failed.")
        print("Make sure Redis is running on REDIS_URL and supports vector search.")
        print(f"error: {exc}")
    finally:
        try:
            cache.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
