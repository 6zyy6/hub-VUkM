"""
LLM Smart Cache — 基于 RedisVL 的智能缓存系统

本项目用 Redis 官方 RedisVL 库重新实现课程中的四个模块，
展示生产级 LLM 缓存系统的完整工作流。

四个模块：
  embeddings_cache.py         — Embedding 向量缓存（避免重复编码）
  semantic_cache.py            — LLM 语义缓存（相似问题命中历史回答）
  semantic_message_history.py  — 对话历史管理（语义检索相关上下文）
  semantic_router.py           — 语义路由（意图识别与分发）

演示流程：
  1. 语义路由 → 识别用户意图
  2. 语义缓存 → 检查是否有相似问题的缓存回答
  3. Embedding缓存 → 避免重复计算 Embedding
  4. 对话历史 → 检索相关历史消息作为上下文
"""

import sys
import os
import time
import hashlib
import numpy as np

# 添加当前目录到 path，方便导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings_cache import EmbeddingsCache
from semantic_cache import SemanticCache
from semantic_message_history import SemanticMessageHistory
from semantic_router import SemanticRouter

# ============================================================
# 配置
# ============================================================
REDIS_URL = "redis://localhost:6379"


def create_demo_router() -> SemanticRouter:
    """创建演示用的语义路由器。"""
    router = SemanticRouter(
        name="demo_router",
        redis_url=REDIS_URL,
        distance_threshold=0.3,
    )
    router.add_route(
        name="greeting",
        references=["Hi", "Hello", "Good morning", "Hey there", "你好"],
        metadata={"handler": "greeting_handler"},
    )
    router.add_route(
        name="weather",
        references=[
            "What's the weather?", "今天天气怎么样",
            "Is it raining?", "会下雨吗",
            "temperature today", "今天多少度",
        ],
        metadata={"handler": "weather_handler"},
    )
    router.add_route(
        name="knowledge",
        references=[
            "What is machine learning?", "什么是深度学习",
            "Explain AI", "how does neural network work",
            "Python tutorial", "编程入门",
        ],
        metadata={"handler": "knowledge_handler"},
    )
    return router


def simulate_llm_call(prompt: str) -> str:
    """模拟 LLM 调用（实际项目中替换为真实 API 调用）。"""
    time.sleep(0.5)  # 模拟网络延迟

    responses = {
        "greeting": "Hello! How can I help you today?",
        "weather": "Today will be sunny with a high of 25°C.",
        "knowledge": "Machine learning is a subset of AI that enables systems to learn from data.",
    }
    for key, resp in responses.items():
        if key in prompt.lower():
            return resp
    return "I'm sorry, I don't have an answer for that."


def demo():
    print("=" * 65)
    print("  LLM Smart Cache — 基于 RedisVL 的智能缓存系统演示")
    print("=" * 65)

    # ---- 初始化四个模块 ----
    print("\n[初始化] 连接 Redis 并初始化四个模块...")

    router = create_demo_router()

    embedding_cache = EmbeddingsCache(
        name="demo_embeddings",
        redis_url=REDIS_URL,
        dims=384,
    )

    llm_cache = SemanticCache(
        name="demo_llm_cache",
        redis_url=REDIS_URL,
        distance_threshold=0.2,
    )

    session = SemanticMessageHistory(
        name="demo_session",
        redis_url=REDIS_URL,
    )

    # 清空旧数据
    llm_cache.clear()
    session.clear_history()
    embedding_cache.clear()

    print("  四个模块初始化完成\n")

    # ---- 模拟对话 ----
    user_queries = [
        "Hello, good morning!",
        "What is machine learning?",
        "Can you explain what ML is?",
        "What's the weather like today?",
        "Hi there!",
    ]

    total_llm_calls = 0
    total_cache_hits = 0

    for i, query in enumerate(user_queries):
        print(f"{'─' * 55}")
        print(f"[第 {i+1} 轮] 用户: {query}")

        # Step 1: 语义路由
        route_result = router.route(query)
        route_name = route_result["name"] if route_result else "unknown"
        print(f"  [路由] → {route_name}", end="")
        if route_result:
            print(f" (distance={route_result['distance']:.4f})")
        else:
            print()

        # Step 2: 语义缓存检查
        cache_result = llm_cache.check(prompt=query)
        if cache_result:
            total_cache_hits += 1
            answer = cache_result[0]["response"]
            print(f"  [缓存] ✓ 命中! (distance={cache_result[0].get('distance', 'N/A'):.4f})")
            print(f"  [回答] {answer}")
        else:
            total_llm_calls += 1
            print(f"  [缓存] ✗ 未命中，调用 LLM...")

            # Step 3: 获取相关对话历史作为上下文
            relevant_history = session.get_relevant(query, top_k=3)
            if relevant_history:
                print(f"  [历史] 检索到 {len(relevant_history)} 条相关上下文")

            # 模拟 LLM 调用
            answer = simulate_llm_call(query)
            print(f"  [回答] {answer}")

            # Step 4: 存入语义缓存
            llm_cache.store(prompt=query, response=answer)

        # Step 5: 记录对话历史
        session.add_messages([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])

    # ---- 总结 ----
    print(f"\n{'=' * 65}")
    print(f"  演示结束")
    print(f"  总查询: {len(user_queries)}")
    print(f"  LLM 调用: {total_llm_calls} (节省 {total_cache_hits} 次)")
    print(f"  缓存命中率: {total_cache_hits / len(user_queries) * 100:.0f}%")
    print(f"{'=' * 65}")


def module_unit_tests():
    """各模块独立的单元测试（可选运行）。"""
    print("运行各模块单元测试...\n")
    modules = [
        ("embeddings_cache", "embeddings_cache.py"),
        ("semantic_cache", "semantic_cache.py"),
        ("semantic_message_history", "semantic_message_history.py"),
        ("semantic_router", "semantic_router.py"),
    ]
    for name, filename in modules:
        print(f"--- {name} ---")
        os.system(f"python {os.path.join(os.path.dirname(__file__), filename)}")
        print()


if __name__ == "__main__":
    if "--test" in sys.argv:
        module_unit_tests()
    else:
        demo()
