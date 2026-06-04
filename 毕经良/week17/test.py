"""
LLM Smart Cache 测试脚本
测试所有组件的功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def get_embedding(text):
    """模拟embedding方法"""
    if isinstance(text, str):
        text = [text]
    return np.array([np.ones(128) * (hash(t) % 100) for t in text], dtype=np.float32)


def test_embeddings_cache():
    """测试嵌入缓存"""
    print("\n" + "=" * 50)
    print("测试 EmbeddingsCache")
    print("=" * 50)

    from llm_smart_cache import EmbeddingsCache

    cache = EmbeddingsCache(
        name="test_embed_cache",
        ttl=360,
        redis_url="localhost"
    )

    cache.clear_all()

    text = "hello world"
    embedding = get_embedding(text)

    print("存储 embedding...")
    result = cache.store(text=text, embedding=embedding)
    print(f"  store result: {result}")

    print("获取缓存的 embedding...")
    cached = cache.call(text=text)
    print(f"  call result: {cached is not None}, shape: {cached[0].shape if cached else None}")

    print("删除缓存...")
    result = cache.delete(text=text)
    print(f"  delete result: {result}")

    print("测试完成!")


def test_semantic_cache():
    """测试语义缓存"""
    print("\n" + "=" * 50)
    print("测试 SemanticCache")
    print("=" * 50)

    from llm_smart_cache import SemanticCache

    cache = SemanticCache(
        name="test_semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        distance_threshold=50
    )

    cache.clear_cache()

    print("存储问答对...")
    cache.store(prompt="今天天气怎么样？", response="今天天气很好")

    print("检查缓存...")
    result = cache.check(prompt="今天天气怎么样？")
    print(f"  check result: {result}")

    print("存储另一个问答对...")
    cache.store(prompt="如何退货？", response="请联系客服退货")

    print("检查缓存...")
    result = cache.check(prompt="如何退货？")
    print(f"  check result: {result}")

    print("测试完成!")


def test_semantic_message_history():
    """测试对话历史"""
    print("\n" + "=" * 50)
    print("测试 SemanticMessageHistory")
    print("=" * 50)

    from llm_smart_cache import SemanticMessageHistory

    history = SemanticMessageHistory(
        name="test-session",
        redis_url="localhost",
        ttl=360
    )

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

    print("测试完成!")


def test_semantic_router():
    """测试语义路由"""
    print("\n" + "=" * 50)
    print("测试 SemanticRouter")
    print("=" * 50)

    from llm_smart_cache import SemanticRouter

    router = SemanticRouter(
        name="test-router",
        embedding_method=get_embedding,
        redis_url="localhost",
        ttl=360
    )

    print("添加路由...")
    router.add_route(
        questions=["hello", "hi", "good morning"],
        target="greeting",
        distance_threshold=100
    )
    router.add_route(
        questions=["how to return", "如何退货", "return product"],
        target="refund",
        distance_threshold=100
    )

    print("路由测试...")
    result1 = router.route("hello")
    print(f"  'hello' -> {result1}")

    result2 = router.route("hi")
    print(f"  'hi' -> {result2}")

    result3 = router.route("如何退货")
    print(f"  '如何退货' -> {result3}")

    print("缓存测试（第二次调用应走缓存）...")
    result1_cached = router.route("hello")
    print(f"  'hello' (cached) -> {result1_cached}")

    print("获取所有路由:")
    routes = router.get_all_routes()
    print(f"  routes: {routes}")

    router.clear_cache()
    print("清除缓存完成!")

    print("测试完成!")


if __name__ == "__main__":
    try:
        test_embeddings_cache()
        test_semantic_cache()
        test_semantic_message_history()
        test_semantic_router()
        print("\n" + "=" * 50)
        print("所有测试完成!")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()