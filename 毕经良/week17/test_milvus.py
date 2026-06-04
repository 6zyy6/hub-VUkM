"""
LLM Smart Cache (Milvus版本) 测试脚本
测试所有组件的功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

MILVUS_URI = "https://in03-6fc9fda7586c8a5.serverless.aws-eu-central-1.cloud.zilliz.com"
MILVUS_TOKEN = "319f97861036cbada2e4af735478028c1dda6e728b875e7d698472763eed54c46927310d70760cad623df9071587e2cb19f48637"


def get_embedding(text):
    """确定性embedding方法 - 相同文本产生相同向量"""
    if isinstance(text, str):
        text = [text]
    result = []
    for t in text:
        # 用文本的hash生成固定的随机种子
        h = hash(t)
        np.random.seed(h % (2**32))
        result.append(np.random.rand(128).astype(np.float32))
    return np.array(result)


def test_semantic_cache():
    """测试语义缓存"""
    print("\n" + "=" * 50)
    print("测试 SemanticCache (Milvus)")
    print("=" * 50)

    from llm_smart_cache_milvus import SemanticCache

    cache = SemanticCache(
        name="test_semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        vector_dimension=128,
        distance_threshold=1.0
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

    cache.close()
    print("测试完成!")


def test_embeddings_cache():
    """测试嵌入缓存"""
    print("\n" + "=" * 50)
    print("测试 EmbeddingsCache (Milvus)")
    print("=" * 50)

    from llm_smart_cache_milvus import EmbeddingsCache

    cache = EmbeddingsCache(
        name="test_embed_cache",
        ttl=360,
        redis_url="localhost",
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        vector_dimension=128,
    )

    cache.clear_all()

    text = "hello world"
    embedding = get_embedding(text)

    print("存储 embedding...")
    result = cache.store(text=text, embedding=embedding)
    print(f"  store result: {result}")

    print("获取缓存的 embedding...")
    cached = cache.call(text=text)
    print(f"  call result: {cached}")

    print("删除缓存...")
    result = cache.delete(text=text)
    print(f"  delete result: {result}")

    cache.close()
    print("测试完成!")


def test_semantic_message_history():
    """测试对话历史"""
    print("\n" + "=" * 50)
    print("测试 SemanticMessageHistory (Milvus)")
    print("=" * 50)

    from llm_smart_cache_milvus import SemanticMessageHistory

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


def test_semantic_router():
    """测试语义路由"""
    print("\n" + "=" * 50)
    print("测试 SemanticRouter (Milvus)")
    print("=" * 50)

    from llm_smart_cache_milvus import SemanticRouter

    router = SemanticRouter(
        name="test-router",
        embedding_method=get_embedding,
        redis_url="localhost",
        milvus_uri=MILVUS_URI,
        milvus_token=MILVUS_TOKEN,
        vector_dimension=128,
    )

    print("添加路由...")
    router.add_route(
        questions=["hello", "hi", "good morning"],
        target="greeting",
        distance_threshold=1.0
    )
    router.add_route(
        questions=["how to return", "如何退货", "return product"],
        target="refund",
        distance_threshold=1.0
    )

    print("路由测试...")
    result1 = router.route("hello")
    print(f"  'hello' -> {result1}")

    result2 = router.route("如何退货")
    print(f"  '如何退货' -> {result2}")

    print("缓存测试（第二次调用应走缓存）...")
    result1_cached = router.route("hello")
    print(f"  'hello' (cached) -> {result1_cached}")

    print("获取所有路由:")
    routes = router.get_all_routes()
    print(f"  routes: {routes}")

    router.clear_cache()
    router.close()
    print("测试完成!")


if __name__ == "__main__":
    try:
        test_semantic_cache()
        test_embeddings_cache()
        test_semantic_message_history()
        test_semantic_router()
        print("\n" + "=" * 50)
        print("所有测试完成!")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()