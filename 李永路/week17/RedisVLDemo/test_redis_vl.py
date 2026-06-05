"""
Redis VL Demo 测试文件
测试各个模块的功能（需要先启动Redis服务器并安装依赖）
"""
import unittest
from unittest.mock import Mock, patch
from semantic_cache import LLMCacheManager, create_llm_cache
from embeddings_cache import EmbeddingsCacheManager, create_embeddings_cache
from message_history import MessageHistoryManager, create_message_history
from semantic_router import (
    SemanticRouterManager, 
    create_semantic_router,
    create_greeting_route,
    create_farewell_route
)


class TestLLMCache(unittest.TestCase):
    """测试LLM语义缓存"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 使用mock避免实际连接Redis
        self.mock_cache = Mock()
        self.cache_manager = LLMCacheManager(
            name="test_cache",
            ttl=3600,
            distance_threshold=0.1,
            redis_url="redis://localhost:6379"
        )
        self.cache_manager.cache = self.mock_cache
    
    def test_store(self):
        """测试存储功能"""
        self.mock_cache.store.return_value = None
        result = self.cache_manager.store(
            prompt="What is AI?",
            response="Artificial Intelligence"
        )
        self.assertTrue(result)
        self.mock_cache.store.assert_called_once()
    
    def test_check(self):
        """测试查询功能"""
        mock_result = [{"response": "Paris", "score": 0.05}]
        self.mock_cache.check.return_value = mock_result
        
        result = self.cache_manager.check(prompt="What is France's capital?")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["response"], "Paris")
    
    def test_clear(self):
        """测试清空功能"""
        self.mock_cache.clear.return_value = None
        result = self.cache_manager.clear()
        self.assertTrue(result)
        self.mock_cache.clear.assert_called_once()


class TestEmbeddingsCache(unittest.TestCase):
    """测试嵌入缓存"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.mock_embed_cache = Mock()
        self.mock_vectorizer = Mock()
        
        self.embed_manager = EmbeddingsCacheManager(
            name="test_embed_cache",
            ttl=3600,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            redis_url="redis://localhost:6379"
        )
        self.embed_manager.embed_cache = self.mock_embed_cache
        self.embed_manager.vectorizer = self.mock_vectorizer
    
    def test_embed(self):
        """测试单个文本向量化"""
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.mock_vectorizer.embed.return_value = mock_embedding
        
        result = self.embed_manager.embed("Hello world")
        self.assertEqual(len(result), 5)
        self.mock_vectorizer.embed.assert_called_once_with("Hello world")
    
    def test_embed_many(self):
        """测试批量文本向量化"""
        mock_embedding = [0.1, 0.2, 0.3]
        self.mock_vectorizer.embed.return_value = mock_embedding
        
        texts = ["text1", "text2", "text3"]
        result = self.embed_manager.embed_many(texts)
        self.assertEqual(len(result), 3)
        self.assertEqual(self.mock_vectorizer.embed.call_count, 3)


class TestMessageHistory(unittest.TestCase):
    """测试消息历史"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.mock_history = Mock()
        
        self.history_manager = MessageHistoryManager(
            name="test_session",
            distance_threshold=0.7,
            redis_url="redis://localhost:6379"
        )
        self.history_manager.history = self.mock_history
    
    def test_add_message(self):
        """测试添加单条消息"""
        self.mock_history.add_messages.return_value = None
        result = self.history_manager.add_message(
            role="user",
            content="Hello"
        )
        self.assertTrue(result)
        self.mock_history.add_messages.assert_called_once()
    
    def test_add_messages(self):
        """测试批量添加消息"""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "llm", "content": "Hello!"}
        ]
        self.mock_history.add_messages.return_value = None
        
        result = self.history_manager.add_messages(messages)
        self.assertTrue(result)
        self.mock_history.add_messages.assert_called_once_with(messages)
    
    def test_get_messages(self):
        """测试获取消息"""
        mock_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "llm", "content": "Hello!"}
        ]
        self.mock_history.get_recent.return_value = mock_messages
        
        result = self.history_manager.get_messages(limit=10)
        self.assertEqual(len(result), 2)
        self.mock_history.get_recent.assert_called_once_with(limit=10)


class TestSemanticRouter(unittest.TestCase):
    """测试语义路由"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.routes = [
            create_greeting_route(),
            create_farewell_route()
        ]
        
        self.router_manager = SemanticRouterManager(
            name="test_router",
            routes=self.routes,
            distance_threshold=0.3,
            redis_url="redis://localhost:6379"
        )
        # Mock路由器
        self.router_manager.router = Mock()
    
    def test_add_route(self):
        """测试添加路由"""
        result = self.router_manager.add_route(
            name="question",
            references=["what", "how", "why"]
        )
        self.assertTrue(result)
        self.assertEqual(len(self.router_manager.routes), 3)
    
    def test_route(self):
        """测试路由分类"""
        mock_result = {"name": "greeting", "metadata": {"type": "greeting"}}
        self.router_manager.router.return_value = mock_result
        
        result = self.router_manager.route("Hi there!")
        self.assertEqual(result["name"], "greeting")
        self.router_manager.router.assert_called_once_with("Hi there!")
    
    def test_route_with_fallback(self):
        """测试带默认值的路由"""
        self.router_manager.router.return_value = None
        
        result = self.router_manager.route_with_fallback(
            query="unknown query",
            default_route="unknown"
        )
        self.assertEqual(result["name"], "unknown")
    
    def test_get_all_routes(self):
        """测试获取所有路由"""
        routes = self.router_manager.get_all_routes()
        self.assertEqual(len(routes), 2)
        self.assertIn("greeting", [r["name"] for r in routes])


class TestConfig(unittest.TestCase):
    """测试配置模块"""
    
    def test_redis_config(self):
        """测试Redis配置"""
        from config import redis_config
        self.assertIsNotNone(redis_config.REDIS_URL)
        self.assertIsInstance(redis_config.REDIS_DB, int)
    
    def test_model_config(self):
        """测试模型配置"""
        from config import model_config
        self.assertEqual(
            model_config.DEFAULT_MODEL,
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.assertEqual(model_config.VECTOR_DIMENSION, 384)
    
    def test_cache_config(self):
        """测试缓存配置"""
        from config import cache_config
        self.assertEqual(cache_config.SEMANTIC_CACHE_TTL, 3600)
        self.assertEqual(cache_config.SEMANTIC_CACHE_DISTANCE_THRESHOLD, 0.1)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestLLMCache))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingsCache))
    suite.addTests(loader.loadTestsFromTestCase(TestMessageHistory))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Redis VL Demo - 单元测试")
    print("=" * 60)
    print("\n注意：这些测试使用Mock，不需要实际的Redis连接\n")
    
    run_tests()
