"""
Redis VL Demo 主程序
整合所有模块，提供统一的使用入口
"""
from semantic_cache import LLMCacheManager, create_llm_cache
from embeddings_cache import EmbeddingsCacheManager, create_embeddings_cache
from message_history import MessageHistoryManager, create_message_history
from semantic_router import (
    SemanticRouterManager, 
    create_semantic_router,
    create_greeting_route,
    create_farewell_route,
    create_question_route
)
from config import redis_config


class RedisVLService:
    """Redis VL 服务类 - 整合所有功能模块"""
    
    def __init__(self, redis_url: str = None):
        """
        初始化Redis VL服务
        
        Args:
            redis_url: Redis连接地址（可选，默认从配置文件读取）
        """
        self.redis_url = redis_url or redis_config.REDIS_URL
        
        # 初始化各个模块（延迟初始化，按需创建）
        self._llm_cache = None
        self._embeddings_cache = None
        self._message_history = None
        self._semantic_router = None
    
    @property
    def llm_cache(self) -> LLMCacheManager:
        """获取LLM语义缓存实例"""
        if self._llm_cache is None:
            self._llm_cache = create_llm_cache(redis_url=self.redis_url)
        return self._llm_cache
    
    @property
    def embeddings_cache(self) -> EmbeddingsCacheManager:
        """获取嵌入缓存实例"""
        if self._embeddings_cache is None:
            self._embeddings_cache = create_embeddings_cache(redis_url=self.redis_url)
        return self._embeddings_cache
    
    def get_message_history(self, session_name: str = "default") -> MessageHistoryManager:
        """
        获取消息历史实例（支持多会话）
        
        Args:
            session_name: 会话名称
            
        Returns:
            MessageHistoryManager: 消息历史管理器
        """
        return create_message_history(
            name=session_name,
            redis_url=self.redis_url
        )
    
    def get_semantic_router(self, router_name: str = "default") -> SemanticRouterManager:
        """
        获取语义路由实例
        
        Args:
            router_name: 路由器名称
            
        Returns:
            SemanticRouterManager: 语义路由管理器
        """
        return create_semantic_router(
            name=router_name,
            redis_url=self.redis_url
        )
    
    def demo_llm_cache(self):
        """演示LLM语义缓存功能"""
        print("=" * 60)
        print("演示：LLM语义缓存")
        print("=" * 60)
        
        cache = self.llm_cache
        
        # 存储问答对
        print("\n1. 存储问答对到缓存...")
        cache.store(
            prompt="What is the capital city of France?",
            response="Paris"
        )
        cache.store(
            prompt="中国的首都是什么？",
            response="北京"
        )
        print("✓ 已存储两个问答对")
        
        # 查询缓存
        print("\n2. 查询相似问题...")
        results = cache.check(prompt="What is France's capital city?")
        if results:
            print(f"✓ 找到缓存答案: {results[0]['response']}")
        else:
            print("✗ 未找到缓存")
        
        # 查看统计信息
        print("\n3. 缓存统计信息:")
        stats = cache.get_stats()
        print(f"   统计信息: {stats}")
    
    def demo_embeddings_cache(self):
        """演示嵌入缓存功能"""
        print("\n" + "=" * 60)
        print("演示：嵌入缓存")
        print("=" * 60)
        
        embed_cache = self.embeddings_cache
        
        # 第一次调用 - 计算并缓存
        print("\n1. 第一次向量化（需要计算）...")
        text = "What is machine learning?"
        embedding1 = embed_cache.embed(text)
        print(f"✓ 向量维度: {len(embedding1)}")
        print(f"   前5个值: {embedding1[:5]}")
        
        # 第二次调用 - 从缓存获取
        print("\n2. 第二次向量化（从缓存获取，更快）...")
        embedding2 = embed_cache.embed(text)
        print(f"✓ 向量维度: {len(embedding2)}")
        print(f"   前5个值: {embedding2[:5]}")
        
        # 验证两次结果一致
        if embedding1 == embedding2:
            print("✓ 两次结果一致，缓存生效！")
        
        # 批量处理
        print("\n3. 批量向量化...")
        texts = [
            "Hello world",
            "Machine learning is fascinating",
            "Redis is a powerful database"
        ]
        embeddings = embed_cache.embed_many(texts)
        print(f"✓ 处理了 {len(embeddings)} 个文本")
        
        # 查看统计信息
        print("\n4. 缓存统计信息:")
        stats = embed_cache.get_stats()
        print(f"   统计信息: {stats}")
    
    def demo_message_history(self):
        """演示消息历史功能"""
        print("\n" + "=" * 60)
        print("演示：语义消息历史")
        print("=" * 60)
        
        history = self.get_message_history(session_name="demo-session")
        
        # 添加对话
        print("\n1. 添加对话历史...")
        messages = [
            {"role": "user", "content": "hello, how are you?"},
            {"role": "llm", "content": "I'm doing fine, thanks."},
            {"role": "user", "content": "what is the weather going to be today?"},
            {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}}
        ]
        history.add_messages(messages)
        print(f"✓ 添加了 {len(messages)} 条消息")
        
        # 获取最近消息
        print("\n2. 获取最近的对话...")
        recent = history.get_messages(limit=3)
        for i, msg in enumerate(recent, 1):
            print(f"   {i}. [{msg['role']}] {msg['content']}")
        
        # 搜索相似消息
        print("\n3. 搜索相似的问候语...")
        similar = history.search_similar(query="hi, how are you doing?", limit=2)
        if similar:
            for i, msg in enumerate(similar, 1):
                print(f"   {i}. [{msg.get('role')}] {msg.get('content')}")
        else:
            print("   未找到相似消息")
        
        # 查看统计信息
        print("\n4. 消息历史统计:")
        count = history.get_message_count()
        print(f"   消息总数: {count}")
    
    def demo_semantic_router(self):
        """演示语义路由功能"""
        print("\n" + "=" * 60)
        print("演示：语义路由（意图识别）")
        print("=" * 60)
        
        # 创建预定义路由
        routes = [
            create_greeting_route(),
            create_farewell_route(),
            create_question_route()
        ]
        
        router = self.get_semantic_router(router_name="intent-router")
        router.add_routes(routes)
        print("\n1. 创建了3个路由类别: greeting, farewell, question")
        
        # 测试路由分类
        test_queries = [
            "Hi, good morning",
            "Bye, see you later",
            "What is the weather today?",
            "How are you?",
            "再见"
        ]
        
        print("\n2. 测试路由分类:")
        for query in test_queries:
            result = router.route(query)
            if result:
                print(f"   '{query}' -> {result.get('name', 'unknown')}")
            else:
                print(f"   '{query}' -> 未匹配")
        
        # 查看所有路由
        print("\n3. 所有路由信息:")
        all_routes = router.get_all_routes()
        for route in all_routes:
            print(f"   - {route['name']}: {route['references'][:3]}...")
        
        # 查看统计信息
        print("\n4. 路由统计信息:")
        stats = router.get_stats()
        print(f"   统计信息: {stats}")
    
    def run_all_demos(self):
        """运行所有演示"""
        print("\n" + "#" * 60)
        print("# Redis VL Demo - 完整功能演示")
        print("#" * 60)
        
        try:
            self.demo_llm_cache()
            self.demo_embeddings_cache()
            self.demo_message_history()
            self.demo_semantic_router()
            
            print("\n" + "#" * 60)
            print("# 所有演示完成！")
            print("#" * 60)
        except Exception as e:
            print(f"\n✗ 演示过程中出现错误: {e}")
            print("请确保Redis服务器正在运行，并且已安装所需的依赖包")


def main():
    """主函数"""
    # 创建服务实例
    service = RedisVLService()
    
    # 运行所有演示
    service.run_all_demos()


if __name__ == "__main__":
    main()
