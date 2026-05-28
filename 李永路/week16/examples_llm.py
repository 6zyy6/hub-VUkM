"""
LLM Agent 使用示例
演示如何使用不同的大语言模型运行狼人杀游戏
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from werewolf_agents.engine import WerewolfGameEngine
from werewolf_agents.config import get_config


def example_mock_llm():
    """示例1：使用 Mock LLM（无需 API Key，用于测试）"""
    print("=" * 70)
    print("示例1：使用 Mock LLM Agent")
    print("=" * 70)
    
    player_configs = get_config("quick_4")
    
    engine = WerewolfGameEngine(
        player_configs,
        use_llm=True,
        llm_provider="mock"
    )
    
    engine.run_game(max_rounds=3)
    report = engine.get_game_report()
    
    print(f"\n游戏结果: {report['winner']}")
    print(f"详细日志已保存\n")


def example_openai():
    """示例2：使用 OpenAI GPT（需要 API Key）"""
    print("=" * 70)
    print("示例2：使用 OpenAI GPT Agent")
    print("=" * 70)
    
    # 方式1：通过环境变量设置 API Key
    # export OPENAI_API_KEY="" (Linux/Mac)
    # set OPENAI_API_KEY= (Windows)
    
    # 方式2：直接传入 API Key
    api_key = ""  # 替换为你的 API Key
    
    try:
        from werewolf_agents.llm_client import OpenAIClient
        
        player_configs = get_config("quick_4")
        
        # 创建自定义 LLM 客户端
        llm_client = OpenAIClient(api_key=api_key, model="qwen3.6-flash")
        
        # 创建游戏引擎并使用自定义客户端
        engine = WerewolfGameEngine(
            player_configs,
            use_llm=True,
            llm_client=llm_client
        )
        
        engine.run_game(max_rounds=3)
        report = engine.get_game_report()
        
        print(f"\n游戏结果: {report['winner']}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已设置正确的 API Key\n")


def example_qwen():
    """示例3：使用通义千问（需要 API Key）"""
    print("=" * 70)
    print("示例3：使用通义千问 Agent")
    print("=" * 70)
    
    # 获取 API Key: https://dashscope.console.aliyun.com/
    api_key = ""  # 替换为你的 API Key
    
    try:
        from werewolf_agents.llm_client import QwenClient
        
        player_configs = get_config("quick_4")
        
        llm_client = QwenClient(api_key=api_key, model="qwen-plus")
        
        engine = WerewolfGameEngine(
            player_configs,
            use_llm=True,
            llm_client=llm_client
        )
        
        engine.run_game(max_rounds=3)
        report = engine.get_game_report()
        
        print(f"\n游戏结果: {report['winner']}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已安装 dashscope 并设置正确的 API Key\n")


def example_deepseek():
    """示例4：使用 DeepSeek（需要 API Key）"""
    print("=" * 70)
    print("示例4：使用 DeepSeek Agent")
    print("=" * 70)
    
    # 获取 API Key: https://platform.deepseek.com/
    api_key = "your-deepseek-api-key-here"  # 替换为你的 API Key
    
    try:
        from werewolf_agents.llm_client import DeepSeekClient
        
        player_configs = get_config("quick_4")
        
        llm_client = DeepSeekClient(api_key=api_key, model="deepseek-chat")
        
        engine = WerewolfGameEngine(
            player_configs,
            use_llm=True,
            llm_client=llm_client
        )
        
        engine.run_game(max_rounds=3)
        report = engine.get_game_report()
        
        print(f"\n游戏结果: {report['winner']}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保已设置正确的 API Key\n")


def example_mixed_agents():
    """示例5：混合使用规则 Agent 和 LLM Agent"""
    print("=" * 70)
    print("示例5：混合 Agent（部分玩家使用 LLM）")
    print("=" * 70)
    
    # 这个功能需要扩展 engine.py 来支持每个玩家独立配置
    # 这里仅作概念演示
    print("提示：可以通过修改 player_configs 为每个玩家指定是否使用 LLM")
    print("目前所有玩家统一使用相同类型的 Agent\n")


if __name__ == "__main__":
    print("\nAI 狼人杀 - LLM Agent 使用示例\n")
    print("请选择要运行的示例：")
    print("1. Mock LLM（推荐首次使用）")
    print("2. OpenAI GPT")
    print("3. 通义千问")
    print("4. DeepSeek")
    print("5. 混合 Agent")
    print()
    
    choice = input("请输入选项 (1-5): ").strip()
    
    examples = {
        "1": example_mock_llm,
        "2": example_openai,
        "3": example_qwen,
        "4": example_deepseek,
        "5": example_mixed_agents,
    }
    
    if choice in examples:
        examples[choice]()
    else:
        print("无效选项，运行默认示例（Mock LLM）")
        example_mock_llm()
