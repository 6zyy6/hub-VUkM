"""
快速测试脚本 - 验证 LLM Agent 功能
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from werewolf_agents.engine import WerewolfGameEngine
from werewolf_agents.config import get_config


def test_mock_llm():
    """测试 Mock LLM Agent"""
    print("\n" + "="*70)
    print("测试：Mock LLM Agent")
    print("="*70 + "\n")
    
    player_configs = get_config("quick_4")
    
    engine = WerewolfGameEngine(
        player_configs,
        use_llm=True,
        llm_provider="mock"
    )
    
    print("✓ 引擎初始化成功")
    print(f"✓ 玩家数量: {len(engine.game_state.players)}")
    print(f"✓ 使用 LLM: {engine.use_llm}")
    print(f"✓ LLM 提供商: {engine.llm_provider}")
    
    # 运行游戏
    engine.run_game(max_rounds=2)
    
    report = engine.get_game_report()
    print(f"\n✓ 游戏结束，获胜方: {report['winner']}")
    print(f"✓ 总回合数: {report['total_rounds']}")
    
    return True


def test_rule_agent():
    """测试规则 Agent（对比）"""
    print("\n" + "="*70)
    print("测试：规则 Agent（对比）")
    print("="*70 + "\n")
    
    player_configs = get_config("quick_4")
    
    engine = WerewolfGameEngine(
        player_configs,
        use_llm=False  # 使用规则 Agent
    )
    
    print("✓ 引擎初始化成功")
    print(f"✓ 玩家数量: {len(engine.game_state.players)}")
    print(f"✓ 使用 LLM: {engine.use_llm}")
    
    # 运行游戏
    engine.run_game(max_rounds=2)
    
    report = engine.get_game_report()
    print(f"\n✓ 游戏结束，获胜方: {report['winner']}")
    print(f"✓ 总回合数: {report['total_rounds']}")
    
    return True


def test_agent_creation():
    """测试 Agent 创建"""
    print("\n" + "="*70)
    print("测试：Agent 创建")
    print("="*70 + "\n")
    
    from werewolf_agents.llm_agents import LLMAgent
    from werewolf_agents.models import Player, GameState, RoleType
    
    # 创建测试玩家
    player = Player(
        player_id=1,
        name="测试玩家",
        role=RoleType.SEER
    )
    
    game_state = GameState(game_id="test")
    game_state.players[1] = player
    
    # 创建 LLM Agent
    agent = LLMAgent(player, game_state, llm_provider="mock")
    
    print(f"✓ Agent 创建成功")
    print(f"✓ 角色: {agent.role.value}")
    print(f"✓ 系统提示词长度: {len(agent.system_prompt)}")
    
    # 测试发言生成
    speech = agent.day_speech()
    print(f"✓ 发言生成成功: {speech[:50]}...")
    
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("AI 狼人杀 - LLM Agent 功能测试")
    print("="*70)
    
    tests = [
        ("Agent 创建测试", test_agent_creation),
        ("Mock LLM 测试", test_mock_llm),
        ("规则 Agent 对比测试", test_rule_agent),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "✅ 通过", None))
        except Exception as e:
            results.append((name, "❌ 失败", str(e)))
            import traceback
            traceback.print_exc()
    
    # 打印测试结果汇总
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    for name, status, error in results:
        print(f"{status} {name}")
        if error:
            print(f"   错误: {error}")
    
    passed = sum(1 for _, status, _ in results if status == "✅ 通过")
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！LLM Agent 功能正常。")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
