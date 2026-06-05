"""
AI 狼人杀 - 主程序入口
运行完整的狼人杀多智能体博弈系统
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from werewolf_agents.engine import WerewolfGameEngine
from werewolf_agents.logger import GameLogger, ObservableGameState
from werewolf_agents.config import get_config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI 狼人杀多智能体博弈系统")
    parser.add_argument(
        "--config", 
        type=str, 
        default="simple_6",
        choices=["standard_12", "simple_6", "quick_4"],
        help="游戏配置类型"
    )
    parser.add_argument(
        "--max-rounds", 
        type=int, 
        default=10,
        help="最大回合数"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="日志目录"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="使用 LLM Agent（需要配置 API Key）"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="mock",
        choices=["openai", "qwen", "deepseek", "mock"],
        help="LLM 提供商"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key（可选，也可通过环境变量设置）"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("       AI 狼人杀 - 多智能体协作与博弈系统")
    print("=" * 70)
    print()
    
    # 初始化日志系统
    game_logger = GameLogger(log_dir=args.log_dir)
    observable_state = ObservableGameState(game_logger)
    
    # 获取配置
    player_configs = get_config(args.config)
    
    print(f"使用配置: {args.config}")
    print(f"玩家数量: {len(player_configs)}")
    print(f"最大回合数: {args.max_rounds}")
    print(f"Agent 类型: {'LLM Agent' if args.use_llm else '规则 Agent'}")
    if args.use_llm:
        print(f"LLM 提供商: {args.llm_provider}")
    print()
    
    # 显示角色分布
    role_counts = {}
    for config in player_configs:
        role = config["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print("角色分布:")
    for role, count in role_counts.items():
        print(f"  - {role}: {count}人")
    print()
    
    # 创建 LLM 客户端（如果使用）
    llm_client = None
    if args.use_llm and args.api_key:
        from werewolf_agents.llm_client import create_llm_client
        try:
            llm_client = create_llm_client(
                provider=args.llm_provider,
                api_key=args.api_key
            )
            print(f"✓ LLM 客户端初始化成功")
        except Exception as e:
            print(f"✗ LLM 客户端初始化失败: {e}")
            print("将使用 Mock LLM 客户端")
            args.llm_provider = "mock"
    
    # 创建游戏引擎
    engine = WerewolfGameEngine(
        player_configs,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_client=llm_client
    )
    
    # 记录游戏开始
    game_logger.log_game_start(engine.game_id, len(player_configs))
    
    # 运行游戏
    try:
        engine.run_game(max_rounds=args.max_rounds)
        
        # 生成游戏报告
        report = engine.get_game_report()
        
        # 保存报告
        game_logger.save_report(report)
        
        # 导出观测数据
        observable_state.export_observations("observations.json")
        
        print("\n" + "=" * 70)
        print("游戏结束！详细日志和报告已保存。")
        print(f"日志文件: {game_logger.get_log_file_path()}")
        print(f"报告文件: {game_logger.get_report_file_path()}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n游戏运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
