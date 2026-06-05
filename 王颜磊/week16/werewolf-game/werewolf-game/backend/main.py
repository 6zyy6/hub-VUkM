"""
Main 入口 - 启动完整系统
支持命令行和API两种方式启动
"""

import sys
import os
import asyncio
import argparse
import json
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_engine import WerewolfGameEngine
from agent_manager import AgentManager, AgentTeamSimulator, run_single_game_demo
from websocket_server import WerewolfWebSocketServer


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('werewolf_game.log', encoding='utf-8')
        ]
    )


def run_cli_mode():
    """命令行模式运行"""
    print("""
    ╔══════════════════════════════════════════════╗
    ║         狼人杀 AI Agent Team 系统            ║
    ║   Werewolf Multi-Agent Collaboration System  ║
    ╚══════════════════════════════════════════════╝
    """)

    while True:
        print("\n请选择模式：")
        print("1. 单局演示（纯AI对战）")
        print("2. 批量模拟（多局数据分析）")
        print("3. 启动WebSocket服务器（前端观战）")
        print("4. 查看规则说明")
        print("5. 退出")

        choice = input("\n请输入选项 (1-5): ").strip()

        if choice == "1":
            # 单局演示
            print("\n开始单局AI对战...")
            game = WerewolfGameEngine(
                player_names=[f"玩家{i}" for i in range(1, 13)]
            )
            manager = AgentManager(game)
            result = manager.run_full_game()

            print("\n" + "=" * 50)
            print("游戏结果")
            print("=" * 50)
            print(f"获胜阵营: {result['winner']}")
            print(f"总回合数: {result['total_rounds']}")
            print(f"存活玩家数: {len([a for a in manager.agents.values() if a.alive])}")

            print("\n玩家身份揭示：")
            for pid, agent in sorted(manager.agents.items(), key=lambda x: x[0]):
                status = "存活" if agent.alive else "死亡"
                print(f"  玩家{pid:2d} ({agent.role.value:3s}) [{agent.team.value}] - {status}")

            print("\n回合统计：")
            for stat in result.get("round_stats", []):
                print(f"  第{stat['round']}回合: {stat.get('phase', '?')} - 存活{stat.get('alive_count', '?')}人")

            # 保存结果
            output_path = os.path.join(os.path.dirname(__file__), '..', 'game_result.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n详细结果已保存到: {output_path}")

        elif choice == "2":
            # 批量模拟
            try:
                num_games = int(input("请输入模拟局数 (默认10): ").strip() or "10")
                print(f"\n开始模拟 {num_games} 局游戏...")
                simulator = AgentTeamSimulator(num_games=num_games)
                results = simulator.run_simulation()
                report = simulator.get_detailed_report()

                print("\n" + "=" * 50)
                print("模拟结果统计")
                print("=" * 50)
                print(f"总局数: {results['total_games']}")
                print(f"好人胜率: {results['good_wins'] / max(1, results['total_games']) * 100:.1f}%")
                print(f"狼人胜率: {results['evil_wins'] / max(1, results['total_games']) * 100:.1f}%")
                print(f"平均回合数: {results['avg_rounds']:.1f}")

                print("\n角色表现：")
                analysis = report.get("analysis", {})
                role_eff = analysis.get("role_effectiveness", {})
                for role, stats in role_eff.items():
                    print(f"  {role}: 胜率 {stats['win_rate'] * 100:.1f}% ({stats['total_games']}局)")

                # 保存报告
                output_path = os.path.join(os.path.dirname(__file__), '..', 'simulation_report.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                print(f"\n详细报告已保存到: {output_path}")

            except ValueError:
                print("请输入有效的数字")

        elif choice == "3":
            # 启动WebSocket服务器
            print("\n启动WebSocket服务器...")
            print("前端观战页面请打开: frontend/index.html")
            print("WebSocket连接地址: ws://localhost:8765")

            server = WerewolfWebSocketServer(host="localhost", port=8765)
            try:
                asyncio.run(server.start_server())
            except KeyboardInterrupt:
                print("\n服务器已停止")
            except Exception as e:
                print(f"服务器启动失败: {e}")

        elif choice == "4":
            # 规则说明
            print("\n" + "=" * 50)
            print("狼人杀 AI Agent Team 系统说明")
            print("=" * 50)
            print("""
游戏规则：
- 12人标准局：4狼人 + 4神职（预言家、女巫、猎人、守卫） + 4村民
- 狼人阵营：每晚击杀一名玩家，白天伪装发言
- 好人阵营：通过发言分析和投票放逐狼人

回合流程：
1. 夜晚阶段：狼人刀人 → 预言家查验 → 女巫用药 → 守卫守护
2. 白天阶段：公布死者 → 讨论发言 → 投票放逐
3. 胜负判定：狼人全灭→好人胜；狼人数≥好人数→狼人胜

Agent特点：
- 每个Agent拥有独立记忆、信念系统和策略
- 信息严格隔离：Agent只能看到公开信息和自己的夜间情报
- 自主决策：基于角色、局势和对手行为推理决策
- 性格差异：每个Agent有随机的性格参数影响发言和决策风格

技术架构：
- 游戏引擎 (game_engine.py)：回合流转、信息管理、胜负裁决
- 智能体系统 (agents/)：多角色Agent，基于规则+概率决策
- 通信系统 (agent_manager.py)：Agent间消息传递与信念更新
- 观战系统 (websocket_server.py)：实时推送游戏状态
- 前端UI (frontend/index.html)：可视化观战界面
            """)

        elif choice == "5":
            print("感谢使用狼人杀 AI Agent Team 系统！")
            break

        else:
            print("无效选项，请重试")


def main():
    parser = argparse.ArgumentParser(
        description="狼人杀 AI Agent Team 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["cli", "server", "simulate", "demo"],
        default="cli",
        help="运行模式: cli(命令行交互), server(WebSocket服务器), simulate(批量模拟), demo(快速演示)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8765,
        help="WebSocket服务器端口 (默认8765)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="WebSocket服务器地址 (默认localhost)"
    )
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=10,
        help="批量模拟的局数 (默认10)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.mode == "cli":
        run_cli_mode()
    elif args.mode == "server":
        print(f"启动WebSocket服务器 ws://{args.host}:{args.port}")
        server = WerewolfWebSocketServer(host=args.host, port=args.port)
        try:
            asyncio.run(server.start_server())
        except KeyboardInterrupt:
            print("\n服务器已停止")
    elif args.mode == "simulate":
        print(f"开始批量模拟 {args.num_games} 局...")
        simulator = AgentTeamSimulator(num_games=args.num_games)
        results = simulator.run_simulation()
        report = simulator.get_detailed_report()

        output_path = os.path.join(os.path.dirname(__file__), '..', 'simulation_report.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"模拟完成，报告已保存到: {output_path}")
    elif args.mode == "demo":
        run_single_game_demo()


if __name__ == "__main__":
    main()
