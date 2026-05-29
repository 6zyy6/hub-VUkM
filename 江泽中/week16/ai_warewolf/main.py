"""
AI狼人杀主入口
"""

import yaml
import json
from pathlib import Path
from ai_werewolf.core.game_engine import GameEngine
from ai_werewolf.llm.deepseek_llm import DeepSeekLLM
from ai_werewolf.utils.logger import GameLogger


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    logger = GameLogger()
    logger.info("🎮 AI狼人杀多智能体系统启动中...")

    config = load_config()

    llm_config = config.get('llm', {})
    llm_client = None

    try:
        if llm_config.get('provider') == 'deepseek':
            llm_client = DeepSeekLLM(
                api_key=llm_config.get('api_key'),
                model=llm_config.get('model', 'deepseek-chat'),
                temperature=llm_config.get('temperature', 0.7),
                max_tokens=llm_config.get('max_tokens', 500)
            )
            logger.success("LLM客户端初始化成功")
    except Exception as e:
        logger.warning(f"LLM客户端初始化失败: {e}，将使用默认模式")

    game_config = config.get('game', {})
    engine = GameEngine(game_config, llm_client)

    engine.initialize_game()

    logger.info("\n开始游戏对局...\n")
    result = engine.run_game()

    logger.info("\n" + "=" * 60)
    logger.info("📊 游戏结果统计")
    logger.info("=" * 60)
    logger.info(f"获胜方: {result['winner']}")
    logger.info(f"总轮次: {result['total_rounds']}")

    logger.info("\n玩家状态:")
    for pid, pinfo in result['players'].items():
        status_icon = "✅" if pinfo['is_alive'] else "❌"
        logger.info(
            f"  {status_icon} Player_{pid} - {pinfo['role']} - "
            f"{'存活' if pinfo['is_alive'] else '淘汰'}"
        )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    result_file = output_dir / f"game_result_{result['total_rounds']}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.success(f"\n游戏结果已保存到: {result_file}")

    return result


if __name__ == "__main__":
    main()
