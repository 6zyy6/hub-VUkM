"""
AI Werewolf - 全自动狼人杀对战系统
直接运行 python main.py 即可开始游戏

功能：
- 自动创建6个AI Agent（2狼人+2平民+1预言家+1女巫）
- 自动执行夜晚/白天回合
- 结构化日志输出
- 自动判定胜负
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.engine.game_engine import (
    GameEngine, GameLogger, Phase,
    NightActions, Player, DeathRecord, CauseOfDeath,
    MockAI, GameRunner,
)
from src.agents import (
    create_all_agents, BaseAgent, WerewolfAgent, SeerAgent, WitchAgent, VillagerAgent,
    ActionResult, ActionType, GameContext,
)
from src.roles.role_def import DEFAULT_6P_ROLES


# ============================================================
# 模拟 LLM 客户端（用于测试）
# ============================================================

class SimpleMockLLM:
    """简化模拟LLM - 用于测试"""

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        await asyncio.sleep(0.1)
        return "模拟决策"


class SmartMockLLM:
    """智能模拟LLM - 基于规则的决策"""

    def __init__(self):
        self.player_name = ""
        self.role = ""

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        await asyncio.sleep(0.2)

        if "狼人" in prompt and "杀害" in prompt:
            return self._wolf_kill_decision(prompt)
        elif "查验" in prompt or "预言家" in prompt:
            return self._seer_check_decision(prompt)
        elif "女巫" in prompt:
            return self._witch_decision(prompt)
        elif "发言" in prompt:
            return self._speech_decision(prompt)
        elif "投票" in prompt:
            return self._vote_decision(prompt)

        return "等待"

    def _wolf_kill_decision(self, prompt: str) -> str:
        import random
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        candidates = []
        if "候选目标:" in prompt:
            parts = prompt.split("候选目标:")
            if len(parts) > 1:
                cand_part = parts[1].split("\n")[0].strip()
                candidates = [c.strip() for c in cand_part.split(",") if c.strip() in names]
        if candidates:
            return random.choice(candidates)
        return random.choice(names)

    def _seer_check_decision(self, prompt: str) -> str:
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        for name in names:
            if name in prompt:
                return name
        return "Bob"

    def _witch_decision(self, prompt: str) -> str:
        if "救" in prompt.lower() and "解药" in prompt:
            return "救 Bob"
        elif "毒" in prompt.lower() and "毒药" in prompt:
            return "毒 Bob"
        return "等待"

    def _speech_decision(self, prompt: str) -> str:
        """根据其他玩家的发言内容生成针对性回应"""
        # 提取其他玩家的发言
        speeches = []
        if "今天已有玩家的发言" in prompt or "今天所有玩家的发言记录" in prompt:
            lines = prompt.split("\n")
            for line in lines:
                if "\": \"" in line or '"' in line:
                    speeches.append(line.strip())

        import random
        # 如果有其他玩家的发言，引用分析
        if speeches and random.random() > 0.5:
            # 随机选一个发言来分析
            target_line = random.choice(speeches)
            # 简单提取发言人
            parts = target_line.split(":")
            speaker = parts[0].strip().split()[-1] if len(parts) > 1 else "某人"
            responses = [
                f"我听了{speaker}的发言，感觉有些地方逻辑不太通顺，需要大家注意。",
                f"关于{speaker}说的内容，我有不同看法，我觉得这个分析有问题。",
                f"我同意{speaker}的部分观点，但还有一些细节值得商榷。",
                f"{speaker}的发言让我觉得有点可疑，为什么会对这个话题这么关注？",
                f"我注意到{speaker}在转移话题，是不是想掩盖什么？",
            ]
            return random.choice(responses)

        # 没有可引用的发言，返回通用内容
        templates = [
            "我觉得场上局势比较复杂，需要仔细分析每个人的发言。",
            "目前没有明显线索，但我怀疑有人故意搅混水。",
            "从发言来看，我认为我们需要重点关注逻辑矛盾的人。",
            "我倾向于相信跳预言家的人，但也要保持警惕。",
            "大家不要被带节奏，我们应该理性分析每个人的发言。",
            "我注意到有人一直在转移焦点，这很可疑。",
            "作为好人，我会认真分析每个玩家的发言和投票行为。",
        ]
        return random.choice(templates)

    def _vote_decision(self, prompt: str) -> str:
        """基于发言分析投票"""
        import random
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

        # 提取所有发言
        speeches = []
        lines = prompt.split("\n")
        for line in lines:
            for name in names:
                if f'  {name}:' in line or f'  {name} "' in line:
                    speeches.append(name)
                    break

        # 如果有发言，选一个发言过的玩家（不在死亡列表中）
        # 先提取死亡玩家
        dead = []
        for i, line in enumerate(lines):
            if "死亡玩家" in line:
                dead_part = line.split(":")[-1].strip()
                if dead_part and dead_part != "无":
                    dead = [n.strip() for n in dead_part.replace("'", "").split(",")]

        living = [n for n in names if n not in dead]

        # 有发言记录时，选一个发言过的人（模拟"基于发言"）
        voted_speakers = [n for n in speeches if n in living]
        if voted_speakers:
            return random.choice(voted_speakers)
        return random.choice(living) if living else "Bob"


# ============================================================
# AI Agent 包装器
# ============================================================

class AgentWrapper:
    """将新Agent系统适配到GameEngine"""

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.name = agent.name
        self.role = agent.role

    @property
    def is_wolf(self) -> bool:
        return self.role == "werewolf"

    @property
    def is_good(self) -> bool:
        return not self.is_wolf

    @property
    def is_alive(self) -> bool:
        return self.agent.memory.public_info.get("is_alive", True)

    @property
    def can_speak(self) -> bool:
        return self.agent.memory.public_info.get("can_speak", True)

    @property
    def wolf_teammates(self) -> List[str]:
        return self.agent.memory.private_info.get("teammates", [])

    @property
    def seer_checks(self) -> Dict[str, bool]:
        return self.agent.memory.private_info.get("checks", {})

    @property
    def heal_potion(self) -> int:
        return self.agent.memory.private_info.get("potions", {}).get("heal", 1)

    @property
    def poison_potion(self) -> int:
        return self.agent.memory.private_info.get("potions", {}).get("poison", 1)

    def set_alive(self, alive: bool):
        self.agent.memory.public_info["is_alive"] = alive
        self.agent.memory.public_info["can_speak"] = alive

    async def night_action(self, context: GameContext) -> ActionResult:
        self.agent.set_context(context)
        return await self.agent.night_action()

    async def speak(self, context: GameContext) -> str:
        self.agent.set_context(context)
        result = await self.agent.speak()
        return result.content or ""

    async def vote(self, context: GameContext) -> str:
        self.agent.set_context(context)
        result = await self.agent.vote()
        return result.target or ""


# ============================================================
# 游戏运行器
# ============================================================

class WerewolfGame:
    """狼人杀游戏"""

    PLAYER_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

    ROLE_DISTRIBUTION = {
        "werewolf": 2,
        "seer": 1,
        "witch": 1,
        "villager": 2,
    }

    def __init__(self, use_smart_llm: bool = True):
        self.logger = GameLogger("runs/logs")
        self.agents: Dict[str, AgentWrapper] = {}
        self.engine: Optional[GameEngine] = None
        self.day = 0
        self.is_game_over = False
        self.winner: Optional[str] = None
        self._day_speeches: Dict[str, str] = {}  # 当天发言记录

        if use_smart_llm:
            self.llm = SmartMockLLM()
        else:
            self.llm = SimpleMockLLM()

        self._init_game()

    def _init_game(self):
        """初始化游戏"""
        print("=" * 60)
        print("AI Werewolf - 全自动狼人杀对战系统")
        print("=" * 60)
        print(f"\n日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"玩家数量: {len(self.PLAYER_NAMES)}")
        print(f"角色配置: {self.ROLE_DISTRIBUTION}")

        # 创建 Agent
        import random
        roles = []
        for role, count in self.ROLE_DISTRIBUTION.items():
            roles.extend([role] * count)
        random.shuffle(roles)

        role_mapping = {}
        for name, role in zip(self.PLAYER_NAMES, roles):
            role_mapping[name] = role

        print("\n角色分配:")
        for name in self.PLAYER_NAMES:
            role = role_mapping[name]
            print(f"  {name}: {role}")

        # 创建 Agent 实例
        agents = create_all_agents(self.PLAYER_NAMES, role_mapping, self.llm)
        self.agents = {name: AgentWrapper(agent) for name, agent in agents.items()}

        # 创建 GameEngine
        self.engine = GameEngine(
            player_names=self.PLAYER_NAMES,
            role_distribution=self.ROLE_DISTRIBUTION,
            log_dir="runs/logs",
            role_mapping=role_mapping,  # 同步角色分配
        )

        # 设置 AI 决策
        self._setup_ai_decisions()

        # 记录游戏开始
        player_info = [{"name": n, "role": role_mapping[n]} for n in self.PLAYER_NAMES]
        self.logger.log_game_start(player_info)

        print("\n游戏初始化完成!")
        print("-" * 60)

    def _setup_ai_decisions(self):
        """设置 AI 决策函数"""

        async def wolf_decision(player, engine):
            context = self._create_context(player.name)
            context.set_private_data({
                "role": "werewolf",
                "teammates": [n for n in self.PLAYER_NAMES if n != player.name and self.agents[n].is_wolf],
            })
            result = await self.agents[player.name].night_action(context)
            if result.action != ActionType.KILL:
                print(f"[DEBUG] wolf {player.name} returned WAIT")
            return result.target

        async def seer_decision(player, engine):
            context = self._create_context(player.name)
            context.set_private_data({
                "role": "seer",
                "checks": self.agents[player.name].seer_checks,
            })
            result = await self.agents[player.name].night_action(context)
            target = result.target
            is_wolf = None
            if target:
                is_wolf = self.agents[target].is_wolf
                self.agents[player.name].agent.memory.private_info.setdefault("checks", {})[target] = is_wolf
            return (target, is_wolf)

        async def witch_heal_decision(player, engine):
            context = self._create_context(player.name)
            context.set_private_data({
                "role": "witch",
                "potions": {"heal": self.agents[player.name].heal_potion, "poison": self.agents[player.name].poison_potion},
                "tonight_victim": engine.night_actions.wolf_kill_target if hasattr(engine, 'night_actions') else None,
            })
            result = await self.agents[player.name].night_action(context)
            return result.target if result.action == ActionType.HEAL else None

        async def poison_decision(player, engine):
            return None

        async def speech(player, engine):
            context = self._create_context(player.name)

            # 共享已收集的发言给当前说话者
            context.set_public_data({
                "speeches": dict(self._day_speeches),
                "dead_players": [p.name for p in self.engine.players.values() if not p.is_alive],
            })

            private_data = {"role": self.agents[player.name].role}
            if self.agents[player.name].is_wolf:
                private_data["teammates"] = self.agents[player.name].wolf_teammates
            elif self.agents[player.name].role == "seer":
                private_data["checks"] = self.agents[player.name].seer_checks
            elif self.agents[player.name].role == "witch":
                private_data["potions"] = {
                    "heal": self.agents[player.name].heal_potion,
                    "poison": self.agents[player.name].poison_potion,
                }
            context.set_private_data(private_data)

            content = await self.agents[player.name].speak(context)

            # 记录发言
            self._day_speeches[player.name] = content
            role = self.agents[player.name].role
            print(f"    [{player.name}]({role}): {content}")

            return content

        async def vote(player, engine):
            context = self._create_context(player.name)

            # 传递所有发言给投票决策
            context.set_public_data({
                "speeches": dict(self._day_speeches),
                "dead_players": [p.name for p in self.engine.players.values() if not p.is_alive],
            })

            context.set_private_data({"role": self.agents[player.name].role})
            return await self.agents[player.name].vote(context)

        self.engine.set_ai_decision_maker("werewolf", wolf_decision)
        self.engine.set_ai_decision_maker("seer", seer_decision)
        self.engine.set_ai_decision_maker("witch", witch_heal_decision)
        self.engine.set_ai_decision_maker("witch_poison", poison_decision)
        self.engine.set_ai_decision_maker("speak", speech)
        self.engine.set_ai_decision_maker("vote", vote)

    def _sync_state(self):
        """同步引擎状态到 Agent 包装器"""
        for name, wrapper in self.agents.items():
            if name in self.engine.players:
                p = self.engine.players[name]
                wrapper.set_alive(p.is_alive)
                # 同步女巫药瓶
                if p.role.value == "witch":
                    priv = wrapper.agent.memory.private_info
                    potions = priv.setdefault("potions", {"heal": 1, "poison": 1})
                    potions["heal"] = p.heal_potion
                    potions["poison"] = p.poison_potion
                # 同步预言家查验记录
                if p.role.value == "seer":
                    priv = wrapper.agent.memory.private_info
                    priv["checks"] = dict(p.seer_checks)

    def _create_context(self, player_name: str) -> GameContext:
        """创建玩家上下文"""
        living = [p.name for p in self.engine.living_players]
        if not living:
            living = self.PLAYER_NAMES.copy()
        return GameContext(player_name, living)

    async def run(self):
        """运行完整游戏"""
        print("\n游戏开始!\n")

        max_days = 15
        day = 0

        while not self.is_game_over and day < max_days:
            day += 1
            self.day = day

            print(f"\n{'='*60}")
            print(f"第 {day} 天")
            print(f"{'='*60}")

            # 夜晚阶段
            print("\n[夜晚] 进入夜晚...")
            self.logger.log_night_start(day)
            night_result = await self.engine.night_phase()
            self._sync_state()
            self._print_night_result(night_result)

            if self._check_win():
                break

            # 白天阶段
            print("\n[白天] 进入白天...\n")
            self._day_speeches = {}  # 重置当天发言记录
            self.logger.log_day_start(day, night_result.dead_players)
            executed = await self.engine.day_phase()
            self._sync_state()
            self._print_day_result(executed)

            if self._check_win():
                break

            print(f"\n当前状态: 存活 {len(self.engine.living_players)} 人", end="")
            wolf_count = len(self.engine.living_wolf_players)
            good_count = len(self.engine.living_good_players)
            print(f" (狼人: {wolf_count}, 好人: {good_count})")

        # 游戏结束
        self._print_game_result()

        log_path = self.logger.save()
        print(f"\n日志已保存: {log_path}")

        return self.winner

    def _print_night_result(self, result: NightActions):
        """打印夜晚结果"""
        print("\n  [夜晚行动]")
        if result.wolf_kill_target:
            print(f"    - 狼人选择杀害: {result.wolf_kill_target}")
        if result.seer_check_target:
            result_str = "狼人" if result.seer_check_result else "好人"
            print(f"    - 预言家查验: {result.seer_check_target} -> {result_str}")
        if result.witch_heal_target:
            print(f"    - 女巫使用解药救: {result.witch_heal_target}")
        if result.witch_poison_target:
            print(f"    - 女巫使用毒药毒: {result.witch_poison_target}")

        if result.dead_players:
            print("\n  [夜晚死亡]")
            for player in result.dead_players:
                cause = result.death_causes.get(player, CauseOfDeath.WOLF_KILL)
                role = self.agents[player].role
                print(f"    - {player} ({role}) - {cause.value}")
        else:
            print("\n  [今晚平安夜，无人死亡]")

    def _print_day_result(self, executed: str):
        """打印白天结果"""
        print("\n  [白天投票]")
        votes = self.engine.night_actions.vote_map
        for name, vote in votes.items():
            print(f"    - {name} 投票给: {vote}")

        if executed:
            role = self.agents[executed].role
            print(f"\n  [处决结果] {executed} ({role}) 被投票处决")
        else:
            print(f"\n  [平票，无人出局]")

    def _check_win(self) -> bool:
        """检查胜负"""
        living_wolves = len(self.engine.living_wolf_players)
        living_goods = len(self.engine.living_good_players)

        if living_wolves == 0:
            self.is_game_over = True
            self.winner = "good"
            self.logger.log_game_over("good", "所有狼人被放逐")
            return True

        if living_wolves >= living_goods:
            self.is_game_over = True
            self.winner = "wolf"
            self.logger.log_game_over("wolf", "狼人数量占优势")
            return True

        return False

    def _print_game_result(self):
        """打印游戏结果"""
        print(f"\n{'='*60}")
        print("游戏结束!")
        print(f"{'='*60}")

        if self.winner == "good":
            print("\n  好人胜利! 狼人全部被放逐，好人获得最终胜利!")
        elif self.winner == "wolf":
            print("\n  狼人胜利! 狼人数量已占优势，狼人获得最终胜利!")
        else:
            print("\n  游戏结束，无结果")

        print("\n[最终状态]")
        for name in self.PLAYER_NAMES:
            agent = self.agents[name]
            p = self.engine.players[name]
            status = "存活" if p.is_alive else "死亡"
            print(f"  {name}: {agent.role} - {status}")


# ============================================================
# 主入口
# ============================================================

async def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="AI Werewolf - 全自动狼人杀")
    parser.add_argument("--smart", action="store_true", help="使用智能模拟LLM")
    parser.add_argument("--simple", action="store_true", help="使用简单模拟LLM")
    args = parser.parse_args()

    use_smart = args.smart or not args.simple

    print("\n" + "=" * 60)
    print("AI Werewolf - 全自动狼人杀多智能体对战系统")
    print("=" * 60)
    print(f"\n模式: {'智能模拟' if use_smart else '简单模拟'} LLM")

    game = WerewolfGame(use_smart_llm=use_smart)
    winner = await game.run()

    print("\n" + "=" * 60)
    result_str = "好人胜利" if winner == 'good' else "狼人胜利" if winner == 'wolf' else "无结果"
    print(f"最终结果: {result_str}")
    print("=" * 60)


if __name__ == "__main__":
    print("""
    ==============================================================
         AI Werewolf - 全自动狼人杀多智能体对战系统

      6人局: 2狼人 + 1预言家 + 1女巫 + 2村民

      直接运行: python main.py
      智能模式: python main.py --smart
    ==============================================================
    """)

    asyncio.run(main())
