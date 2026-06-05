"""游戏主引擎"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime

from ..agents.base import BaseAgent, Role
from .phase import Phase, GamePhase
from .rule import GameRule
from ..logger.game_logger import GameLogger


@dataclass
class Player:
    """玩家"""
    name: str
    agent: BaseAgent
    role: Role
    is_alive: bool = True
    can_speak: bool = True
    vote_count: int = 0
    last_word: Optional[str] = None


@dataclass
class GameConfig:
    """游戏配置"""
    player_names: List[str]
    role_distribution: Dict[Role, int] = field(default_factory=lambda: {
        Role.WEREWOLF: 2,
        Role.SEER: 1,
        Role.WITCH: 1,
        Role.HUNTER: 1,
    })
    log_dir: str = "runs/logs"


class WerewolfGame:
    """狼人杀游戏引擎"""

    def __init__(self, config: GameConfig, agents: List[BaseAgent]):
        self.config = config
        self.players: List[Player] = []
        self.phase = Phase()
        self.rule = GameRule()
        self.logger = GameLogger(config.log_dir)
        self.wolf_kill_target: Optional[str] = None
        self.seer_check_result: Dict[str, bool] = {}  # True=狼人

        self._init_players(agents)

    def _init_players(self, agents: List[BaseAgent]):
        """初始化玩家"""
        for name, agent in zip(self.config.player_names, agents):
            player = Player(
                name=name,
                agent=agent,
                role=agent.role,
            )
            self.players.append(player)

        self.logger.log_game_start([p.name for p in self.players])

    @property
    def living_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive]

    @property
    def living_good_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive and not self._is_wolf(p)]

    @property
    def werewolf_team(self) -> List[Player]:
        return [p for p in self.players if self._is_wolf(p)]

    def _is_wolf(self, player: Player) -> bool:
        return player.role == Role.WEREWOLF

    def get_context(self) -> Dict:
        """获取游戏上下文"""
        return {
            "living_players": [p.name for p in self.living_players],
            "living_good_players": [p.name for p in self.living_good_players],
            "werewolf_teammates": [p.name for p in self.werewolf_team],
            "wolf_kill_target": self.wolf_kill_target,
            "seer_checks": self.seer_check_result,
            "current_phase": self.phase.current.name,
        }

    async def night_phase(self) -> Dict:
        """夜晚阶段"""
        self.phase.set_phase(GamePhase.NIGHT)
        self.logger.log_phase("夜晚")

        actions = {}

        # 重置女巫状态
        for p in self.players:
            if p.role == Role.WITCH and p.is_alive:
                p.agent.new_night()

        # 狼人行动
        wolves = [p for p in self.living_players if self._is_wolf(p)]
        for wolf in wolves:
            if wolf.can_speak:
                result = await wolf.agent.night_phase(self.get_context())
                if result.get("action") == "kill":
                    self.wolf_kill_target = result.get("target")
                actions[wolf.name] = result

        self.logger.log_wolf_action(actions)

        # 预言家行动
        seers = [p for p in self.living_players if p.role == Role.SEER]
        for seer in seers:
            if seer.can_speak:
                result = await seer.agent.night_phase(self.get_context())
                if result.get("action") == "check":
                    target = result.get("target")
                    is_wolf = target in [p.name for p in self.werewolf_team]
                    self.seer_check_result[target] = is_wolf
                    seer.agent.receive_check_result(target, is_wolf)
                actions[seer.name] = result

        self.logger.log_seer_action(actions)

        # 女巫行动
        witches = [p for p in self.living_players if p.role == Role.WITCH]
        for witch in witches:
            if witch.can_speak:
                result = await witch.agent.night_phase(self.get_context())
                if result.get("action") == "heal":
                    self.wolf_kill_target = None  # 救人成功
                actions[witch.name] = result

        self.logger.log_witch_action(actions)

        return actions

    async def day_phase(self) -> str:
        """白天阶段 - 发言和投票"""
        self.phase.set_phase(GamePhase.DAY)

        # 检查狼人是否杀人
        victim = self.wolf_kill_target
        dead_players = []

        if victim:
            victim_player = self._get_player(victim)
            if victim_player:
                victim_player.is_alive = False
                victim_player.can_speak = False
                dead_players.append(victim)
                self.logger.log_death(victim, "wolf")

        # 女巫毒人
        for p in self.players:
            if p.role == Role.WITCH and p.is_alive:
                witch_actions = await asyncio.sleep(0)  # placeholder

        # 猎人开枪
        for p in self.players:
            if p.role == Role.HUNTER and not p.is_alive and p.can_shoot:
                target = await p.agent.on_death(self.get_context())
                if target:
                    t = self._get_player(target)
                    if t:
                        t.is_alive = False
                        dead_players.append(target)
                        self.logger.log_death(target, "hunter")

        self.logger.log_phase("白天")

        # 发言阶段
        speeches = {}
        for player in self.living_players:
            if player.can_speak:
                result = await player.agent.day_phase(self.get_context())
                speeches[player.name] = result.get("content", "")
                self.logger.log_speech(player.name, result.get("content", ""))

        # 投票阶段
        votes = {}
        for player in self.living_players:
            if player.can_speak:
                vote_target = await player.agent.vote(self.get_context())
                votes[player.name] = vote_target

        self.logger.log_vote(votes)

        # 统计票数
        for vote_target in votes.values():
            target_player = self._get_player(vote_target)
            if target_player:
                target_player.vote_count += 1

        # 找出被投出的人
        max_votes = max((p.vote_count for p in self.living_players), default=0)
        eliminated = [p for p in self.living_players if p.vote_count == max_votes]

        result = ""
        if len(eliminated) == 1:
            eliminated[0].is_alive = False
            eliminated[0].can_speak = False
            result = eliminated[0].name
            self.logger.log_death(result, "vote")
        else:
            # 平票，流局
            result = "draw"
            self.logger.log_event("平票，无人出局")

        # 重置票数
        for p in self.players:
            p.vote_count = 0

        return result

    def _get_player(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None

    def check_win_condition(self) -> Optional[str]:
        """检查胜利条件"""
        living_wolves = [p for p in self.living_players if self._is_wolf(p)]
        living_goods = [p for p in self.living_players if not self._is_wolf(p)]

        if len(living_wolves) == 0:
            return "good"  # 好人胜利
        if len(living_wolves) >= len(living_goods):
            return "wolf"  # 狼人胜利
        return None

    async def run(self) -> str:
        """运行完整游戏"""
        day_count = 0

        while True:
            # 夜晚
            await self.night_phase()

            # 检查胜利
            winner = self.check_win_condition()
            if winner:
                self.logger.log_win(winner)
                break

            # 白天
            day_count += 1
            result = await self.day_phase()

            if result == "draw":
                continue

            if result:
                dead = self._get_player(result)
                if dead:
                    # 猎人死亡
                    if dead.role == Role.HUNTER:
                        target = await dead.agent.on_death(self.get_context())
                        if target:
                            t = self._get_player(target)
                            if t:
                                t.is_alive = False
                                self.logger.log_death(target, "hunter")

            # 检查胜利
            winner = self.check_win_condition()
            if winner:
                self.logger.log_win(winner)
                break

            if day_count > 20:  # 防止无限循环
                break

        self.logger.log_game_end(winner or "draw")
        return winner or "draw"