"""
游戏状态管理模块
维护游戏的完整状态信息，包括玩家状态、回合信息、事件日志等
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
import time


class GamePhase(Enum):
    """游戏阶段"""
    SETUP = "setup"
    NIGHT_WEREWOLF = "night_werewolf"
    NIGHT_SEER = "night_seer"
    NIGHT_WITCH = "night_witch"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTING = "day_voting"
    GAME_OVER = "game_over"


class PlayerStatus(Enum):
    """玩家状态"""
    ALIVE = "alive"
    DEAD = "dead"
    POISONED = "poisoned"
    PROTECTED = "protected"


@dataclass
class PlayerInfo:
    """玩家信息"""
    player_id: int
    role: str
    status: PlayerStatus = PlayerStatus.ALIVE
    is_ai: bool = True
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"Player_{self.player_id}"


@dataclass
class GameEvent:
    """游戏事件"""
    timestamp: float
    phase: GamePhase
    event_type: str
    description: str
    actor_id: Optional[int] = None
    target_id: Optional[int] = None
    details: Dict = field(default_factory=dict)


class GameState:
    """游戏状态管理器"""

    def __init__(self, num_players: int = 12):
        self.num_players = num_players
        self.players: Dict[int, PlayerInfo] = {}
        self.current_round: int = 0
        self.current_phase: GamePhase = GamePhase.SETUP
        self.events: List[GameEvent] = []

        # 夜晚行动记录
        self.werewolf_targets: Set[int] = set()
        self.seer_check_target: Optional[int] = None
        self.seer_check_result: Optional[str] = None
        self.witch_save_used: bool = False
        self.witch_poison_used: bool = False
        self.witch_save_target: Optional[int] = None
        self.witch_poison_target: Optional[int] = None

        # 白天投票
        self.votes: Dict[int, int] = {}
        self.discussion_log: List[Dict] = []

        # 游戏结果
        self.winner: Optional[str] = None
        self.game_over: bool = False

    def add_player(self, player_id: int, role: str, is_ai: bool = True):
        """添加玩家"""
        self.players[player_id] = PlayerInfo(
            player_id=player_id,
            role=role,
            is_ai=is_ai
        )

    def get_alive_players(self) -> List[PlayerInfo]:
        """获取存活玩家"""
        return [p for p in self.players.values() if p.status == PlayerStatus.ALIVE]

    def get_alive_werewolves(self) -> List[PlayerInfo]:
        """获取存活狼人"""
        return [p for p in self.players.values()
                if p.status == PlayerStatus.ALIVE and p.role == "werewolf"]

    def get_alive_villagers(self) -> List[PlayerInfo]:
        """获取存活好人（包括神职）"""
        return [p for p in self.players.values()
                if p.status == PlayerStatus.ALIVE and p.role != "werewolf"]

    def check_game_over(self) -> bool:
        """检查游戏是否结束"""
        alive_werewolves = self.get_alive_werewolves()
        alive_villagers = self.get_alive_villagers()

        if len(alive_werewolves) == 0:
            self.winner = "villagers"
            self.game_over = True
            return True
        elif len(alive_villagers) == 0:
            self.winner = "werewolves"
            self.game_over = True
            return True

        return False

    def add_event(self, phase: GamePhase, event_type: str,
                  description: str, actor_id: Optional[int] = None,
                  target_id: Optional[int] = None, details: Dict = None):
        """添加游戏事件"""
        event = GameEvent(
            timestamp=time.time(),
            phase=phase,
            event_type=event_type,
            description=description,
            actor_id=actor_id,
            target_id=target_id,
            details=details or {}
        )
        self.events.append(event)

    def reset_night_actions(self):
        """重置夜晚行动"""
        self.werewolf_targets.clear()
        self.seer_check_target = None
        self.seer_check_result = None
        self.witch_save_target = None
        self.witch_poison_target = None

    def get_player_by_id(self, player_id: int) -> Optional[PlayerInfo]:
        """根据ID获取玩家"""
        return self.players.get(player_id)

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "current_round": self.current_round,
            "current_phase": self.current_phase.value,
            "players": {
                pid: {
                    "name": p.name,
                    "role": p.role,
                    "status": p.status.value
                }
                for pid, p in self.players.items()
            },
            "winner": self.winner,
            "game_over": self.game_over
        }
