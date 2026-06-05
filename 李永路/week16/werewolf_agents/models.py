"""
基础数据模型定义
包含角色、玩家、游戏状态等核心数据结构
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime


class RoleType(Enum):
    """角色类型枚举"""
    WEREWOLF = "werewolf"  # 狼人
    VILLAGER = "villager"  # 村民
    SEER = "seer"  # 预言家
    WITCH = "witch"  # 女巫
    HUNTER = "hunter"  # 猎人
    GUARD = "guard"  # 守卫
    IDIOT = "idiot"  # 白痴


class PlayerStatus(Enum):
    """玩家状态"""
    ALIVE = "alive"
    DEAD = "dead"
    POISONED = "poisoned"  # 被毒
    PROTECTED = "protected"  # 被守护


class GamePhase(Enum):
    """游戏阶段"""
    NIGHT = "night"
    DAY = "day"
    DISCUSSION = "discussion"
    VOTING = "voting"
    GAME_OVER = "game_over"


@dataclass
class Player:
    """玩家数据类"""
    player_id: int
    name: str
    role: RoleType
    status: PlayerStatus = PlayerStatus.ALIVE
    is_ai: bool = True  # 是否为 AI 玩家
    
    # 游戏信息（根据角色可见性不同）
    known_roles: Dict[int, RoleType] = field(default_factory=dict)  # 已知其他玩家的角色
    night_actions: List[str] = field(default_factory=list)  # 夜间行动记录
    day_actions: List[str] = field(default_factory=list)  # 白天行动记录
    
    def is_alive(self) -> bool:
        return self.status == PlayerStatus.ALIVE
    
    def die(self):
        self.status = PlayerStatus.DEAD
    
    def add_known_role(self, player_id: int, role: RoleType):
        """添加已知角色信息"""
        self.known_roles[player_id] = role


@dataclass
class NightAction:
    """夜间行动记录"""
    actor_id: int
    target_id: Optional[int]
    action_type: str  # kill, save, poison, protect, verify
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DayAction:
    """白天行动记录"""
    speaker_id: int
    content: str
    action_type: str = "speech"  # speech, vote
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GameState:
    """游戏状态"""
    game_id: str
    players: Dict[int, Player] = field(default_factory=dict)
    current_phase: GamePhase = GamePhase.NIGHT
    current_round: int = 0
    day_count: int = 0
    
    # 夜间行动
    night_actions: List[NightAction] = field(default_factory=list)
    
    # 白天讨论和投票
    discussion_log: List[DayAction] = field(default_factory=list)
    votes: Dict[int, int] = field(default_factory=dict)  # voter_id -> target_id
    
    # 特殊状态
    witch_poison_used: bool = False
    witch_antidote_used: bool = False
    guard_last_protected: Optional[int] = None  # 守卫昨晚保护的人（不能连续保护）
    
    # 游戏结果
    winner: Optional[str] = None  # "werewolf" or "villager"
    game_log: List[str] = field(default_factory=list)
    
    def get_alive_players(self) -> List[Player]:
        """获取存活玩家"""
        return [p for p in self.players.values() if p.is_alive()]
    
    def get_alive_werewolves(self) -> List[Player]:
        """获取存活狼人"""
        return [p for p in self.players.values() 
                if p.is_alive() and p.role == RoleType.WEREWOLF]
    
    def get_alive_villagers(self) -> List[Player]:
        """获取存活好人（包括神职）"""
        return [p for p in self.players.values() 
                if p.is_alive() and p.role != RoleType.WEREWOLF]
    
    def check_game_over(self) -> bool:
        """检查游戏是否结束"""
        alive_werewolves = self.get_alive_werewolves()
        alive_villagers = self.get_alive_villagers()
        
        if len(alive_werewolves) == 0:
            self.winner = "villager"
            self.game_log.append(f"游戏结束！好人阵营获胜！")
            return True
        
        if len(alive_werewolves) >= len(alive_villagers):
            self.winner = "werewolf"
            self.game_log.append(f"游戏结束！狼人阵营获胜！")
            return True
        
        return False
    
    def add_log(self, message: str):
        """添加游戏日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.game_log.append(log_entry)
        print(log_entry)
