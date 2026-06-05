"""
玩家基类模块
定义玩家的基本属性和行为接口
"""

from typing import Optional, List, Dict
from abc import ABC, abstractmethod


class BasePlayer(ABC):
    """玩家基类"""

    def __init__(self, player_id: int, name: str, role: str, is_ai: bool = True):
        self.player_id = player_id
        self.name = name
        self.role = role
        self.is_ai = is_ai
        self.is_alive = True
        self.memory: List[Dict] = []

    def add_memory(self, event: Dict):
        """添加记忆"""
        self.memory.append(event)

    def get_recent_memories(self, n: int = 10) -> List[Dict]:
        """获取最近的记忆"""
        return self.memory[-n:]

    @abstractmethod
    def night_action(self, game_state: Dict, teammates: List[int] = None) -> Dict:
        """夜间行动"""
        pass

    @abstractmethod
    def day_speech(self, game_state: Dict, speech_history: List[Dict]) -> str:
        """白天发言"""
        pass

    @abstractmethod
    def voting_decision(self, game_state: Dict, candidates: List[int]) -> int:
        """投票决策"""
        pass

    def on_death(self, game_state: Dict) -> Optional[Dict]:
        """死亡时的处理"""
        return None

    def __repr__(self):
        return f"Player(id={self.player_id}, name={self.name}, role={self.role}, alive={self.is_alive})"
