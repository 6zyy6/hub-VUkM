"""游戏阶段管理"""
from enum import Enum


class GamePhase(Enum):
    """游戏阶段枚举"""
    WAITING = "waiting"
    NIGHT = "night"
    DAY = "day"
    VOTE = "vote"
    END = "end"


class Phase:
    """阶段管理"""

    def __init__(self):
        self.current = GamePhase.WAITING
        self.day_number: int = 0

    def set_phase(self, phase: GamePhase):
        """设置当前阶段"""
        self.current = phase
        if phase == GamePhase.DAY:
            self.day_number += 1

    def next_phase(self):
        """进入下一阶段"""
        if self.current == GamePhase.WAITING:
            self.current = GamePhase.NIGHT
        elif self.current == GamePhase.NIGHT:
            self.current = GamePhase.DAY
        elif self.current == GamePhase.DAY:
            self.current = GamePhase.VOTE
        elif self.current == GamePhase.VOTE:
            self.current = GamePhase.NIGHT

    @property
    def is_night(self) -> bool:
        return self.current == GamePhase.NIGHT

    @property
    def is_day(self) -> bool:
        return self.current in (GamePhase.DAY, GamePhase.VOTE)