"""基础 Agent 类"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from ..llm.base import LLMBase


class Role(Enum):
    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"


@dataclass
class AgentState:
    """Agent 状态"""
    name: str
    role: Role
    is_alive: bool = True
    can_speak: bool = True
    vote_count: int = 0
    last_word: Optional[str] = None


class BaseAgent(ABC):
    """AI Agent 基类"""

    def __init__(
        self,
        name: str,
        role: Role,
        llm: LLMBase,
        log_dir: str = "runs/logs",
    ):
        self.name = name
        self.role = role
        self.llm = llm
        self.log_dir = log_dir
        self.state = AgentState(name=name, role=role)

    @abstractmethod
    async def night_phase(self, game_context: dict) -> dict:
        """夜晚阶段行为"""
        pass

    @abstractmethod
    async def day_phase(self, game_context: dict) -> dict:
        """白天阶段行为"""
        pass

    @abstractmethod
    async def vote(self, game_context: dict) -> str:
        """投票"""
        pass

    async def speak(self, prompt: str) -> str:
        """发言"""
        response = await self.llm.generate(prompt)
        self.state.last_word = response
        return response

    def die(self, cause: str = "vote"):
        """死亡"""
        self.state.is_alive = False
        self.state.can_speak = False

    def reset_vote_count(self):
        """重置票数"""
        self.state.vote_count = 0

    @property
    def identity(self) -> str:
        """身份描述"""
        return self.role.value

    @property
    def is_wolf(self) -> bool:
        """是否是狼人"""
        return self.role == Role.WEREWOLF

    @property
    def is_good(self) -> bool:
        """是否是好人"""
        return not self.is_wolf