"""
AI Agent 基类 - 所有角色的父类
包含：记忆系统、决策接口、信息隔离
"""

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class ActionType(Enum):
    """行动类型"""
    WAIT = "wait"
    KILL = "kill"
    CHECK = "check"
    HEAL = "heal"
    POISON = "poison"
    SPEAK = "speak"
    VOTE = "vote"


@dataclass
class Memory:
    """Agent 记忆"""
    player_name: str
    role: str

    # 私有信息（只有自己能访问）
    private_info: Dict[str, Any] = field(default_factory=dict)

    # 公共信息（所有 Agent 都可见）
    public_info: Dict[str, Any] = field(default_factory=dict)

    # 历史记录
    speech_history: List[Dict] = field(default_factory=list)
    vote_history: List[Dict] = field(default_factory=list)
    night_action_history: List[Dict] = field(default_factory=list)

    # 推理结论
    suspicions: Dict[str, bool] = field(default_factory=dict)  # player_name -> is_suspicious

    def add_speech(self, content: str, day: int):
        """添加发言记录"""
        self.speech_history.append({
            "day": day,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def add_vote(self, target: str, day: int, reason: str = ""):
        """添加投票记录"""
        self.vote_history.append({
            "day": day,
            "target": target,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    def add_night_action(self, action: str, target: str, day: int):
        """添加夜晚行动记录"""
        self.night_action_history.append({
            "day": day,
            "action": action,
            "target": target,
            "timestamp": datetime.now().isoformat(),
        })

    def update_suspicion(self, player: str, is_suspicious: bool, reason: str = ""):
        """更新对某玩家的怀疑程度"""
        self.suspicions[player] = {
            "is_suspicious": is_suspicious,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }

    def get_private_data(self) -> Dict:
        """获取私有数据（只返回该 Agent 应该知道的信息）"""
        return {
            "role": self.role,
            "private_info": self.private_info,
            "suspicions": self.suspicions,
            "speech_history": self.speech_history[-5:],  # 只返回最近5条
            "vote_history": self.vote_history[-5:],
            "night_action_history": self.night_action_history[-5:],
        }


@dataclass
class ActionResult:
    """行动结果"""
    action: ActionType
    target: Optional[str] = None
    content: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.5


class GameContext:
    """
    游戏上下文 - 封装游戏状态访问
    Agent 只能通过此接口访问游戏状态，实现信息隔离
    """

    def __init__(self, player_name: str, living_players: List[str]):
        self.player_name = player_name
        self.living_players = living_players

        # 这些由 GameEngine 在调用时设置
        self._private_data: Optional[Dict] = None  # 该玩家的私有信息
        self._public_data: Dict = {}  # 公共信息（无敏感内容）

    def set_private_data(self, data: Dict):
        """GameEngine 调用，设置该玩家的私有信息"""
        self._private_data = data

    def set_public_data(self, data: Dict):
        """GameEngine 调用，设置公共信息"""
        self._public_data = data

    @property
    def my_name(self) -> str:
        """我的名字"""
        return self.player_name

    @property
    def alive_players(self) -> List[str]:
        """存活玩家（公开信息）"""
        return self.living_players.copy()

    def i_am(self) -> str:
        """我的身份（私有）"""
        return self._private_data.get("role", "unknown") if self._private_data else "unknown"

    def my_teammates(self) -> List[str]:
        """我的狼人队友（狼人专属）"""
        return self._private_data.get("teammates", []) if self._private_data else []

    def my_checks(self) -> Dict[str, bool]:
        """我的查验记录（预言家专属）"""
        return self._private_data.get("checks", {}) if self._private_data else {}

    def my_potions(self) -> Dict[str, int]:
        """我的药瓶（女巫专属）"""
        potions = self._private_data.get("potions", {}) if self._private_data else {}
        return {
            "heal": potions.get("heal", 0),
            "poison": potions.get("poison", 0),
        }

    def other_players(self) -> List[str]:
        """其他玩家（排除自己）"""
        return [p for p in self.living_players if p != self.player_name]

    def get_others_speech(self, player_name: str) -> Optional[str]:
        """获取某玩家的最近发言（公开信息）"""
        return self._public_data.get("speeches", {}).get(player_name)

    def get_others_vote(self, player_name: str) -> Optional[str]:
        """获取某玩家的最近投票（公开信息）"""
        return self._public_data.get("votes", {}).get(player_name)

    def get_all_speeches(self) -> Dict[str, str]:
        """获取所有玩家的发言记录（公开信息）"""
        return self._public_data.get("speeches", {}).copy()

    def get_dead_players(self) -> List[str]:
        """死亡玩家（公开信息）"""
        return self._public_data.get("dead_players", [])

    def get_recent_events(self) -> List[Dict]:
        """最近事件（公开信息）"""
        return self._public_data.get("recent_events", [])


class BaseAgent(ABC):
    """
    AI Agent 基类

    设计原则：
    1. 每个 Agent 只能访问自己的私有信息和公共信息
    2. 不能直接访问其他 Agent 的私有信息
    3. 通过 GameContext 实现信息隔离
    4. 所有决策必须基于自己的记忆和推理
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm_client: Optional[Any] = None,
    ):
        self.name = name
        self.role = role
        self.llm_client = llm_client

        # 记忆系统
        self.memory = Memory(
            player_name=name,
            role=role,
        )

        # 当前游戏上下文
        self.context: Optional[GameContext] = None

    def set_context(self, context: GameContext):
        """GameEngine 调用，设置游戏上下文"""
        self.context = context

    @abstractmethod
    async def speak(self) -> ActionResult:
        """
        发言行动
        子类必须实现自己的发言策略
        """
        pass

    @abstractmethod
    async def vote(self) -> ActionResult:
        """
        投票行动
        子类必须实现自己的投票策略
        """
        pass

    @abstractmethod
    async def night_action(self) -> ActionResult:
        """
        夜晚行动
        子类必须实现自己的夜晚行动策略
        """
        pass

    def remember_speech(self, content: str, day: int):
        """记录自己的发言"""
        self.memory.add_speech(content, day)

    def remember_vote(self, target: str, day: int, reason: str = ""):
        """记录自己的投票"""
        self.memory.add_vote(target, day, reason)

    def remember_night_action(self, action: str, target: str, day: int):
        """记录自己的夜晚行动"""
        self.memory.add_night_action(action, target, day)

    def update_suspicion(self, player: str, is_suspicious: bool, reason: str = ""):
        """更新对某玩家的怀疑"""
        self.memory.update_suspicion(player, is_suspicious, reason)

    async def think(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        使用 LLM 进行推理
        如果没有 LLM 客户端，返回默认响应
        """
        if self.llm_client:
            return await self.llm_client.generate(prompt, system_prompt)
        return self._default_think(prompt)

    def _default_think(self, prompt: str) -> str:
        """默认推理（当没有 LLM 时）"""
        return f"[{self.name}] {prompt}"

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return f"""你是 {self.name}，身份是 {self.role}。

游戏规则：
- 狼人每晚可以杀害一名玩家
- 预言家每晚可以查验一名玩家的身份
- 女巫有解药和毒药各一瓶
- 白天所有存活玩家轮流发言，然后投票

你的目标：
{self._get_role_objective()}

注意事项：
- 不要暴露自己的真实身份
- 通过发言引导其他玩家
- 基于你的私有信息和公共信息做出决策
"""

    @abstractmethod
    def _get_role_objective(self) -> str:
        """获取角色目标（子类实现）"""
        pass

    def get_decision_context(self) -> str:
        """获取决策上下文（用于 LLM 推理）"""
        my_role = self.role  # Use self.role instead of context.i_am()

        ctx = f"""当前玩家: {self.name}
你的身份: {my_role}
存活玩家: {', '.join(self.context.alive_players) if self.context else ''}
其他玩家: {', '.join(self.context.other_players()) if self.context else ''}
"""
        # 添加角色特定信息
        if my_role == "werewolf":
            ctx += f"狼人队友: {', '.join(self.context.my_teammates()) if self.context else ''}\n"
        elif my_role == "seer":
            checks = self.context.my_checks() if self.context else {}
            if checks:
                check_strs = [f"{k}: {'狼人' if v else '好人'}" for k, v in checks.items()]
                ctx += f"查验记录: {', '.join(check_strs)}\n"
        elif my_role == "witch":
            potions = self.context.my_potions() if self.context else {"heal": 0, "poison": 0}
            ctx += f"解药剩余: {potions['heal']} 瓶\n"
            ctx += f"毒药剩余: {potions['poison']} 瓶\n"

        # 添加推测
        if self.memory.suspicions:
            ctx += "你的推测:\n"
            for player, info in self.memory.suspicions.items():
                if info.get("is_suspicious"):
                    reason = info.get('reason', '')
                    ctx += f"  - {player}: 可疑 ({reason})\n"

        return ctx

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, role={self.role})"