"""
基础智能体抽象类
所有角色Agent的基类，定义通用接口与通信机制
"""

import abc
import json
import random
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

try:
    from ..game_engine import Role, Team, Phase, GameEvent, PlayerState
except ImportError:
    from game_engine import Role, Team, Phase, GameEvent, PlayerState


class MessageType(Enum):
    """消息类型枚举"""
    SYSTEM = "system"          # 系统消息
    PUBLIC = "public"          # 公开发言
    PRIVATE = "private"        # 私聊（狼人/情侣）
    ACTION = "action"          # 行动指令
    VOTE = "vote"              # 投票
    OBSERVATION = "observation" # 观察结果
    STRATEGY = "strategy"      # 策略讨论
    EMOTION = "emotion"        # 情绪表达


@dataclass
class AgentMessage:
    """智能体消息结构"""
    sender_id: int
    message_type: MessageType
    content: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    target_ids: List[int] = field(default_factory=list)  # 接收者列表，空表示广播
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sender": self.sender_id,
            "type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "targets": self.target_ids,
            "metadata": self.metadata
        }


@dataclass
class AgentMemory:
    """智能体记忆单元"""
    player_id: int
    role: Role
    # 记忆存储
    observations: List[Dict[str, Any]] = field(default_factory=list)
    conversations: List[AgentMessage] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    # 信念系统
    beliefs: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # 对其他玩家的信念
    strategy: str = ""
    # 情绪状态
    confidence: float = 0.5  # 自信度 0-1
    trust_levels: Dict[int, float] = field(default_factory=dict)  # 对其他玩家的信任度
    suspicion_levels: Dict[int, float] = field(default_factory=dict)  # 怀疑度

    def add_observation(self, event: dict):
        self.observations.append({
            "timestamp": datetime.now().timestamp(),
            "event": event
        })

    def add_conversation(self, msg: AgentMessage):
        self.conversations.append(msg)

    def update_belief(self, player_id: int, key: str, value: Any):
        if player_id not in self.beliefs:
            self.beliefs[player_id] = {}
        self.beliefs[player_id][key] = value

    def get_belief(self, player_id: int, key: str, default=None) -> Any:
        return self.beliefs.get(player_id, {}).get(key, default)

    def get_suspicious_players(self, threshold: float = 0.6) -> List[int]:
        """获取高度可疑的玩家"""
        return [pid for pid, level in self.suspicion_levels.items() if level >= threshold]

    def get_trusted_players(self, threshold: float = 0.7) -> List[int]:
        """获取高度信任的玩家"""
        return [pid for pid, level in self.trust_levels.items() if level >= threshold]

    def to_summary(self) -> dict:
        return {
            "player_id": self.player_id,
            "role": self.role.value,
            "observations_count": len(self.observations),
            "conversations_count": len(self.conversations),
            "beliefs_count": len(self.beliefs),
            "confidence": self.confidence,
            "strategy": self.strategy[:100] + "..." if len(self.strategy) > 100 else self.strategy
        }


class BaseAgent(abc.ABC):
    """智能体基类"""

    def __init__(self, player_id: int, role: Role, name: str = ""):
        self.player_id = player_id
        self.role = role
        self.team = Team.EVIL if role == Role.WEREWOLF else Team.GOOD
        self.name = name or f"Agent_{player_id}"
        self.alive = True
        self.memory = AgentMemory(player_id, role)

        # 通信接口
        self.message_queue: List[AgentMessage] = []
        self.send_message_callback: Optional[Callable[[AgentMessage], None]] = None

        # 状态
        self.is_sheriff = False
        self.vote_weight = 1

        # 初始化信任/怀疑度
        for i in range(1, 13):
            if i != player_id:
                self.memory.trust_levels[i] = random.uniform(0.3, 0.7)
                self.memory.suspicion_levels[i] = random.uniform(0.2, 0.5)

    # ========== 抽象方法 ==========

    @abc.abstractmethod
    def decide_action(self, game_state: dict, phase: Phase, options: List[dict]) -> dict:
        """根据游戏状态决定行动"""
        pass

    @abc.abstractmethod
    def generate_speech(self, game_state: dict, context: str) -> str:
        """生成发言内容"""
        pass

    @abc.abstractmethod
    def analyze_message(self, message: AgentMessage) -> dict:
        """分析接收到的消息"""
        pass

    @abc.abstractmethod
    def update_strategy(self, game_state: dict):
        """更新策略"""
        pass

    # ========== 通用方法 ==========

    def set_sheriff(self, is_sheriff: bool):
        self.is_sheriff = is_sheriff
        self.vote_weight = 2 if is_sheriff else 1

    def receive_message(self, message: AgentMessage):
        """接收消息"""
        self.message_queue.append(message)
        self.memory.add_conversation(message)

        # 分析消息
        analysis = self.analyze_message(message)
        if analysis:
            self.memory.add_observation(analysis)

    def get_messages(self, message_type: Optional[MessageType] = None) -> List[AgentMessage]:
        """获取消息队列"""
        if message_type:
            return [msg for msg in self.message_queue if msg.message_type == message_type]
        return self.message_queue.copy()

    def clear_messages(self):
        self.message_queue.clear()

    def send_message(self, content: str, msg_type: MessageType,
                     target_ids: Optional[List[int]] = None, metadata: dict = None):
        """发送消息"""
        if not self.alive:
            return

        msg = AgentMessage(
            sender_id=self.player_id,
            message_type=msg_type,
            content=content,
            target_ids=target_ids or [],
            metadata=metadata or {}
        )

        if self.send_message_callback:
            self.send_message_callback(msg)
        else:
            # 本地存储
            self.memory.add_conversation(msg)

    def observe_game_event(self, event: dict):
        """观察游戏事件"""
        self.memory.add_observation(event)

        # 根据事件更新信念
        event_type = event.get("type")
        if event_type == "vote_cast":
            voter = event.get("voter_id")
            target = event.get("target_id")
            if voter and target:
                # 投票行为分析
                if voter == self.player_id:
                    return
                if target == self.player_id:
                    # 有人投我，增加怀疑度
                    self.memory.suspicion_levels[voter] = min(1.0, self.memory.suspicion_levels.get(voter, 0.5) + 0.2)
                else:
                    # 分析投票模式
                    pass

        elif event_type == "player_eliminated":
            eliminated = event.get("player_id")
            role = event.get("role")
            if eliminated and role:
                # 更新对已出局玩家的信念
                self.memory.update_belief(eliminated, "confirmed_role", role)
                if role == "狼人":
                    # 狼人出局，好人阵营信任度提升
                    for pid in self.memory.trust_levels:
                        if pid != eliminated:
                            self.memory.trust_levels[pid] = min(1.0, self.memory.trust_levels.get(pid, 0.5) + 0.1)

    def decide_vote(self, game_state: dict, candidates: List[int]) -> int:
        """决定投票目标"""
        if not self.alive:
            return 0

        # 默认策略：投给最可疑的玩家
        suspicious = self.memory.get_suspicious_players(0.5)
        alive_candidates = [pid for pid in candidates if pid in game_state.get("alive_ids", [])]

        if suspicious and alive_candidates:
            # 优先投可疑的候选人
            intersect = [pid for pid in suspicious if pid in alive_candidates]
            if intersect:
                return random.choice(intersect)

        # 随机投一个活着的候选人
        if alive_candidates:
            return random.choice(alive_candidates)

        return 0  # 弃票

    def get_belief_summary(self) -> dict:
        """获取信念摘要"""
        return {
            "player_id": self.player_id,
            "role": self.role.value,
            "alive": self.alive,
            "confidence": self.memory.confidence,
            "suspicious_players": self.memory.get_suspicious_players(0.6),
            "trusted_players": self.memory.get_trusted_players(0.7),
            "strategy": self.memory.strategy[:200] if self.memory.strategy else "未定"
        }

    def to_dict(self) -> dict:
        return {
            "id": self.player_id,
            "name": self.name,
            "role": self.role.value,
            "team": self.team.value,
            "alive": self.alive,
            "is_sheriff": self.is_sheriff,
            "memory_summary": self.memory.to_summary()
        }


class AgentFactory:
    """智能体工厂"""

    @staticmethod
    def create_agent(player_id: int, role: Role, name: str = "") -> BaseAgent:
        """创建对应角色的智能体"""
        if role == Role.WEREWOLF:
            from .werewolf_agent import WerewolfAgent
            return WerewolfAgent(player_id, name)
        elif role == Role.SEER:
            from .seer_agent import SeerAgent
            return SeerAgent(player_id, name)
        elif role == Role.WITCH:
            from .witch_agent import WitchAgent
            return WitchAgent(player_id, name)
        elif role == Role.HUNTER:
            from .hunter_agent import HunterAgent
            return HunterAgent(player_id, name)
        elif role == Role.GUARD:
            from .guard_agent import GuardAgent
            return GuardAgent(player_id, name)
        else:  # VILLAGER
            from .villager_agent import VillagerAgent
            return VillagerAgent(player_id, name)
