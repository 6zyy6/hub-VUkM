"""
守卫智能体 - 每晚守护一名玩家
策略：优先守护关键神职（预言家/女巫），其次守护自己
"""

import random
from typing import Dict, List, Optional

try:
    from .base_agent import BaseAgent, AgentMessage, MessageType
except ImportError:
    from base_agent import BaseAgent, AgentMessage, MessageType
try:
    from ..game_engine import Role, Team, Phase
except ImportError:
    from game_engine import Role, Team, Phase


class GuardAgent(BaseAgent):
    def __init__(self, player_id: int, name: str = ""):
        super().__init__(player_id, Role.GUARD, name)
        self.team = Team.GOOD
        self.last_protect_target: Optional[int] = None
        self.protection_history: List[int] = []
        self.revealed: bool = False

    def decide_action(self, game_state: dict, phase: Phase, options: List[dict]) -> dict:
        alive_ids = game_state.get("alive_ids", [])

        if phase == Phase.NIGHT:
            target = self._select_protect_target(game_state)
            return {"action": "protect", "target_id": target}

        elif phase == Phase.VOTE:
            return {"action": "vote", "target_id": self.decide_vote(game_state, options)}

        return {"action": "wait"}

    def _select_protect_target(self, game_state: dict) -> Optional[int]:
        """选择守护目标"""
        alive_ids = game_state.get("alive_ids", [])

        # 不能连续守护同一人
        candidates = [pid for pid in alive_ids if pid != self.last_protect_target]

        if not candidates:
            # 如果不能换人，放弃守护
            return None

        # 优先级：
        # 1. 已跳身份的预言家
        # 2. 已跳身份的女巫
        # 3. 自己
        # 4. 其他高信任度玩家

        for pid in candidates:
            belief = self.memory.beliefs.get(pid, {})
            if belief.get("likely_role") in ["预言家"]:
                # 检查是否已跳身份
                recent_msgs = self.memory.conversations[-10:]
                for msg in recent_msgs:
                    if msg.sender_id == pid and "我是预言家" in msg.content:
                        return pid

        for pid in candidates:
            belief = self.memory.beliefs.get(pid, {})
            if belief.get("likely_role") in ["女巫"]:
                return pid

        # 守护自己
        if self.player_id in candidates:
            # 第一夜大概率守自己
            if game_state.get("round", 0) <= 1:
                return self.player_id
            # 后续以一定概率
            if random.random() < 0.4:
                return self.player_id

        # 守护高信任度玩家
        trusted = [(pid, self.memory.trust_levels.get(pid, 0.5))
                   for pid in candidates if pid != self.player_id]
        if trusted:
            trusted.sort(key=lambda x: x[1], reverse=True)
            return trusted[0][0]

        # 随机守护
        if candidates:
            return random.choice(candidates)

        return None

    def set_last_protect(self, target_id: int):
        self.last_protect_target = target_id
        self.protection_history.append(target_id)

    def generate_speech(self, game_state: dict, context: str) -> str:
        if self.should_reveal(game_state):
            return self._reveal_speech(game_state)
        else:
            return self._hidden_speech(game_state, context)

    def should_reveal(self, game_state: dict) -> bool:
        if self.revealed:
            return True

        alive_ids = game_state.get("alive_ids", [])
        if len(alive_ids) <= 4:
            self.revealed = True
            return True

        return False

    def _reveal_speech(self, game_state: dict) -> str:
        parts = [f"玩家{self.player_id}：我是守卫！"]
        if self.protection_history:
            parts.append(f"守护记录：{self.protection_history}")
        parts.append("我会继续配合预言家和女巫守护关键角色。")
        return " ".join(parts)

    def _hidden_speech(self, game_state: dict, context: str) -> str:
        templates = [
            f"玩家{self.player_id}：我是村民。建议大家稳扎稳打，不要被带节奏。",
            f"玩家{self.player_id}：目前的局势对好人有利。我建议继续分析投票记录。",
            f"玩家{self.player_id}：我观察了很多，觉得有些玩家的发言存在矛盾。大家仔细想想。",
            f"玩家{self.player_id}：建议神职在合适的时候出来带节奏，我们配合行动。",
        ]
        return random.choice(templates)

    def analyze_message(self, message: AgentMessage) -> dict:
        analysis = {"sender": message.sender_id, "type": message.message_type.value}

        content = message.content

        if "我是预言家" in content:
            analysis["seer_claim"] = True
            self.memory.update_belief(message.sender_id, "likely_role", "预言家")

        if "我是女巫" in content:
            analysis["witch_claim"] = True
            self.memory.update_belief(message.sender_id, "likely_role", "女巫")

        if "我是守卫" in content and message.sender_id != self.player_id:
            analysis["guard_claim_alert"] = True
            self.memory.suspicion_levels[message.sender_id] = 0.95
            self.memory.update_belief(message.sender_id, "likely_role", "假守卫/狼")

        return analysis

    def update_strategy(self, game_state: dict):
        if self.revealed:
            self.memory.strategy = "已亮身份，明确守护目标"
            self.memory.confidence = 0.75
        else:
            self.memory.strategy = "隐藏身份，暗中守护"
            self.memory.confidence = 0.6

    def decide_vote(self, game_state: dict, candidates: List[int]) -> int:
        return super().decide_vote(game_state, candidates)
