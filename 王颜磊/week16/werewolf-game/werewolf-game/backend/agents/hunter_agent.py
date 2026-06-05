"""
猎人智能体 - 死亡时带走一人的强力神职
策略：隐藏身份，死亡时精准带走最可疑的玩家
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


class HunterAgent(BaseAgent):
    def __init__(self, player_id: int, name: str = ""):
        super().__init__(player_id, Role.HUNTER, name)
        self.team = Team.GOOD
        self.can_shoot: bool = True
        self.revealed: bool = False
        self.shoot_target: Optional[int] = None
        # 策略参数
        self.aggressiveness: float = random.uniform(0.5, 0.9)

    def decide_action(self, game_state: dict, phase: Phase, options: List[dict]) -> dict:
        if phase == Phase.NIGHT:
            return {"action": "wait"}  # 猎人夜晚无行动

        elif phase == Phase.VOTE:
            return {"action": "vote", "target_id": self.decide_vote(game_state, options)}

        return {"action": "wait"}

    def decide_shoot(self, game_state: dict) -> Optional[int]:
        """死亡时决定开枪目标"""
        if not self.can_shoot:
            return None

        alive_ids = game_state.get("alive_ids", [])

        # 评估所有存活玩家的可疑度
        candidates = [pid for pid in alive_ids if pid != self.player_id]
        if not candidates:
            return None

        # 计算威胁/可疑度评分
        scores = {}
        for pid in candidates:
            score = 0.0
            belief = self.memory.beliefs.get(pid, {})

            # 确认的狼人
            if belief.get("confirmed") == "狼人":
                score += 100
            # 高度可疑
            suspicion = self.memory.suspicion_levels.get(pid, 0.0)
            score += suspicion * 50
            # 可能是狼
            if belief.get("likely_role") in ["狼人", "悍跳狼"]:
                score += 80
            # 信任度低
            trust = self.memory.trust_levels.get(pid, 0.5)
            score += (1 - trust) * 30
            # 随机因素
            score += random.uniform(-5, 5)

            scores[pid] = score

        if scores:
            self.shoot_target = max(scores, key=scores.get)
            self.can_shoot = False
            return self.shoot_target

        return None

    def generate_speech(self, game_state: dict, context: str) -> str:
        if self.should_reveal(game_state):
            return self._reveal_speech(game_state)
        else:
            return self._hidden_speech(game_state, context)

    def should_reveal(self, game_state: dict) -> bool:
        if self.revealed:
            return True

        alive_ids = game_state.get("alive_ids", [])
        # 危急时刻跳身份
        if len(alive_ids) <= 4:
            self.revealed = True
            return True

        return False

    def _reveal_speech(self, game_state: dict) -> str:
        parts = [f"玩家{self.player_id}：我是猎人！"]
        parts.append("谁敢动我，我就带谁走。")
        parts.append("建议狼人别来刀我，否则你们会后悔的。")
        return " ".join(parts)

    def _hidden_speech(self, game_state: dict, context: str) -> str:
        templates = [
            f"玩家{self.player_id}：我是普通村民，但我有在认真分析。建议大家关注投票记录。",
            f"玩家{self.player_id}：我觉得局势还算明朗。好人阵营加油！",
            f"玩家{self.player_id}：我建议大家不要被带节奏。理性分析每个人的发言。",
            f"玩家{self.player_id}：这把好人优势。我建议稳扎稳打，不要冒进。",
        ]
        return random.choice(templates)

    def analyze_message(self, message: AgentMessage) -> dict:
        analysis = {"sender": message.sender_id, "type": message.message_type.value}

        content = message.content

        # 有人跳预言家
        if "我是预言家" in content:
            analysis["seer_claim"] = True
            self.memory.trust_levels[message.sender_id] = min(1.0, self.memory.trust_levels.get(message.sender_id, 0.5) + 0.15)

        # 有人跳女巫
        if "我是女巫" in content:
            analysis["witch_claim"] = True
            self.memory.trust_levels[message.sender_id] = min(1.0, self.memory.trust_levels.get(message.sender_id, 0.5) + 0.15)

        # 有人跳猎人（可能是狼）
        if "我是猎人" in content and message.sender_id != self.player_id:
            analysis["hunter_claim_alert"] = True
            self.memory.suspicion_levels[message.sender_id] = 0.95
            self.memory.update_belief(message.sender_id, "likely_role", "假猎人/狼")

        # 有人攻击我
        if f"玩家{self.player_id}" in content and any(kw in content for kw in ["可疑", "有问题", "狼", "投"]):
            analysis["attacking_me"] = True
            self.memory.suspicion_levels[message.sender_id] = min(1.0, self.memory.suspicion_levels.get(message.sender_id, 0.5) + 0.25)

        return analysis

    def update_strategy(self, game_state: dict):
        if self.revealed:
            self.memory.strategy = "已亮身份，威慑狼人"
            self.memory.confidence = 0.9
        else:
            self.memory.strategy = "隐藏身份，等待时机"
            self.memory.confidence = 0.7

    def decide_vote(self, game_state: dict, candidates: List[int]) -> int:
        return super().decide_vote(game_state, candidates)
