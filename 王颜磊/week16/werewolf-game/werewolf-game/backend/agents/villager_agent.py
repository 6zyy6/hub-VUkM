"""
村民智能体 - 信息不对称中最弱势的角色
策略：通过发言分析、投票记录推断狼人身份，配合神职投票
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


class VillagerAgent(BaseAgent):
    def __init__(self, player_id: int, name: str = ""):
        super().__init__(player_id, Role.VILLAGER, name)
        self.team = Team.GOOD
        # 村民的性格维度
        self.analytical: float = random.uniform(0.3, 0.9)  # 分析能力
        self.assertiveness: float = random.uniform(0.2, 0.8)  # 坚定程度
        self.follow_tendency: float = random.uniform(0.3, 0.8)  # 跟票倾向
        # 跟踪投票模式
        self.vote_log: List[dict] = []

    def decide_action(self, game_state: dict, phase: Phase, options: List[dict]) -> dict:
        if phase == Phase.NIGHT:
            return {"action": "wait"}  # 村民夜晚无行动

        elif phase == Phase.VOTE:
            target = self.decide_vote(game_state, options if isinstance(options, list) else [])
            return {"action": "vote", "target_id": target}

        return {"action": "wait"}

    def generate_speech(self, game_state: dict, context: str) -> str:
        """根据性格因子选择发言风格"""
        if self.analytical > 0.7:
            return self._analytical_speech(game_state)
        elif self.assertiveness > 0.7:
            return self._assertive_speech(game_state)
        elif self.follow_tendency > 0.7:
            return self._follower_speech(game_state)
        else:
            return self._balanced_speech(game_state)

    def _analytical_speech(self, game_state: dict) -> str:
        """分析型发言"""
        alive_ids = game_state.get("alive_ids", [])

        # 找出最可疑和最近的行为模式
        suspicious = self.memory.get_suspicious_players(0.5)
        trusted = self.memory.get_trusted_players(0.7)

        parts = [f"玩家{self.player_id}：我来分析一下当前局势。"]

        if suspicious:
            parts.append(f"可疑玩家：{suspicious}，理由包括发言矛盾或投票异常。")
        if trusted:
            parts.append(f"可信玩家：{trusted}，发言逻辑清晰、投票合理。")

        # 检查是否有神职已跳
        claimed_roles = []
        for pid in alive_ids:
            belief = self.memory.beliefs.get(pid, {})
            role = belief.get("likely_role")
            if role:
                claimed_roles.append(f"玩家{pid}可能是{role}")
        if claimed_roles:
            parts.append("可能的身份分布：" + "，".join(claimed_roles[:3]))

        if suspicious:
            parts.append(f"我建议今天投票玩家{suspicious[0]}。")

        return " ".join(parts)

    def _assertive_speech(self, game_state: dict) -> str:
        """坚定型发言"""
        suspicious = self.memory.get_suspicious_players(0.5)
        if suspicious:
            target = suspicious[0]
            return f"玩家{self.player_id}：我坚信玩家{target}是狼！他的发言和投票都有问题。今天我一定要投TA，请大家跟我一起！"
        else:
            return f"玩家{self.player_id}：我是好人村民。虽然还没确定谁是狼，但我相信自己的判断。请大家多发言。"

    def _follower_speech(self, game_state: dict) -> str:
        """跟票型发言"""
        # 寻找已跳的神职
        alive_ids = game_state.get("alive_ids", [])
        for pid in alive_ids:
            belief = self.memory.beliefs.get(pid, {})
            if belief.get("likely_role") in ["预言家", "女巫"]:
                return f"玩家{self.player_id}：我是村民。我选择相信玩家{pid}的判断，今天跟TA投票。"
        return f"玩家{self.player_id}：我是普通村民。有没有大佬出来带一下？我跟票。"

    def _balanced_speech(self, game_state: dict) -> str:
        """均衡型发言"""
        templates = [
            f"玩家{self.player_id}：我是村民，在认真分析。建议大家关注每个人的发言逻辑，寻找矛盾点。",
            f"玩家{self.player_id}：这把好人阵营。我认为应该冷静分析，不要被情绪左右。",
            f"玩家{self.player_id}：我暂时没有确凿的证据，但我注意到有些玩家的行为模式前后不一致。",
            f"玩家{self.player_id}：我是好人。希望大家不要内讧，集中票力投狼。",
        ]
        return random.choice(templates)

    def analyze_message(self, message: AgentMessage) -> dict:
        analysis = {"sender": message.sender_id, "type": message.message_type.value}

        content = message.content
        sender = message.sender_id

        # 神职跳身份
        if "我是预言家" in content:
            analysis["seer_claim"] = True
            self.memory.update_belief(sender, "likely_role", "预言家")
            self.memory.trust_levels[sender] = min(1.0, self.memory.trust_levels.get(sender, 0.5) + 0.25)

            # 关注预言家的查验信息
            if "查验" in content and ("狼人" in content or "好人" in content):
                analysis["seer_check_info"] = True
                # 提取被查的玩家
                for line in content.split("。"):
                    if "玩家" in line and ("狼人" in line or "好人" in line):
                        analysis["check_detail"] = line.strip()

        elif "我是女巫" in content:
            analysis["witch_claim"] = True
            self.memory.trust_levels[sender] = min(1.0, self.memory.trust_levels.get(sender, 0.5) + 0.2)
            self.memory.update_belief(sender, "likely_role", "女巫")

        elif "我是猎人" in content:
            analysis["hunter_claim"] = True
            self.memory.update_belief(sender, "likely_role", "猎人")

        elif "我是守卫" in content:
            analysis["guard_claim"] = True
            self.memory.update_belief(sender, "likely_role", "守卫")

        # 攻击行为分析
        if any(kw in content for kw in ["怀疑玩家", "有问题", "是狼", "投票"]):
            # 提取攻击目标
            for pid in range(1, 13):
                if f"玩家{pid}" in content and "可疑" in content:
                    if pid != self.player_id:
                        # 有人被攻击，观察攻击者的动机
                        analysis[f"suspecting_player_{pid}"] = True

        # 防御行为分析
        if f"玩家{sender}" in content and "好人" in content:
            analysis["self_defense"] = True

        # 投票记录
        if message.message_type == MessageType.VOTE:
            target = message.metadata.get("target_id")
            if target:
                self.vote_log.append({"voter": sender, "target": target})
                analysis["vote_record"] = {"voter": sender, "target": target}

        return analysis

    def update_strategy(self, game_state: dict):
        alive_ids = game_state.get("alive_ids", [])

        # 检查是否有神职已跳
        has_seer = False
        for pid in alive_ids:
            belief = self.memory.beliefs.get(pid, {})
            if belief.get("likely_role") == "预言家":
                has_seer = True
                break

        if has_seer:
            self.memory.strategy = "跟随预言家"
            self.follow_tendency = min(1.0, self.follow_tendency + 0.2)
        else:
            self.memory.strategy = "独立分析"
            self.memory.confidence = 0.6

    def decide_vote(self, game_state: dict, candidates: List[int]) -> int:
        alive_ids = game_state.get("alive_ids", [])
        candidate_set = set(candidates) if candidates else set(alive_ids)

        # 如果有已跳的预言家指定了目标，跟票
        for pid in alive_ids:
            belief = self.memory.beliefs.get(pid, {})
            if belief.get("likely_role") == "预言家" and self.follow_tendency > 0.5:
                # 尝试从预言家最近发言中找到投票建议
                recent = [m for m in self.memory.conversations[-5:] if m.sender_id == pid]
                for msg in recent:
                    for target_id in range(1, 13):
                        if f"投票玩家{target_id}" in msg.content or f"投玩家{target_id}" in msg.content:
                            if target_id in candidate_set:
                                return target_id

        # 优先投最可疑的
        suspicious = self.memory.get_suspicious_players(0.5)
        for pid in suspicious:
            if pid in candidate_set:
                return pid

        return super().decide_vote(game_state, list(candidate_set))
