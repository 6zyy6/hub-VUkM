"""
狼人智能体 - 信息不对称博弈中的欺诈者
策略：隐藏身份、伪造发言、统一口径、夜间集火
"""

import random
from typing import Dict, List, Optional

try:
    from .base_agent import BaseAgent, AgentMessage, MessageType, AgentMemory
except ImportError:
    from base_agent import BaseAgent, AgentMessage, MessageType, AgentMemory
try:
    from ..game_engine import Role, Team, Phase, GameEvent
except ImportError:
    from game_engine import Role, Team, Phase, GameEvent


class WerewolfAgent(BaseAgent):
    def __init__(self, player_id: int, name: str = ""):
        super().__init__(player_id, Role.WEREWOLF, name)
        self.team = Team.EVIL
        self.wolf_partners: List[int] = []  # 狼队友
        self.fake_claim: str = ""           # 假身份声明
        self.night_kill_preference: List[int] = []  # 刀人优先级
        # 策略因子
        self.aggression: float = random.uniform(0.3, 0.9)  # 攻击性
        self.concealment: float = random.uniform(0.5, 0.95)  # 隐蔽性
        self.agitation: float = random.uniform(0.2, 0.8)    # 煽动性

    def set_wolf_partners(self, partners: List[int]):
        self.wolf_partners = partners
        # 狼队友天然高度信任
        for pid in partners:
            self.memory.trust_levels[pid] = 0.95
            self.memory.suspicion_levels[pid] = 0.05

    def decide_action(self, game_state: dict, phase: Phase, options: List[dict]) -> dict:
        alive_ids = game_state.get("alive_ids", [])

        if phase == Phase.NIGHT:
            # 夜晚刀人决策
            candidates = [pid for pid in alive_ids if pid != self.player_id and pid not in self.wolf_partners]

            # 优先级策略：
            # 1. 优先刀疑似神职的玩家
            # 2. 其次刀高信任度玩家
            # 3. 遵循团队统一目标

            # 从信念中评估各玩家威胁等级
            threat_scores = {}
            for pid in candidates:
                score = 0.0
                belief = self.memory.beliefs.get(pid, {})

                # 可能是预言家的玩家威胁最大
                if belief.get("likely_role") in ["预言家", "女巫", "猎人", "守卫"]:
                    score += 10
                elif belief.get("likely_role") == "村民":
                    score += 2

                # 发言中表现出洞察力的
                if self.memory.suspicion_levels.get(pid, 0.3) < 0.3:
                    score += 3

                # 信任度高（可能被女巫救）
                if self.memory.trust_levels.get(pid, 0.5) > 0.7:
                    score += 2

                # 加入随机因素
                score += random.uniform(-2, 2)
                threat_scores[pid] = score

            # 选威胁最高的
            if threat_scores:
                target = max(threat_scores, key=threat_scores.get)
                reason = f"威胁评分: {threat_scores[target]:.1f}"
                return {"action": "kill", "target_id": target, "reason": reason}

        elif phase == Phase.VOTE:
            return {"action": "vote", "target_id": self.decide_vote(game_state, options)}

        return {"action": "wait"}

    def generate_speech(self, game_state: dict, context: str) -> str:
        speech_templates = [
            self._generate_calm_speech,
            self._generate_aggressive_speech,
            self._generate_confused_speech,
            self._generate_analytic_speech,
            self._generate_cooperative_speech,
        ]

        # 根据性格因子选择发言风格
        if self.aggression > 0.7:
            template = self._generate_aggressive_speech
        elif self.concealment > 0.8:
            template = self._generate_calm_speech
        elif self.agitation > 0.7:
            template = self._generate_confused_speech
        else:
            template = random.choice(speech_templates)

        return template(game_state, context)

    def _generate_calm_speech(self, game_state: dict, context: str) -> str:
        templates = [
            f"玩家{self.player_id}：我听着大家的发言，目前没有特别明确的信息。建议今天稳扎稳打，不要盲目投票。",
            f"玩家{self.player_id}：我暂时没有确认身份的信息，但我建议大家理性分析，不要被个别发言带偏。",
            f"玩家{self.player_id}：我是普通村民，这把局势还不明朗。我建议大家多发言交流，不要急着投票。",
            f"玩家{self.player_id}：我觉得目前的线索还不够充分。谁是狼还需要进一步观察。",
        ]
        return random.choice(templates)

    def _generate_aggressive_speech(self, game_state: dict, context: str) -> str:
        # 找一个非狼队友来攻击
        alive_ids = game_state.get("alive_ids", [])
        candidates = [pid for pid in alive_ids if pid != self.player_id and pid not in self.wolf_partners]

        if candidates:
            target = random.choice(candidates)

            # 检查是否有死人信息可以嫁祸
            dead_info = game_state.get("last_dead", [])
            if dead_info:
                return f"玩家{self.player_id}：我严重怀疑玩家{target}！昨晚{dead_info[0]}号死了，而{target}的发言一直很可疑。"

        templates = [
            f"玩家{self.player_id}：我认为玩家{candidates[0] if candidates else 'X'}有问题。他的发言逻辑矛盾，建议大家仔细回想。",
            f"玩家{self.player_id}：这把我有强烈的直觉，玩家{candidates[0] if candidates else 'X'}和{candidates[1] if len(candidates)>1 else 'Y'}中至少有一个狼！",
            f"玩家{self.player_id}：请注意玩家{candidates[0] if candidates else 'X'}的投票行为，非常可疑。我建议今天先出TA。",
        ]
        return random.choice(templates)

    def _generate_confused_speech(self, game_state: dict, context: str) -> str:
        templates = [
            f"玩家{self.player_id}：啊我现在完全混乱了，感觉谁说的都有道理。有没有神职出来带一下队？",
            f"玩家{self.player_id}：我这把信息量太少了，有没有大佬来分析一下？我跟着走就行。",
            f"玩家{self.player_id}：我觉得大家说得都对……我投不出手啊，能不能再来一轮发言？",
        ]
        return random.choice(templates)

    def _generate_analytic_speech(self, game_state: dict, context: str) -> str:
        """伪装成分析型村民"""
        alive_ids = game_state.get("alive_ids", [])
        # 找一个非队友来分析
        candidates = [pid for pid in alive_ids if pid != self.player_id and pid not in self.wolf_partners]
        suspect = random.choice(candidates) if candidates else None

        if suspect:
            return f"玩家{self.player_id}：我梳理了之前的发言和投票，发现玩家{suspect}的行为模式有以下疑点：(1)投票过于果断 (2)没有给出合理解释 (3)与已出局玩家的互动可疑。建议查验或投票出TA。"
        return f"玩家{self.player_id}：根据当前局势，场上活着的还有{len(alive_ids)}人。按概率还有{len(self.wolf_partners)+1}只狼。我认为大家需要集中票力。"

    def _generate_cooperative_speech(self, game_state: dict, context: str) -> str:
        """伪装合作，实则误导"""
        templates = [
            f"玩家{self.player_id}：我愿意配合预言家的查验。如果我是狼，根本不会主动要求被查。",
            f"玩家{self.player_id}：大家冷静，我建议别急着投票。让神职再出来带一波节奏。",
            f"玩家{self.player_id}：我的建议是今天先不要投。等等看有没有更多信息。",
        ]
        return random.choice(templates)

    def analyze_message(self, message: AgentMessage) -> dict:
        analysis = {"sender": message.sender_id, "type": message.message_type.value}

        if message.sender_id in self.wolf_partners:
            # 狼队友的发言，提取暗号
            analysis["trust_level"] = "high"
            analysis["is_partner"] = True
            # 狼队友的暗示信息
            content_lower = message.content.lower()
            if "查" in content_lower or "验" in content_lower:
                analysis["partner_signal"] = "targeting_seer"
            elif "投" in content_lower or "出" in content_lower:
                analysis["partner_signal"] = "vote_coordination"
        else:
            # 非队友发言分析
            analysis["trust_level"] = "unknown"

            # 更新怀疑度
            if message.message_type == MessageType.PUBLIC:
                content = message.content
                # 有人在分析，可能是神职
                if any(kw in content for kw in ["查验", "预言", "我是预言家", "昨晚查了"]):
                    self.memory.update_belief(message.sender_id, "likely_role", "预言家")
                    self.memory.suspicion_levels[message.sender_id] = 0.9
                    analysis["likely_role"] = "seer"
                elif any(kw in content for kw in ["死", "救", "毒", "解药", "我是女巫"]):
                    self.memory.update_belief(message.sender_id, "likely_role", "女巫")
                    self.memory.suspicion_levels[message.sender_id] = 0.8
                    analysis["likely_role"] = "witch"
                elif any(kw in content for kw in ["守", "保护", "我是守卫"]):
                    self.memory.update_belief(message.sender_id, "likely_role", "守卫")
                    analysis["likely_role"] = "guard"

                # 有人在攻击我
                if f"怀疑{self.player_id}" in content or f"玩家{self.player_id}有问题" in content:
                    analysis["threat"] = True
                    self.memory.suspicion_levels[message.sender_id] = min(1.0, self.memory.suspicion_levels.get(message.sender_id, 0.5) + 0.15)

        return analysis

    def update_strategy(self, game_state: dict):
        alive_ids = game_state.get("alive_ids", [])
        dead_ids = [pid for pid in range(1, 13) if pid not in alive_ids]

        # 计算狼人/好人比例
        alive_wolves = len([pid for pid in self.wolf_partners if pid in alive_ids]) + 1
        alive_good = len(alive_ids) - alive_wolves

        if alive_wolves >= alive_good:
            self.memory.strategy = "advance"  # 优势局，可以更激进
            self.aggression = min(1.0, self.aggression + 0.1)
            self.concealment = max(0.3, self.concealment - 0.1)
        elif alive_good - alive_wolves >= 2:
            self.memory.strategy = "conceal"  # 劣势局，低调隐藏
            self.concealment = min(1.0, self.concealment + 0.1)
            self.aggression = max(0.2, self.aggression - 0.1)
        else:
            self.memory.strategy = "balanced"

    def coordinate_with_wolves(self, game_state: dict) -> Optional[int]:
        """狼人夜间协调：返回统一击杀目标"""
        candidates = [pid for pid in game_state.get("alive_ids", [])
                      if pid != self.player_id and pid not in self.wolf_partners]

        if not candidates:
            return None

        # 威胁评估
        threat_ranking = []
        for pid in candidates:
            belief = self.memory.beliefs.get(pid, {})
            threat = 0
            if belief.get("likely_role") in ["预言家", "女巫"]:
                threat += 100
            elif belief.get("likely_role") in ["猎人", "守卫"]:
                threat += 60
            elif self.memory.trust_levels.get(pid, 0.5) > 0.7:
                threat += 30
            # 发言活跃度
            recent_msgs = [m for m in self.memory.conversations if m.sender_id == pid]
            threat += len(recent_msgs) * 5
            threat_ranking.append((pid, threat + random.uniform(-10, 10)))

        threat_ranking.sort(key=lambda x: x[1], reverse=True)
        return threat_ranking[0][0] if threat_ranking else None
