"""
女巫智能体 - 拥有双药的强力神职
策略：解药用于挽救关键神职或证明自己，毒药用于确认的狼人或威胁
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


class WitchAgent(BaseAgent):
    def __init__(self, player_id: int, name: str = ""):
        super().__init__(player_id, Role.WITCH, name)
        self.team = Team.GOOD
        self.has_save_potion: bool = True
        self.has_poison_potion: bool = True
        self.saved_player: Optional[int] = None
        self.poisoned_player: Optional[int] = None
        self.revealed: bool = False
        # 策略参数
        self.save_aggressiveness: float = random.uniform(0.4, 0.8)  # 解药使用激进程度
        self.poison_caution: float = random.uniform(0.3, 0.7)  # 毒药谨慎程度

    def decide_action(self, game_state: dict, phase: Phase, options: List[dict]) -> dict:
        alive_ids = game_state.get("alive_ids", [])

        if phase == Phase.NIGHT:
            night_info = options[0] if options else {}
            killed_player = night_info.get("werewolf_target")

            decisions = self._night_decision(game_state, killed_player)
            return decisions

        elif phase == Phase.VOTE:
            return {"action": "vote", "target_id": self.decide_vote(game_state, options)}

        return {"action": "wait"}

    def _night_decision(self, game_state: dict, killed_player: Optional[int]) -> dict:
        decisions = {"action": "night_actions", "save": None, "poison": None}

        # 解药决策
        if self.has_save_potion and killed_player:
            if self._should_save(game_state, killed_player):
                decisions["save"] = killed_player
                self.saved_player = killed_player

        # 毒药决策
        if self.has_poison_potion:
            poison_target = self._select_poison_target(game_state)
            if poison_target:
                decisions["poison"] = poison_target
                self.poisoned_player = poison_target

        return decisions

    def _should_save(self, game_state: dict, killed_player: int) -> bool:
        """判断是否使用解药"""
        # 第一夜：大概率救
        if game_state.get("round", 0) == 1:
            return random.random() < self.save_aggressiveness

        # 后续：判断被杀的是否是关键神职或高度可信玩家
        belief = self.memory.beliefs.get(killed_player, {})
        confirmed_role = belief.get("confirmed")

        if confirmed_role in ["预言家", "女巫", "守卫"]:
            return True  # 必定救关键神职

        if self.memory.trust_levels.get(killed_player, 0.5) > 0.75:
            return True

        # 自己被杀？不能自救时跳过
        if killed_player == self.player_id:
            return False

        # 基于策略激进程度
        return random.random() < self.save_aggressiveness * 0.7

    def _select_poison_target(self, game_state: dict) -> Optional[int]:
        """选择毒药目标"""
        alive_ids = game_state.get("alive_ids", [])

        # 优先毒确认的狼人
        suspected_wolves = []
        for pid in alive_ids:
            if pid == self.player_id:
                continue
            belief = self.memory.beliefs.get(pid, {})
            suspicion = self.memory.suspicion_levels.get(pid, 0.0)

            if belief.get("confirmed") == "狼人":
                return pid  # 确认的狼人，直接毒
            elif suspicion > 0.8:
                suspected_wolves.append(pid)

        if suspected_wolves and random.random() > self.poison_caution:
            return random.choice(suspected_wolves)

        # 不急于用毒，保留到关键时机
        alive_count = len(alive_ids)
        if alive_count <= 5 and suspected_wolves:
            return random.choice(suspected_wolves)

        # 如果有人跳了可疑身份且强烈怀疑
        for pid in alive_ids:
            if pid == self.player_id:
                continue
            belief = self.memory.beliefs.get(pid, {})
            if belief.get("likely_role") == "悍跳狼":
                return pid

        return None

    def use_save(self) -> bool:
        if self.has_save_potion:
            self.has_save_potion = False
            return True
        return False

    def use_poison(self) -> bool:
        if self.has_poison_potion:
            self.has_poison_potion = False
            return True
        return False

    def generate_speech(self, game_state: dict, context: str) -> str:
        if self.should_reveal(game_state):
            return self._reveal_speech(game_state)
        else:
            return self._hidden_speech(game_state, context)

    def should_reveal(self, game_state: dict) -> bool:
        """判断是否亮身份"""
        if self.revealed:
            return True

        alive_ids = game_state.get("alive_ids", [])

        # 解药或毒药已使用，可以跳
        if not self.has_save_potion or not self.has_poison_potion:
            if len(alive_ids) <= 6:
                self.revealed = True
                return True

        # 有预言家跳了，配合
        if any("我是预言家" in self._get_recent_context() for _ in [1]):
            # 不急，先观察
            pass

        # 危急时刻
        if len(alive_ids) <= 4:
            self.revealed = True
            return True

        return False

    def _reveal_speech(self, game_state: dict) -> str:
        parts = [f"玩家{self.player_id}：我是女巫！"]
        if not self.has_save_potion and self.saved_player:
            parts.append(f"第1晚解药已用，救了玩家{self.saved_player}。")
        elif not self.has_save_potion:
            parts.append("解药已使用。")

        if not self.has_poison_potion and self.poisoned_player:
            parts.append(f"毒药已用，毒杀了玩家{self.poisoned_player}。")
        elif not self.has_poison_potion:
            parts.append("毒药已使用。")

        if self.has_save_potion:
            parts.append("我还有解药。")
        if self.has_poison_potion:
            parts.append("我还有毒药。")

        parts.append("请预言家配合，守卫保护我。")
        return " ".join(parts)

    def _hidden_speech(self, game_state: dict, context: str) -> str:
        templates = [
            f"玩家{self.player_id}：建议神职玩家谨慎操作。目前局势需要稳扎稳打。",
            f"玩家{self.player_id}：我是村民，但我观察了很多。大家注意分析发言的逻辑漏洞。",
            f"玩家{self.player_id}：这把好人局势不错。我建议大家继续分析，不要贸然投票。",
            f"玩家{self.player_id}：我觉得我们应该仔细分析每个人的立场和动机。",
        ]
        return random.choice(templates)

    def _get_recent_context(self) -> str:
        recent = self.memory.conversations[-5:]
        return " ".join([m.content for m in recent])

    def analyze_message(self, message: AgentMessage) -> dict:
        analysis = {"sender": message.sender_id, "type": message.message_type.value}

        content = message.content

        # 预言家跳身份
        if "我是预言家" in content:
            analysis["seer_claim"] = True
            # 提高该玩家的信任度（可能是真预言家）
            self.memory.trust_levels[message.sender_id] = min(1.0, self.memory.trust_levels.get(message.sender_id, 0.5) + 0.2)

        # 有人跳女巫（可能是狼）
        if "我是女巫" in content and message.sender_id != self.player_id:
            analysis["witch_claim_alert"] = True
            self.memory.suspicion_levels[message.sender_id] = 0.9
            self.memory.update_belief(message.sender_id, "likely_role", "假女巫/狼")

        # 有人攻击疑似好人
        for pid, trust in self.memory.trust_levels.items():
            if trust > 0.8 and f"玩家{pid}" in content and "可疑" in content:
                analysis["attacking_good"] = True
                self.memory.suspicion_levels[message.sender_id] = min(1.0, self.memory.suspicion_levels.get(message.sender_id, 0.5) + 0.15)

        return analysis

    def update_strategy(self, game_state: dict):
        alive_ids = game_state.get("alive_ids", [])

        if not self.has_save_potion and not self.has_poison_potion:
            self.memory.strategy = "双药已用，转为普通村民"
            self.memory.confidence = 0.7
        elif self.revealed:
            self.memory.strategy = "已亮身份，积极配合预言家"
            self.memory.confidence = 0.85
        else:
            self.memory.strategy = "隐藏身份，保留双药"
            self.memory.confidence = 0.6

    def decide_vote(self, game_state: dict, candidates: List[int]) -> int:
        # 优先投自己毒过但没死的目标（不太可能，但以防万一）
        if self.poisoned_player and self.poisoned_player in candidates:
            return self.poisoned_player

        return super().decide_vote(game_state, candidates)
