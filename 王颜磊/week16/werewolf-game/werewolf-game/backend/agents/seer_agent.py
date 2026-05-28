"""
预言家智能体 - 信息收集者
策略：每晚查验一名玩家，白天选择性透露身份引导投票
"""

import random
from typing import Dict, List, Optional

try:
    from .base_agent import BaseAgent, AgentMessage, MessageType
except ImportError:
    from base_agent import BaseAgent, AgentMessage, MessageType
try:
    from ..game_engine import Role, Team, Phase, GameEvent
except ImportError:
    from game_engine import Role, Team, Phase, GameEvent


class SeerAgent(BaseAgent):
    def __init__(self, player_id: int, name: str = ""):
        super().__init__(player_id, Role.SEER, name)
        self.team = Team.GOOD
        self.check_results: Dict[int, bool] = {}  # player_id -> is_werewolf
        self.revealed: bool = False  # 是否已跳身份
        self.reveal_threshold: int = 2  # 查到几个狼才跳身份
        self.wolves_found: List[int] = []
        self.verified_good: List[int] = []

    def decide_action(self, game_state: dict, phase: Phase, options: List[dict]) -> dict:
        alive_ids = game_state.get("alive_ids", [])

        if phase == Phase.NIGHT:
            # 选择查验目标
            candidates = [pid for pid in alive_ids
                          if pid != self.player_id and pid not in self.check_results]

            if not candidates:
                # 所有存活玩家都查过了，重新查
                candidates = [pid for pid in alive_ids if pid != self.player_id]

            # 选择策略：
            # 1. 优先查可疑的未查玩家
            # 2. 其次查发言活跃的
            # 3. 最后随机查

            suspicious_unchecked = [pid for pid in candidates
                                    if self.memory.suspicion_levels.get(pid, 0.0) > 0.5
                                    and pid not in self.check_results]

            if suspicious_unchecked:
                target = random.choice(suspicious_unchecked)
            elif candidates:
                target = random.choice(candidates)
            else:
                return {"action": "wait"}

            return {"action": "check", "target_id": target,
                    "reason": f"查验玩家{target}"}

        elif phase == Phase.VOTE:
            return {"action": "vote", "target_id": self.decide_vote(game_state, options)}

        return {"action": "wait"}

    def record_check(self, player_id: int, is_werewolf: bool):
        self.check_results[player_id] = is_werewolf
        if is_werewolf:
            self.wolves_found.append(player_id)
            self.memory.suspicion_levels[player_id] = 1.0
            self.memory.update_belief(player_id, "confirmed", "狼人")
        else:
            self.verified_good.append(player_id)
            self.memory.trust_levels[player_id] = 0.95
            self.memory.suspicion_levels[player_id] = 0.05
            self.memory.update_belief(player_id, "confirmed", "好人")

    def should_reveal(self, game_state: dict) -> bool:
        """判断是否应该亮身份"""
        if self.revealed:
            return True

        alive_ids = game_state.get("alive_ids", [])
        alive_count = len(alive_ids)

        # 条件1：查到狼了就跳
        if len(self.wolves_found) >= 1 and len(self.wolves_found) > 0:
            # 只要找到狼且存活狼还在，就可以考虑跳
            alive_wolves_found = [w for w in self.wolves_found if w in alive_ids]
            if alive_wolves_found:
                self.revealed = True
                return True

        # 条件2：危急时刻（人数少于6）
        if alive_count <= 6:
            self.revealed = True
            return True

        # 条件3：查到2只或以上的狼
        if len(self.wolves_found) >= 2:
            self.revealed = True
            return True

        return False

    def generate_speech(self, game_state: dict, context: str) -> str:
        if self.should_reveal(game_state):
            return self._reveal_speech(game_state)
        else:
            return self._hidden_speech(game_state, context)

    def _reveal_speech(self, game_state: dict) -> str:
        """跳预言家发言"""
        alive_ids = game_state.get("alive_ids", [])

        parts = [f"玩家{self.player_id}：我是预言家！"]

        # 报查验结果
        if self.check_results:
            parts.append("以下是我的查验记录：")
            for pid, is_wolf in self.check_results.items():
                if pid in alive_ids:
                    status = "狼人" if is_wolf else "好人"
                    parts.append(f"第N晚查玩家{pid}：{status}")

        # 给出投票建议
        alive_wolves = [w for w in self.wolves_found if w in alive_ids]
        if alive_wolves:
            parts.append(f"建议今天投票玩家{alive_wolves[0]}！")
        elif self.verified_good:
            verified_alive = [v for v in self.verified_good if v in alive_ids]
            if verified_alive:
                parts.append(f"已验证的好人：{verified_alive}")

        parts.append("请女巫和守卫保护我，我需要继续查验！")
        return " ".join(parts)

    def _hidden_speech(self, game_state: dict, context: str) -> str:
        """隐藏身份的发言"""
        templates = [
            f"玩家{self.player_id}：我观察了几轮，感觉有些玩家的行为不太对劲。建议大家多发言，暴露更多信息。",
            f"玩家{self.player_id}：目前局势还算明朗。我建议大家关注一下前几轮的投票记录，那里可能有线索。",
            f"玩家{self.player_id}：我是村民，但我有在认真分析。现在存活的人里，我认为至少还有两只狼。",
            f"玩家{self.player_id}：建议大家不要轻易跟票。请神职玩家在合适的时机出来带节奏。",
        ]

        # 如果有查到的狼但不想跳身份，隐晦提醒
        alive_ids = game_state.get("alive_ids", [])
        alive_wolves = [w for w in self.wolves_found if w in alive_ids]
        if alive_wolves and not self.revealed:
            templates.append(
                f"玩家{self.player_id}：我有一个强烈的预感，玩家{alive_wolves[0]}可能有问题。"
                f"虽然我没有证据，但建议大家留意一下。"
            )

        return random.choice(templates)

    def analyze_message(self, message: AgentMessage) -> dict:
        analysis = {"sender": message.sender_id, "type": message.message_type.value}

        if message.sender_id == self.player_id:
            return analysis

        content = message.content

        # 如果有人跳预言家
        if "我是预言家" in content and message.sender_id != self.player_id:
            analysis["claim_alert"] = "other_seer_claim"
            self.memory.suspicion_levels[message.sender_id] = 0.95
            # 对方可能是悍跳狼
            self.memory.update_belief(message.sender_id, "likely_role", "悍跳狼")

        # 如果有人攻击已验证的好人
        for good_id in self.verified_good:
            if f"玩家{good_id}" in content and "可疑" in content:
                analysis["defending_good"] = True
                self.memory.suspicion_levels[message.sender_id] = min(1.0, self.memory.suspicion_levels.get(message.sender_id, 0.5) + 0.2)

        # 投票行为分析
        if message.message_type == MessageType.VOTE:
            metadata = message.metadata
            target = metadata.get("target_id")
            if target in self.verified_good:
                analysis["bad_vote"] = True
                self.memory.suspicion_levels[message.sender_id] = min(1.0, self.memory.suspicion_levels.get(message.sender_id, 0.5) + 0.3)

        return analysis

    def update_strategy(self, game_state: dict):
        alive_ids = game_state.get("alive_ids", [])

        if self.revealed:
            self.memory.strategy = "leading"
            self.memory.confidence = 0.9
        else:
            # 计算还需要查多少人
            unchecked = [pid for pid in alive_ids
                         if pid != self.player_id and pid not in self.check_results]
            if unchecked:
                self.memory.strategy = f"继续查验，剩余待查：{len(unchecked)}人"
            else:
                self.memory.strategy = "已查验所有存活玩家"

    def decide_vote(self, game_state: dict, candidates: List[int]) -> int:
        alive_ids = game_state.get("alive_ids", [])

        # 优先投已确认的狼人
        alive_wolves = [w for w in self.wolves_found if w in alive_ids]
        if alive_wolves:
            wolf_in_candidates = [w for w in alive_wolves if w in candidates]
            if wolf_in_candidates:
                return wolf_in_candidates[0]

        return super().decide_vote(game_state, candidates)
