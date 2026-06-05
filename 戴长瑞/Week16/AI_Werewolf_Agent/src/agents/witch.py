"""
女巫 Agent
特性：
- 有解药和毒药各一瓶
- 解药可以救狼人杀的人
- 毒药可以毒死一人
- 每瓶药只能用一次
- 夜晚决定是否用药
"""

from typing import Optional, Dict
from .base_agent import BaseAgent, ActionResult, ActionType, GameContext


class WitchAgent(BaseAgent):
    """
    女巫 Agent

    决策逻辑：
    1. 夜晚：根据狼人杀人目标决定是否用药
    2. 解药策略：优先救关键角色（预言家、猎人）
    3. 毒药策略：在确认狼人时使用
    4. 白天：通过发言引导投票
    """

    def __init__(self, name: str, llm_client=None):
        super().__init__(name, "witch", llm_client)
        self._potion_heal: int = 1  # 解药
        self._potion_poison: int = 1  # 毒药
        self._has_healed_tonight: bool = False
        self._has_poisoned_tonight: bool = False

    def _get_role_objective(self) -> str:
        return """好人阵营胜利条件：所有狼人被放逐

你是女巫，有两瓶药：
- 解药：可以救活狼人当晚杀的人，每局只能用一次
- 毒药：可以毒死一名玩家，每局只能用一次

策略：
1. 解药优先救关键角色（预言家、猎人）
2. 注意狼人可能自刀骗药
3. 毒药在确认狼人身份时使用
4. 白天通过发言引导好人
5. 如果身份暴露，合理规划用药

用药时机：
- 狼人杀预言家/猎人时优先救
- 狼人杀平民时根据情况决定
- 毒药要确认目标身份再使用
- 不要浪费药在错误的目标上"""

    async def night_action(self) -> ActionResult:
        """
        女巫夜晚用药
        决策因素：
        1. 狼人今晚杀的人是谁
        2. 是否需要救人（解药）
        3. 是否需要毒人（毒药）
        """
        victim = self._get_victim_from_context()
        potions = self.context.my_potions() if self.context else {"heal": 1, "poison": 1}

        prompt = f"""{self.get_decision_context()}

药瓶状态：
- 解药剩余: {potions['heal']} 瓶
- 毒药剩余: {potions['poison']} 瓶
狼人今晚要杀的人: {victim or '未知'}

女巫决策选项：
1. 救人：使用解药救活狼人杀的人
2. 毒人：使用毒药毒死一名玩家
3. 等待：不用药

决策策略：
- 如果狼人杀预言家/猎人，优先救
- 如果狼人杀平民，根据局势决定
- 如果有明确狼人目标，可以考虑毒药
- 不要被狼人自刀骗药

请输出你的决策（格式：救人/毒人+目标 或 等待）：
- 救人: "救 [玩家名]"
- 毒人: "毒 [玩家名]"
- 等待: "等待"
"""

        decision = await self.think(prompt, self.get_system_prompt())
        return self._parse_witch_decision(decision, victim)

    async def speak(self) -> ActionResult:
        """
        女巫白天发言
        参考前面玩家的发言，隐藏身份的同时分析局势
        """
        potions = self.context.my_potions() if self.context else {"heal": 1, "poison": 1}
        all_speeches = self.context.get_all_speeches() if self.context else {}
        dead_players = self.context.get_dead_players() if self.context else {}

        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无其他玩家发言"

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}
药瓶状态（仅自己可见）:
- 解药剩余: {potions['heal']} 瓶
- 毒药剩余: {potions['poison']} 瓶

今天已有玩家的发言：
{speeches_text}

轮到你了。你是女巫，但不要让其他人知道你的身份。

发言策略：
1. 不要直接暴露自己有药
2. 针对前面玩家的发言做出回应
3. 以普通好人的角度分析局势
4. 判断哪些发言像是狼人
5. 如果狼人杀了人但你救了（或没救），注意不要暴露信息
6. 可以暗示谁值得关注

请生成发言内容："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(
            action=ActionType.SPEAK,
            content=content,
            reasoning="女巫发言，参考其他玩家发言",
        )

    async def vote(self) -> ActionResult:
        """
        女巫投票
        基于发言分析 + 夜间信息（谁被狼人刀了）进行投票
        """
        other_players = self.context.other_players() if self.context else []
        all_speeches = self.context.get_all_speeches() if self.context else {}
        dead_players = self.context.get_dead_players() if self.context else {}
        potions = self.context.my_potions() if self.context else {"heal": 1, "poison": 1}

        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无发言记录"

        # 获取昨晚被狼人刀的人（女巫知道）
        victim = self._get_victim_from_context()

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}
昨晚狼人目标: {victim or '未知'}
药瓶状态: 解药{potions.get('heal', 0)}瓶 毒药{potions.get('poison', 0)}瓶

今天所有玩家的发言记录：
{speeches_text}

投票环节。你需要选择一个人投死。

作为女巫，你的决策依据：
1. 发言分析：谁的发言最可疑、逻辑最矛盾
2. 夜间信息：谁可能和狼人目标有关联
3. 谁在转移焦点或攻击明显好人的人
4. 谁在保护某些特定玩家（可能是队友）
5. 如果昨晚有人被刀而你救了，注意观察谁在关注这个信息

分析标准：
- 发表矛盾言论的人
- 过度攻击某个人的人（可能是在带节奏）
- 一直保持低调模糊的人
- 总是附和别人没有自己观点的人

请输出你要投票的玩家名字："""

        decision = await self.think(prompt, self.get_system_prompt())
        target = self._parse_target(decision, other_players)

        if target:
            self.remember_vote(target, self._get_current_day(), "基于发言和夜间信息")
            return ActionResult(
                action=ActionType.VOTE,
                target=target,
                reasoning=f"投票给 {target}",
            )

        return ActionResult(action=ActionType.WAIT, reasoning="无法做出投票决策")

    def _parse_witch_decision(self, decision: str, victim: Optional[str]) -> ActionResult:
        """解析女巫决策"""
        decision = decision.strip().lower()

        # 解析救人
        if "救" in decision and self._potion_heal > 0 and not self._has_healed_tonight:
            if victim:
                self._potion_heal -= 1
                self._has_healed_tonight = True
                self.remember_night_action("heal", victim, self._get_current_day())
                return ActionResult(
                    action=ActionType.HEAL,
                    target=victim,
                    reasoning=f"使用解药救 {victim}",
                )

        # 解析毒人
        if "毒" in decision and self._potion_poison > 0 and not self._has_poisoned_tonight:
            target = self._extract_target_from_decision(decision)
            if target:
                self._potion_poison -= 1
                self._has_poisoned_tonight = True
                self.remember_night_action("poison", target, self._get_current_day())
                return ActionResult(
                    action=ActionType.POISON,
                    target=target,
                    reasoning=f"使用毒药毒 {target}",
                )

        return ActionResult(action=ActionType.WAIT, reasoning="选择不用药")

    def _extract_target_from_decision(self, decision: str) -> Optional[str]:
        """从决策中提取目标"""
        words = decision.replace("毒", "").replace("人", "").strip()
        # 简化处理，实际应该匹配玩家名
        return words if words else None

    def _parse_target(self, decision: str, candidates: list) -> Optional[str]:
        """解析目标名称"""
        decision = decision.strip()

        for candidate in candidates:
            if candidate in decision or candidate.lower() in decision.lower():
                return candidate

        words = decision.split()
        for word in words:
            for candidate in candidates:
                if word in candidate or candidate in word:
                    return candidate

        return candidates[0] if candidates else None

    def _get_victim_from_context(self) -> Optional[str]:
        """从上下文获取狼人杀的人"""
        return self.context._private_data.get("tonight_victim") if self.context and self.context._private_data else None

    def _get_current_day(self) -> int:
        """获取当前天数"""
        return max(
            [0] + [h["day"] for h in self.memory.speech_history + self.memory.night_action_history]
        )

    def new_night(self):
        """新夜晚重置用药状态"""
        self._has_healed_tonight = False
        self._has_poisoned_tonight = False
        self.memory.private_info["tonight_victim"] = None

    def set_victim(self, victim: str):
        """GameEngine 调用，设置狼人今晚要杀的人"""
        self.memory.private_info["tonight_victim"] = victim

    @property
    def heal_potion_remaining(self) -> int:
        return self._potion_heal

    @property
    def poison_potion_remaining(self) -> int:
        return self._potion_poison


def create_witch_agent(name: str, llm_client=None) -> WitchAgent:
    """创建女巫 Agent"""
    return WitchAgent(name, llm_client)