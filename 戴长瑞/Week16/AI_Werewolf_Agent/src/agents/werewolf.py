"""
狼人 Agent
特性：
- 知道队友身份
- 夜晚协作杀人
- 白天隐藏身份，混淆视听
- 信息隔离：只能看到队友，不知道其他狼人的具体目标
"""

from typing import List, Optional
from .base_agent import BaseAgent, ActionResult, ActionType, GameContext


class WerewolfAgent(BaseAgent):
    """
    狼人 Agent

    决策逻辑：
    1. 夜晚：基于队友信息和自己掌握的查验结果，选择杀人目标
    2. 白天：扮演好人，分析局势，引导投票
    3. 投票：优先投好人，消灭关键角色
    """

    def __init__(self, name: str, llm_client=None):
        super().__init__(name, "werewolf", llm_client)
        self._target_preference: List[str] = []  # 优先目标列表

    def _get_role_objective(self) -> str:
        return """狼人阵营胜利条件：狼人数量 >= 好人数量

策略：
1. 隐藏狼人身份，假装好人发言
2. 白天引导舆论，将嫌疑引向好人
3. 夜晚优先消灭关键好人（预言家、女巫）
4. 必要时可以牺牲队友保存自己
5. 注意：不要在发言中暴露和队友的关系"""

    async def night_action(self) -> ActionResult:
        """
        狼人夜晚杀人
        决策因素：
        1. 队友已经选择的目标（如果有信息共享）
        2. 关键好人优先消灭
        3. 伪装策略：避免总是杀同类型玩家
        """
        teammates = self.context.my_teammates() if self.context else []
        other_players = self.context.other_players() if self.context else []

        if not other_players:
            return ActionResult(
                action=ActionType.WAIT,
                reasoning="没有可杀的目标",
            )

        # 排除狼人队友，只从好人中选择目标
        candidates = [p for p in other_players if p not in teammates]
        if not candidates:
            return ActionResult(
                action=ActionType.WAIT,
                reasoning="没有可杀的好人目标",
            )

        # 决策逻辑
        prompt = self._build_night_prompt(teammates, candidates)
        target = await self.think(prompt, self.get_system_prompt())

        # 解析目标
        target = self._parse_target(target, candidates)
        if target:
            self.remember_night_action("kill", target, self._get_current_day())
            return ActionResult(
                action=ActionType.KILL,
                target=target,
                reasoning=f"选择 {target} 作为杀害目标",
            )

        return ActionResult(action=ActionType.WAIT, reasoning="等待其他狼人行动")

    async def speak(self) -> ActionResult:
        """
        狼人白天发言
        关键：伪装成好人，不能暴露队友关系
        策略：参考前面玩家的发言，做出看似合理的分析
        """
        alive_players = self.context.alive_players if self.context else []
        dead_players = self.context.get_dead_players() if self.context else []
        all_speeches = self.context.get_all_speeches() if self.context else {}

        # 构建发言摘要
        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无其他玩家发言"

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}

今天已有玩家的发言：
{speeches_text}

现在轮到你发言。作为狼人，你需要伪装成好人发言。

发言策略：
1. 针对前面玩家的发言做出回应，显得你在认真分析
2. 假装好人推理，把怀疑引向好人的方向
3. 不要攻击狼人队友，必要时可以轻微质疑但不要强烈针对
4. 如果队友已经发言，不要重复他们的观点（避免暴露队形）
5. 适当给出分析理由，避免过于模糊

发言要求：
- 不要提及任何狼人队友
- 不要使用只有狼人才知道的信息
- 表现出你在认真分析局势，找出"可疑"的人
- 语气要自然，不要太激进也不要太保守

请生成一段发言内容："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(
            action=ActionType.SPEAK,
            content=content,
            reasoning="伪装好人发言，参考其他玩家发言",
        )

    async def vote(self) -> ActionResult:
        """
        狼人投票
        基于发言分析，表面找狼但实际上投威胁大的好人
        """
        other_players = self.context.other_players() if self.context else []
        teammates = self.context.my_teammates() if self.context else []
        all_speeches = self.context.get_all_speeches() if self.context else {}
        dead_players = self.context.get_dead_players() if self.context else []

        # 构建所有发言记录
        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无发言记录"

        # 排除队友后可选的目标
        valid_targets = [p for p in other_players if p not in teammates]

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}

今天所有玩家的发言记录：
{speeches_text}

投票环节。你需要选择一个人投死。
你可以投票的目标: {', '.join(valid_targets)}

作为狼人，你的投票策略：
1. 表面理由：分析发言，找出"逻辑矛盾"或"可疑"的人
2. 实际策略：优先投有威胁的好人（可能是预言家或女巫）
3. 不要投自己的狼人队友
4. 给出一个合理的、听起来像好人的投票理由
5. 如果队友被严重怀疑，可以投队友保全局（但解释为"他的发言确实可疑"）

分析标准（表面理由）：
- 谁发言逻辑矛盾
- 谁在转移话题
- 谁过度关注特定玩家
- 谁的分析最像"好人"

请直接输出你要投票的玩家名字（只输出名字）："""

        decision = await self.think(prompt, self.get_system_prompt())

        # 优先从有效目标中解析
        target = self._parse_target(decision, valid_targets)
        if not target:
            target = self._parse_target(decision, other_players)

        if target:
            self.remember_vote(target, self._get_current_day(), "基于发言分析投票")
            return ActionResult(
                action=ActionType.VOTE,
                target=target,
                reasoning=f"投票给 {target}",
            )

        return ActionResult(action=ActionType.WAIT, reasoning="无法做出投票决策")

    def _build_night_prompt(self, teammates: List[str], candidates: List[str]) -> str:
        """构建夜晚决策提示"""
        return f"""{self.get_decision_context()}

狼人队友: {', '.join(teammates) if teammates else '无'}

这是夜晚环节，你需要选择今晚要杀害的目标。

杀害策略：
1. 优先消灭预言家、女巫等关键角色
2. 避免杀害已知是狼人的队友
3. 考虑伪装：不要总是杀害特定类型的玩家

候选目标: {', '.join(candidates)}

请直接输出你要杀害的玩家名字（只输出名字，不要其他内容）："""

    def _parse_target(self, decision: str, candidates: List[str]) -> Optional[str]:
        """解析目标名称"""
        # 清理决策文本
        decision = decision.strip()

        # 如果决策就是候选人之一
        for candidate in candidates:
            if candidate in decision or candidate.lower() in decision.lower():
                return candidate

        # 尝试提取第一个名字
        words = decision.split()
        for word in words:
            for candidate in candidates:
                if word in candidate or candidate in word:
                    return candidate

        # 默认返回第一个候选人（安全兜底）
        return candidates[0] if candidates else None

    def _get_current_day(self) -> int:
        """获取当前天数（从记忆历史推断）"""
        return max(
            [0] + [h["day"] for h in self.memory.speech_history + self.memory.night_action_history]
        )

    def add_target_preference(self, player: str):
        """添加优先目标（预言家/女巫等）"""
        if player not in self._target_preference:
            self._target_preference.insert(0, player)

    def receive_teammate_info(self, teammate: str, target: str):
        """接收队友信息（如果有信息共享机制）"""
        # 可以记录队友的选择，但狼人不能直接知道队友的具体决策
        pass


class WerewolfTeam:
    """
    狼人团队协调器
    用于多狼人场景下的信息共享和策略协调
    """

    def __init__(self):
        self.wolves: List[WerewolfAgent] = []
        self.shared_targets: List[str] = []  # 共享的优先目标
        self.eliminated_candidates: List[str] = []  # 已排除的目标

    def add_wolf(self, wolf: WerewolfAgent):
        """添加狼人"""
        self.wolves.append(wolf)

    def share_intel(self, intel: dict):
        """共享情报（谨慎使用，防止暴露）"""
        # 不直接共享具体目标，而是共享推理
        pass

    def get_consensus_target(self) -> Optional[str]:
        """获取共识目标"""
        # 狼人之间不应该有太强的协调，防止被听出队形
        return None


# 工厂函数
def create_werewolf_agent(name: str, llm_client=None) -> WerewolfAgent:
    """创建狼人 Agent"""
    return WerewolfAgent(name, llm_client)