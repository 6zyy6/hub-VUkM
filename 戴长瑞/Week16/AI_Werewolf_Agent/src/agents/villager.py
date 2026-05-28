"""
平民 Agent
特性：
- 没有任何特殊能力
- 通过分析发言和投票行为找出狼人
- 信息完全基于公共信息和推测
- 需要依靠预言家和女巫的信息
"""

import json
from typing import Dict, List, Optional
from .base_agent import BaseAgent, ActionResult, ActionType, GameContext


class VillagerAgent(BaseAgent):
    """
    平民 Agent

    决策逻辑：
    1. 白天：通过发言分析找出狼人
    2. 投票：基于发言和投票行为
    3. 不能依赖任何私有信息
    4. 需要识别狼人的伪装
    """

    def __init__(self, name: str, llm_client=None):
        super().__init__(name, "villager", llm_client)

    def _get_role_objective(self) -> str:
        return """好人阵营胜利条件：所有狼人被放逐

你是平民，没有特殊能力。

分析策略：
1. 观察发言逻辑：狼人发言通常会有逻辑矛盾
2. 观察投票行为：狼人可能投票给队友或引导偏离
3. 观察立场变化：狼人可能会突然改变立场
4. 听从预言家：如果有人跳预言家，分析其发言可信度
5. 警惕过于激动的人：狼人可能通过激烈发言转移注意力

发言要求：
- 作为平民，应该表现得更"普通"
- 不要假装有特殊能力
- 可以分析其他人的发言
- 适当表示"我在观察"等

找狼线索：
- 发言过于完美或过于模糊
- 总是附和别人没有主见
- 过度关注某个特定玩家
- 发言中暴露只有狼人才知道的信息
- 投票行为异常"""

    async def night_action(self) -> ActionResult:
        """
        平民夜晚没有行动
        等待天亮
        """
        return ActionResult(
            action=ActionType.WAIT,
            reasoning="平民夜晚没有行动",
        )

    async def speak(self) -> ActionResult:
        """
        平民白天发言
        分析局势，找出可疑玩家，参考前面玩家的发言
        """
        other_players = self.context.other_players() if self.context else []
        dead_players = self.context.get_dead_players() if self.context else []
        all_speeches = self.context.get_all_speeches() if self.context else {}

        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无其他玩家发言"

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}

今天已有玩家的发言：
{speeches_text}

轮到你发言了。你是平民，需要通过分析发言找出狼人。

发言目标：
1. 针对前面玩家的发言内容做出回应
2. 分析谁发言可疑，指出逻辑矛盾
3. 提出你的怀疑对象和理由
4. 不要只附和别人，要有自己的判断

分析策略：
- 谁在带节奏引导投票方向
- 谁的发言前后矛盾
- 谁过度攻击某个人（狼人常用手法）
- 谁发言模糊、回避关键问题
- 谁总是附和大家没有主见

请生成发言内容："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(
            action=ActionType.SPEAK,
            content=content,
            reasoning="平民发言，分析其他玩家发言",
        )

    async def vote(self) -> ActionResult:
        """
        平民投票
        基于发言分析，没有特殊信息，全靠推理
        """
        other_players = self.context.other_players() if self.context else []
        all_speeches = self.context.get_all_speeches() if self.context else {}
        dead_players = self.context.get_dead_players() if self.context else {}

        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无发言记录"

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}

今天所有玩家的发言记录：
{speeches_text}

投票环节。你需要选择一个人投死。

作为平民，你没有特殊信息，只能靠发言分析来识别狼人。

分析每位玩家的发言：
1. Alice的发言是否逻辑自洽？
2. Bob的发言是否在转移焦点？
3. Charlie的发言是否真实？
4. 谁在刻意引导投票方向？
5. 谁的表现像是知道内情（可能是狼人）？
6. 谁的发言最像好人？

投票标准：
- 优先投发言最可疑、逻辑最矛盾的人
- 避免投分析清晰、逻辑合理的人（可能是好人）
- 警惕发言过于完美的人（可能是狼人准备）

请输出你要投票的玩家名字："""

        decision = await self.think(prompt, self.get_system_prompt())
        target = self._parse_target(decision, other_players)

        if target:
            self.remember_vote(target, self._get_current_day(), "基于发言分析投票")
            return ActionResult(
                action=ActionType.VOTE,
                target=target,
                reasoning=f"投票给 {target}（发言分析）",
            )

        return ActionResult(action=ActionType.WAIT, reasoning="无法做出投票决策")

    def _parse_target(self, decision: str, candidates: List[str]) -> Optional[str]:
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

        return None

    def _get_current_day(self) -> int:
        """获取当前天数"""
        return max(
            [0] + [h["day"] for h in self.memory.speech_history + self.memory.night_action_history]
        )

    def analyze_speech(self, speaker: str, content: str) -> bool:
        """
        分析其他玩家的发言
        返回：是否可疑
        """
        # 基于发言内容分析
        suspicious_keywords = [
            "我觉得",  # 过于保守
            "可能吧",  # 没有主见
            "我不知道",  # 假装无辜
        ]

        is_suspicious = any(kw in content for kw in suspicious_keywords)
        if is_suspicious:
            self.update_suspicion(speaker, True, "发言可疑")

        return is_suspicious


def create_villager_agent(name: str, llm_client=None) -> VillagerAgent:
    """创建平民 Agent"""
    return VillagerAgent(name, llm_client)


# ============================================================
# Agent 工厂
# ============================================================

def create_agent(name: str, role: str, llm_client=None) -> BaseAgent:
    """根据角色创建对应的 Agent"""
    if role == "werewolf":
        return WerewolfAgent(name, llm_client)
    elif role == "seer":
        return SeerAgent(name, llm_client)
    elif role == "witch":
        return WitchAgent(name, llm_client)
    elif role == "villager":
        return VillagerAgent(name, llm_client)
    else:
        raise ValueError(f"Unknown role: {role}")


def create_all_agents(player_names: List[str], role_mapping: Dict[str, str], llm_client=None) -> Dict[str, BaseAgent]:
    """
    创建所有 Agent

    Args:
        player_names: 玩家名称列表
        role_mapping: 角色映射 {player_name: role}
        llm_client: LLM 客户端

    Returns:
        Dict[str, BaseAgent]: 玩家名称 -> Agent
    """
    agents = {}
    for name in player_names:
        role = role_mapping.get(name, "villager")
        agents[name] = create_agent(name, role, llm_client)
    return agents