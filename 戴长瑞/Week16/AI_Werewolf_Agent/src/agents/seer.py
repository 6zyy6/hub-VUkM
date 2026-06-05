"""
预言家 Agent
特性：
- 夜晚验人，知道被验者的阵营
- 持有私有查验记录
- 白天可以选择报身份或隐藏
- 关键角色，需要保护自己
"""

import json
from typing import List, Optional, Tuple
from .base_agent import BaseAgent, ActionResult, ActionType, GameContext


class SeerAgent(BaseAgent):
    """
    预言家 Agent

    决策逻辑：
    1. 夜晚：优先查验疑似狼人的人
    2. 白天：根据局势决定是否跳身份
    3. 投票：基于查验结果投狼人
    """

    def __init__(self, name: str, llm_client=None):
        super().__init__(name, "seer", llm_client)
        self._checked_players: List[str] = []  # 已查验的玩家

    def _get_role_objective(self) -> str:
        return """好人阵营胜利条件：所有狼人被放逐

你是预言家，每晚可以查验一名玩家的身份。

策略：
1. 优先查验发言可疑或逻辑矛盾的人
2. 查验顺序：疑似狼人 > 未知 > 明显好人
3. 白天根据情况决定是否跳身份报查验
4. 注意保护自己，不要过早暴露
5. 如果被狼人杀，遗言尽量报出关键信息

查验结果只有你自己知道，不要直接说"我昨晚查了谁"，而是可以说"有人需要被关注"等暗示性发言。"""

    async def night_action(self) -> ActionResult:
        """
        预言家夜晚验人
        决策因素：
        1. 优先查验没有明确身份的疑似狼人
        2. 已知好人不重复查验
        3. 考虑查验发言逻辑矛盾的人
        """
        all_players = self.context.alive_players if self.context else []
        unchecked = [p for p in all_players if p != self.name and p not in self._checked_players]

        if not unchecked:
            return ActionResult(
                action=ActionType.WAIT,
                reasoning="所有玩家都已查验过",
            )

        prompt = self._build_check_prompt(unchecked)
        target = await self.think(prompt, self.get_system_prompt())

        # 解析目标
        target = self._parse_target(target, unchecked)
        if target:
            self._checked_players.append(target)
            self.remember_night_action("check", target, self._get_current_day())

            return ActionResult(
                action=ActionType.CHECK,
                target=target,
                reasoning=f"选择查验 {target}",
            )

        return ActionResult(action=ActionType.WAIT, reasoning="无法做出查验决策")

    async def speak(self) -> ActionResult:
        """
        预言家白天发言
        参考前面玩家的发言，决定是否跳身份
        可以选择：
        1. 跳身份报查验
        2. 暗示性发言引导
        3. 伪装村民分析
        """
        checks = self.context.my_checks() if self.context else {}
        other_players = self.context.other_players() if self.context else []
        all_speeches = self.context.get_all_speeches() if self.context else {}
        dead_players = self.context.get_dead_players() if self.context else []

        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无其他玩家发言"

        # 判断是否有已知狼人
        known_wolves = [p for p, is_wolf in checks.items() if is_wolf]
        known_goods = [p for p, is_wolf in checks.items() if not is_wolf]
        can_reveal = len(known_wolves) > 0  # 有查杀时可以跳

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}
查验记录: {json.dumps(checks, ensure_ascii=False) if checks else '暂无'}
已知狼人: {', '.join(known_wolves) if known_wolves else '暂无'}
已知好人: {', '.join(known_goods) if known_goods else '暂无'}

今天已有玩家的发言：
{speeches_text}

轮到你发言了。你是预言家，请做出决策。

发言选项分析：
1. 跳身份报查验 {'（推荐：有确凿查杀）' if can_reveal else '：目前没有查杀，跳身份说服力不足'}
2. 暗示性发言：不直接跳，但暗示谁值得关注
3. 伪装分析：假装平民分析局势

决策要点：
- 针对前面玩家的发言做出回应
- 如果查验到狼人，考虑跳身份带队出狼
- 如果还没查到狼人，暗示性发言引导投票方向
- 注意保护自己：跳身份后可能会被狼人刀
- 回应前面玩家的质疑或分析

请生成发言内容："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(
            action=ActionType.SPEAK,
            content=content,
            reasoning="预言家发言，参考其他玩家发言",
        )

    async def vote(self) -> ActionResult:
        """
        预言家投票
        基于查验结果 + 发言分析，优先投狼人
        """
        checks = self.context.my_checks() if self.context else {}
        other_players = self.context.other_players() if self.context else []
        all_speeches = self.context.get_all_speeches() if self.context else {}
        dead_players = self.context.get_dead_players() if self.context else []

        speeches_text = "\n".join(
            f"  {speaker}: \"{content}\""
            for speaker, content in all_speeches.items()
        ) if all_speeches else "  暂无发言记录"

        # 优先投已知狼人
        known_wolves = [p for p, is_wolf in checks.items() if is_wolf]
        if known_wolves:
            target = known_wolves[0]
            self.remember_vote(target, self._get_current_day(), "查验确认是狼人")
            print(f"    [预言家决策] 查验确认 {target} 是狼人，直接投票")
            return ActionResult(
                action=ActionType.VOTE,
                target=target,
                reasoning=f"投狼人 {target}（查验确认）",
            )

        # 已知好人名单
        known_goods = [p for p, is_wolf in checks.items() if not is_wolf]

        prompt = f"""{self.get_decision_context()}

死亡玩家: {', '.join(dead_players) if dead_players else '无'}
查验记录: {json.dumps(checks, ensure_ascii=False) if checks else '暂无'}
已知好人（不投）: {', '.join(known_goods) if known_goods else '暂无'}

今天所有玩家的发言记录：
{speeches_text}

投票环节。你需要选择一个人投死。

目前没有查杀确认的狼人，请基于发言分析投票。

分析标准：
1. 谁发言逻辑最矛盾
2. 谁在转移焦点或搅混水
3. 谁过度附和别人没有独立判断
4. 谁的分析不像好人的思路
5. 结合你的查验信息（已知好人不要投）

请输出你要投票的玩家名字："""

        decision = await self.think(prompt, self.get_system_prompt())
        target = self._parse_target(decision, other_players)

        if target:
            reason = "查验确认是狼人" if target in known_wolves else "基于发言分析"
            self.remember_vote(target, self._get_current_day(), reason)
            return ActionResult(
                action=ActionType.VOTE,
                target=target,
                reasoning=f"投票给 {target}: {reason}",
            )

        return ActionResult(action=ActionType.WAIT, reasoning="无法做出投票决策")

    def _build_check_prompt(self, candidates: List[str]) -> str:
        """构建查验决策提示"""
        # 添加已查验玩家信息
        checked_info = ""
        if self.memory.night_action_history:
            checked_info = "已查验:\n"
            for record in self.memory.night_action_history:
                if record["action"] == "check":
                    checked_info += f"- {record['target']} (结果未知，只有你自己知道)\n"

        return f"""{self.get_decision_context()}

{checked_info}
可选查验目标: {', '.join(candidates)}

查验策略：
1. 优先查发言逻辑有问题的人
2. 查身份未明的人（不是明显好人也不是明显狼人）
3. 避免重复查验已经确定是好人的人

请直接输出你要查验的玩家名字："""

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

        return candidates[0] if candidates else None

    def _get_current_day(self) -> int:
        """获取当前天数"""
        return max(
            [0] + [h["day"] for h in self.memory.speech_history + self.memory.night_action_history]
        )

    def receive_check_result(self, target: str, is_wolf: bool):
        """
        接收查验结果（GameEngine 调用）
        注意：这个方法应该由 GameEngine 在验人后调用
        """
        if target not in self._checked_players:
            self._checked_players.append(target)

        # 记录到私有信息
        self.memory.private_info["checks"] = self.memory.private_info.get("checks", {})
        self.memory.private_info["checks"][target] = is_wolf

        # 同时更新推测
        if is_wolf:
            self.update_suspicion(target, True, "查验确认是狼人")


def create_seer_agent(name: str, llm_client=None) -> SeerAgent:
    """创建预言家 Agent"""
    return SeerAgent(name, llm_client)