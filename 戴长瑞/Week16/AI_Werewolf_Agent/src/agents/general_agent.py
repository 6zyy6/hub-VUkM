"""
通用智能体 GeneralAgent
能够动态适配狼人/预言家/平民/女巫等角色

特性：
1. 读取身份目标 - 根据角色自动加载目标和策略
2. 自动生成策略 - 基于LLM生成适合当前角色的决策
3. 动态修改行为 - 支持运行时调整行为模式
4. 动态适配 - 一个Agent可以扮演任何角色
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from .base_agent import BaseAgent, ActionResult, ActionType, GameContext, Memory


class RoleType(Enum):
    """角色类型枚举"""
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"  # 守卫（如果扩展）


@dataclass
class Strategy:
    """策略定义"""
    name: str
    description: str
    priority: int  # 优先级，数值越高越优先
    conditions: List[str]  # 触发条件
    actions: List[str]  # 执行动作
    prompt_template: str  # LLM提示词模板


@dataclass
class Objective:
    """角色目标"""
    role: RoleType
    team: str  # "wolf" or "good"
    win_condition: str
    core_strategy: str
    forbidden_actions: List[str]  # 禁止的行为


# ============================================================
# 角色目标库
# ============================================================

ROLE_OBJECTIVES: Dict[RoleType, Objective] = {
    RoleType.WEREWOLF: Objective(
        role=RoleType.WEREWOLF,
        team="wolf",
        win_condition="狼人数量 >= 好人数量",
        core_strategy="""
1. 隐藏身份，伪装好人发言
2. 夜晚优先击杀关键好人（预言家、女巫）
3. 白天引导舆论，将嫌疑引向好人
4. 与队友配合但不要暴露队形
5. 必要时牺牲队友保存自己
""",
        forbidden_actions=[
            "直接提及狼人队友",
            "使用只有狼人才知道的信息",
            "过度激动或过度冷静",
        ]
    ),
    RoleType.VILLAGER: Objective(
        role=RoleType.VILLAGER,
        team="good",
        win_condition="所有狼人被放逐",
        core_strategy="""
1. 认真分析每个玩家的发言
2. 找出发言逻辑矛盾的人
3. 注意狼人的伪装特征
4. 听从跳身份的好人
5. 不要盲目跟风也不要独断
""",
        forbidden_actions=[
            "假装有特殊能力",
            "无缘无故怀疑明确的好人",
            "投错票给好人",
        ]
    ),
    RoleType.SEER: Objective(
        role=RoleType.SEER,
        team="good",
        win_condition="所有狼人被放逐",
        core_strategy="""
1. 每晚查验疑似狼人的人
2. 平衡信息暴露与自我保护
3. 查验到狼人后决定是否跳身份
4. 如果被杀，遗言报出关键信息
5. 不要重复查验已经确认的好人
""",
        forbidden_actions=[
            "查验自己",
            "泄露已查验的好人信息",
            "过度暴露身份过早",
        ]
    ),
    RoleType.WITCH: Objective(
        role=RoleType.WITCH,
        team="good",
        win_condition="所有狼人被放逐",
        core_strategy="""
1. 解药优先救关键角色（预言家、猎人）
2. 毒药用于确认的狼人
3. 注意狼人可能自刀骗药
4. 白天通过发言引导投票
5. 合理规划用药时机
""",
        forbidden_actions=[
            "浪费药在错误目标上",
            "被狼人骗药",
            "过早暴露身份",
        ]
    ),
    RoleType.HUNTER: Objective(
        role=RoleType.HUNTER,
        team="good",
        win_condition="所有狼人被放逐",
        core_strategy="""
1. 白天像平民一样分析局势
2. 被杀时带走一名疑似狼人
3. 如果身份暴露，可以更激进引导
4. 注意开枪时不要误杀好人
""",
        forbidden_actions=[
            "过早暴露猎人身份",
            "随意开枪带错人",
            "在好人明显时不开枪",
        ]
    ),
}


# ============================================================
# 策略库
# ============================================================

class StrategyLibrary:
    """策略库 - 预定义各角色的常用策略"""

    @staticmethod
    def get_werewolf_strategies() -> List[Strategy]:
        """狼人策略"""
        return [
            Strategy(
                name="隐匿伪装",
                description="隐藏狼人身份，伪装成好人",
                priority=10,
                conditions=["day_phase", "alive"],
                actions=["speak_like_villager", "analyze_others"],
                prompt_template="你是一个狼人，需要伪装成好人发言。不要暴露你的狼人身份，也不要提及任何狼人队友。分析其他玩家的发言，找出可疑的好人引导投票。"
            ),
            Strategy(
                name="夜间猎杀",
                description="夜晚选择杀害目标",
                priority=10,
                conditions=["night_phase", "is_wolf"],
                actions=["select_kill_target", "avoid_killing_teammates"],
                prompt_template="你是狼人，需要选择今晚杀害的目标。优先选择威胁大的好人（预言家、女巫），不要杀害狼人队友。候选目标：{candidates}"
            ),
            Strategy(
                name="带队投票",
                description="引导投票方向",
                priority=8,
                conditions=["vote_phase", "has_target"],
                actions=["vote_for_suspicious", "avoid_voting_teammates"],
                prompt_template="你是狼人，需要投票。优先投给可疑的好人，不要投给狼人队友。如果有明确目标，选择那个目标。"
            ),
        ]

    @staticmethod
    def get_villager_strategies() -> List[Strategy]:
        """平民策略"""
        return [
            Strategy(
                name="观察分析",
                description="通过发言分析找出狼人",
                priority=10,
                conditions=["day_phase", "alive"],
                actions=["analyze_speeches", "identify_contradictions"],
                prompt_template="你是平民，没有特殊能力。认真分析其他玩家的发言，找出逻辑矛盾或可疑的人。不要盲目怀疑，要有逻辑依据。"
            ),
            Strategy(
                name="跟随好人",
                description="听从跳身份的好人",
                priority=8,
                conditions=["someone_claimed_role"],
                actions=["trust_claimed_good", "analyze_claims"],
                prompt_template="你是平民。如果有预言家或女巫跳身份报信息，认真分析其可信度。如果信息合理，可以跟随他们的判断。"
            ),
            Strategy(
                name="投票处决",
                description="基于分析投票",
                priority=7,
                conditions=["vote_phase", "has_suspicion"],
                actions=["vote_for_suspicious"],
                prompt_template="你是平民，需要投票。基于之前的发言分析，选择最可疑的人。不要投给明显的好人。"
            ),
        ]

    @staticmethod
    def get_seer_strategies() -> List[Strategy]:
        """预言家策略"""
        return [
            Strategy(
                name="查验狼人",
                description="夜晚查验疑似狼人",
                priority=10,
                conditions=["night_phase", "is_seer"],
                actions=["select_check_target", "prioritize_suspicious"],
                prompt_template="你是预言家，每晚可以查验一名玩家的身份。选择发言可疑或逻辑矛盾的人查验。已查验：{checked}"
            ),
            Strategy(
                name="报查验信息",
                description="白天报告查验结果",
                priority=9,
                conditions=["day_phase", "has_check_result", "should_claim"],
                actions=["claim_seer", "report_check_result"],
                prompt_template="你是预言家。根据查验结果决定是否跳身份报信息。如果查验到狼人且确认度高，可以考虑跳身份。查验结果：{checks}"
            ),
            Strategy(
                name="保护自己",
                description="不暴露或有限暴露",
                priority=7,
                conditions=["day_phase", "has_check_result", "should_hide"],
                actions=["speak_cryptically", "hint_without_claiming"],
                prompt_template="你是预言家，需要决定是否跳身份。如果狼人数量多或自己危险高，可以跳身份报信息引导好人。否则可以暗示性发言。"
            ),
        ]

    @staticmethod
    def get_witch_strategies() -> List[Strategy]:
        """女巫策略"""
        return [
            Strategy(
                name="用药救人",
                description="夜晚决定是否用药救",
                priority=10,
                conditions=["night_phase", "is_witch", "has_victim"],
                actions=["decide_heal", "consider_save_key_roles"],
                prompt_template="你是女巫，有解药可以救狼人杀的人。解药剩余：{heal_potion}。狼人今晚要杀的人：{victim}。如果是要杀预言家或猎人，优先救。"
            ),
            Strategy(
                name="毒杀狼人",
                description="夜晚决定是否毒人",
                priority=9,
                conditions=["night_phase", "is_witch", "has_poison_target"],
                actions=["decide_poison", "avoid_poisoning_good"],
                prompt_template="你是女巫，有毒药可以毒死一人。毒药剩余：{poison_potion}。如果你有明确狼人目标，可以考虑毒。如果没有明确目标，不要浪费药。"
            ),
            Strategy(
                name="发言引导",
                description="白天发言引导投票",
                priority=7,
                conditions=["day_phase", "is_witch", "has_info"],
                actions=["speak_to_guide", "hint_without_exposing"],
                prompt_template="你是女巫。白天通过发言引导好人投狼人。不要直接暴露身份，但可以暗示你在观察局势。"
            ),
        ]

    @classmethod
    def get_strategies_for_role(cls, role: RoleType) -> List[Strategy]:
        """获取指定角色的策略"""
        if role == RoleType.WEREWOLF:
            return cls.get_werewolf_strategies()
        elif role == RoleType.VILLAGER:
            return cls.get_villager_strategies()
        elif role == RoleType.SEER:
            return cls.get_seer_strategies()
        elif role == RoleType.WITCH:
            return cls.get_witch_strategies()
        else:
            return []


# ============================================================
# 通用智能体
# ============================================================

class GeneralAgent(BaseAgent):
    """
    通用智能体 - 动态适配任意角色

    使用方法：
    1. 创建Agent时指定角色
    2. Agent自动加载对应的目标和策略
    3. 通过play()方法执行对应角色的行动
    4. 可以动态切换角色或修改策略
    """

    def __init__(
        self,
        name: str,
        role: RoleType,
        llm_client: Optional[Any] = None,
        log_dir: str = "runs/logs",
    ):
        super().__init__(name, role.value, llm_client)

        self.role_type = role
        self.objective = ROLE_OBJECTIVES.get(role)
        self.strategies = StrategyLibrary.get_strategies_for_role(role)

        # 行为模式
        self.behavior_mode = "normal"  # normal, aggressive, defensive, conservative
        self.custom_prompts: Dict[str, str] = {}  # 自定义提示词
        self.action_hooks: Dict[str, Callable] = {}  # 行动钩子

        # 角色特定数据
        self.role_data: Dict[str, Any] = {
            "checked_players": {},  # 预言家查验记录
            "potions": {"heal": 1, "poison": 1},  # 女巫用药
            "wolf_teammates": [],  # 狼人队友
            "guard_target": None,  # 守卫守护目标
        }

    # ============================================================
    # 角色切换
    # ============================================================

    def switch_role(self, new_role: RoleType):
        """动态切换角色"""
        self.role_type = new_role
        self.role = new_role.value
        self.objective = ROLE_OBJECTIVES.get(new_role)
        self.strategies = StrategyLibrary.get_strategies_for_role(new_role)

        # 重置角色数据
        self.role_data = {
            "checked_players": {},
            "potions": {"heal": 1, "poison": 1},
            "wolf_teammates": [],
            "guard_target": None,
        }

        self.memory.role = new_role.value

    # ============================================================
    # 策略管理
    # ============================================================

    def set_behavior_mode(self, mode: str):
        """设置行为模式
        - normal: 正常策略
        - aggressive: 激进（优先攻击）
        - defensive: 保守（优先自保）
        - conservative: 谨慎（三思而后行）
        """
        self.behavior_mode = mode

    def add_strategy(self, strategy: Strategy):
        """添加自定义策略"""
        self.strategies.append(strategy)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)

    def remove_strategy(self, strategy_name: str):
        """移除策略"""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]

    def set_custom_prompt(self, action: str, prompt: str):
        """设置自定义提示词"""
        self.custom_prompts[action] = prompt

    def set_action_hook(self, action: str, hook: Callable):
        """设置行动钩子"""
        self.action_hooks[action] = hook

    # ============================================================
    # 行动执行
    # ============================================================

    async def night_action(self) -> ActionResult:
        """夜晚行动 - 根据角色自动选择"""
        if self.context:
            self._update_role_data_from_context()

        if self.role_type == RoleType.WEREWOLF:
            return await self._wolf_night_action()
        elif self.role_type == RoleType.SEER:
            return await self._seer_night_action()
        elif self.role_type == RoleType.WITCH:
            return await self._witch_night_action()
        elif self.role_type == RoleType.VILLAGER:
            return ActionResult(action=ActionType.WAIT, reasoning="平民夜晚无行动")
        else:
            return ActionResult(action=ActionType.WAIT, reasoning="该角色夜晚无特殊行动")

    async def speak(self) -> ActionResult:
        """发言 - 根据角色自动选择"""
        if self.role_type == RoleType.WEREWOLF:
            return await self._wolf_speak()
        elif self.role_type == RoleType.SEER:
            return await self._seer_speak()
        elif self.role_type == RoleType.WITCH:
            return await self._witch_speak()
        elif self.role_type == RoleType.VILLAGER:
            return await self._villager_speak()
        else:
            return await self._default_speak()

    async def vote(self) -> ActionResult:
        """投票 - 根据角色自动选择"""
        if self.role_type == RoleType.WEREWOLF:
            return await self._wolf_vote()
        elif self.role_type == RoleType.SEER:
            return await self._seer_vote()
        elif self.role_type == RoleType.WITCH:
            return await self._witch_vote()
        elif self.role_type == RoleType.VILLAGER:
            return await self._villager_vote()
        else:
            return await self._default_vote()

    # ============================================================
    # 角色特定行动实现
    # ============================================================

    async def _wolf_night_action(self) -> ActionResult:
        """狼人夜晚杀人"""
        all_others = self.context.other_players() if self.context else []
        teammates = self.role_data.get("wolf_teammates", [])
        candidates = [p for p in all_others if p not in teammates]

        if not candidates:
            return ActionResult(action=ActionType.WAIT, reasoning="无目标可选")

        prompt = self._build_night_prompt(candidates)
        decision = await self.think(prompt, self.get_system_prompt())

        target = self._parse_target(decision, candidates)
        if target:
            self.remember_night_action("kill", target, self._get_current_day())
            return ActionResult(
                action=ActionType.KILL,
                target=target,
                reasoning=f"选择杀害 {target}",
            )

        return ActionResult(action=ActionType.WAIT)

    async def _seer_night_action(self) -> ActionResult:
        """预言家夜晚验人"""
        all_players = self.context.alive_players if self.context else []
        checked = list(self.role_data["checked_players"].keys())
        unchecked = [p for p in all_players if p != self.name and p not in checked]

        if not unchecked:
            return ActionResult(action=ActionType.WAIT, reasoning="已查验所有玩家")

        prompt = self._build_seer_check_prompt(unchecked)
        decision = await self.think(prompt, self.get_system_prompt())

        target = self._parse_target(decision, unchecked)
        if target:
            self.role_data["checked_players"][target] = True
            self.remember_night_action("check", target, self._get_current_day())

            return ActionResult(
                action=ActionType.CHECK,
                target=target,
                reasoning=f"查验 {target}",
            )

        return ActionResult(action=ActionType.WAIT)

    async def _witch_night_action(self) -> ActionResult:
        """女巫夜晚用药"""
        victim = self.context._private_data.get("tonight_victim") if self.context and self.context._private_data else None
        potions = self.role_data["potions"]

        prompt = self._build_witch_prompt(potions, victim)
        decision = await self.think(prompt, self.get_system_prompt())

        decision = decision.strip().lower()

        if "救" in decision and potions["heal"] > 0 and victim:
            potions["heal"] -= 1
            self.remember_night_action("heal", victim, self._get_current_day())
            return ActionResult(action=ActionType.HEAL, target=victim, reasoning=f"使用解药救 {victim}")

        if "毒" in decision and potions["poison"] > 0:
            candidates = self.context.other_players() if self.context else []
            target = self._parse_target(decision.replace("毒", "").strip(), candidates)
            if target:
                potions["poison"] -= 1
                self.remember_night_action("poison", target, self._get_current_day())
                return ActionResult(action=ActionType.POISON, target=target, reasoning=f"使用毒药毒 {target}")

        return ActionResult(action=ActionType.WAIT, reasoning="选择不用药")

    async def _wolf_speak(self) -> ActionResult:
        """狼人发言"""
        prompt = f"""{self.get_decision_context()}

你的身份是狼人，需要伪装成好人发言。
策略：{self.objective.core_strategy if self.objective else ''}
行为模式：{self.behavior_mode}

发言要求：
- 不要提及任何狼人队友
- 表现得像认真分析局势的好人
- 可以指出其他人的"可疑"之处

请生成一段发言："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(action=ActionType.SPEAK, content=content, reasoning="狼人伪装发言")

    async def _seer_speak(self) -> ActionResult:
        """预言家发言"""
        checks = self.role_data["checked_players"]

        prompt = f"""{self.get_decision_context()}

你的身份是预言家。查验记录：{json.dumps(checks, ensure_ascii=False) if checks else '暂无'}

策略：{self.objective.core_strategy if self.objective else ''}
行为模式：{self.behavior_mode}

决定是否跳身份：
- 如果有明确狼人且确认度高，可以跳身份报查验
- 如果局势不明朗，可以暗示性发言
- 考虑是否需要暴露自己引导好人

请生成发言："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(action=ActionType.SPEAK, content=content, reasoning="预言家发言")

    async def _witch_speak(self) -> ActionResult:
        """女巫发言"""
        potions = self.role_data["potions"]

        prompt = f"""{self.get_decision_context()}

你的身份是女巫。药瓶状态：解药 {potions['heal']}，毒药 {potions['poison']}。

策略：{self.objective.core_strategy if self.objective else ''}
行为模式：{self.behavior_mode}

发言要求：
- 不要直接暴露有药
- 可以暗示"我在观察"等
- 分析其他玩家发言

请生成发言："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(action=ActionType.SPEAK, content=content, reasoning="女巫发言")

    async def _villager_speak(self) -> ActionResult:
        """平民发言"""
        prompt = f"""{self.get_decision_context()}

你的身份是平民，没有特殊能力。

策略：{self.objective.core_strategy if self.objective else ''}
行为模式：{self.behavior_mode}

发言要求：
- 分析其他玩家的发言
- 找出逻辑矛盾或可疑的人
- 表现得像一个认真分析的村民

请生成发言："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(action=ActionType.SPEAK, content=content, reasoning="平民发言分析")

    async def _wolf_vote(self) -> ActionResult:
        """狼人投票"""
        other = self.context.other_players() if self.context else []

        # 优先投好人
        suspicious = list(self.memory.suspicions.keys())
        target = next((p for p in suspicious if p in other), None)

        if not target and other:
            target = other[0]

        if target:
            self.remember_vote(target, self._get_current_day(), "狼人投票")
            return ActionResult(action=ActionType.VOTE, target=target, reasoning=f"投给 {target}")

        return ActionResult(action=ActionType.WAIT)

    async def _seer_vote(self) -> ActionResult:
        """预言家投票"""
        checks = self.role_data["checked_players"]
        known_wolves = [p for p, is_wolf in checks.items() if is_wolf]

        if known_wolves:
            target = known_wolves[0]
            self.remember_vote(target, self._get_current_day(), "查验确认投狼人")
            return ActionResult(action=ActionType.VOTE, target=target, reasoning=f"投狼人 {target}")

        other = self.context.other_players() if self.context else []
        if other:
            target = other[0]
            self.remember_vote(target, self._get_current_day(), "基于推测投票")
            return ActionResult(action=ActionType.VOTE, target=target, reasoning=f"投票给 {target}")

        return ActionResult(action=ActionType.WAIT)

    async def _witch_vote(self) -> ActionResult:
        """女巫投票"""
        suspicious = [p for p, info in self.memory.suspicions.items() if info.get("is_suspicious")]

        if suspicious:
            target = suspicious[0]
            self.remember_vote(target, self._get_current_day(), "女巫投票")
            return ActionResult(action=ActionType.VOTE, target=target, reasoning=f"投给 {target}")

        other = self.context.other_players() if self.context else []
        if other:
            target = other[0]
            self.remember_vote(target, self._get_current_day(), "女巫投票")
            return ActionResult(action=ActionType.VOTE, target=target, reasoning=f"投票给 {target}")

        return ActionResult(action=ActionType.WAIT)

    async def _villager_vote(self) -> ActionResult:
        """平民投票"""
        suspicious = [p for p, info in self.memory.suspicions.items() if info.get("is_suspicious")]

        if suspicious:
            target = suspicious[0]
            self.remember_vote(target, self._get_current_day(), "平民投票")
            return ActionResult(action=ActionType.VOTE, target=target, reasoning=f"投给 {target}")

        other = self.context.other_players() if self.context else []
        if other:
            target = other[0]
            self.remember_vote(target, self._get_current_day(), "平民投票")
            return ActionResult(action=ActionType.VOTE, target=target, reasoning=f"投票给 {target}")

        return ActionResult(action=ActionType.WAIT)

    async def _default_speak(self) -> ActionResult:
        """默认发言"""
        prompt = f"""{self.get_decision_context()}

你的身份是 {self.role}。

请生成一段发言："""

        content = await self.think(prompt, self.get_system_prompt())
        self.remember_speech(content, self._get_current_day())

        return ActionResult(action=ActionType.SPEAK, content=content)

    async def _default_vote(self) -> ActionResult:
        """默认投票"""
        other = self.context.other_players() if self.context else []

        if other:
            target = other[0]
            self.remember_vote(target, self._get_current_day(), "默认投票")
            return ActionResult(action=ActionType.VOTE, target=target)

        return ActionResult(action=ActionType.WAIT)

    # ============================================================
    # 辅助方法
    # ============================================================

    def _update_role_data_from_context(self):
        """从上下文更新角色数据"""
        if not self.context or not self.context._private_data:
            return

        data = self.context._private_data

        if self.role_type == RoleType.WEREWOLF:
            self.role_data["wolf_teammates"] = data.get("teammates", [])
        elif self.role_type == RoleType.SEER:
            existing = self.role_data["checked_players"]
            new_checks = data.get("checks", {})
            existing.update(new_checks)
        elif self.role_type == RoleType.WITCH:
            self.role_data["potions"] = data.get("potions", {"heal": 1, "poison": 1})

    def _build_night_prompt(self, candidates: List[str]) -> str:
        """构建夜晚行动提示"""
        return f"""你是狼人，今晚需要选择杀害的目标。

狼人队友：{', '.join(self.role_data.get('wolf_teammates', []))}
候选目标：{', '.join(candidates)}

优先击杀：预言家、女巫等关键好人
不要杀害狼人队友

请直接输出要杀害的玩家名字："""

    def _build_seer_check_prompt(self, candidates: List[str]) -> str:
        """构建预言家查验提示"""
        already_checked = list(self.role_data["checked_players"].keys())
        if already_checked:
            parts = []
            for p in already_checked:
                result = "狼" if self.role_data["checked_players"][p] else "好"
                parts.append(f"{p}：{result}")
            checked_str = "已查验：" + ", ".join(parts)
        else:
            checked_str = "暂无"

        return f"""你是预言家，今晚需要选择要查验的目标。

可选查验：{', '.join(candidates)}
{checked_str}

优先查验发言可疑或逻辑矛盾的人。

请直接输出要查验的玩家名字："""

    def _build_witch_prompt(self, potions: Dict, victim: Optional[str]) -> str:
        """构建女巫用药提示"""
        return f"""你是女巫，今晚需要决定是否用药。

药瓶状态：解药 {potions.get('heal', 0)} 瓶，毒药 {potions.get('poison', 0)} 瓶
狼人要杀的人：{victim or '未知'}

决策选项：
- 救人：使用解药救狼人杀的人
- 毒人：使用毒药毒死一人
- 等待：不用药

请输出你的决策（救/毒+目标 或 等待）："""

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

    def _get_role_objective(self) -> str:
        """获取角色目标"""
        if self.objective:
            return f"目标：{self.objective.win_condition}\n策略：{self.objective.core_strategy}"
        return f"角色：{self.role_type.value}"

    def get_state(self) -> Dict:
        """获取Agent状态"""
        return {
            "name": self.name,
            "role": self.role,
            "role_type": self.role_type.value,
            "behavior_mode": self.behavior_mode,
            "strategies": [s.name for s in self.strategies],
            "role_data": self.role_data.copy(),
            "memory": {
                "suspicions": self.memory.suspicions,
                "speech_count": len(self.memory.speech_history),
                "vote_count": len(self.memory.vote_history),
            },
        }

    def __repr__(self):
        return f"GeneralAgent(name={self.name}, role={self.role_type.value}, mode={self.behavior_mode})"


# ============================================================
# 工厂函数
# ============================================================

def create_general_agent(
    name: str,
    role: RoleType,
    llm_client=None,
    behavior_mode: str = "normal",
) -> GeneralAgent:
    """创建通用智能体"""
    agent = GeneralAgent(name, role, llm_client)
    agent.set_behavior_mode(behavior_mode)
    return agent


def create_team(
    player_names: List[str],
    role_mapping: Dict[str, RoleType],
    llm_client=None,
) -> Dict[str, GeneralAgent]:
    """创建游戏团队"""
    agents = {}
    for name, role_type in role_mapping.items():
        agents[name] = create_general_agent(name, role_type, llm_client)
    return agents


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GeneralAgent",
    "RoleType",
    "Strategy",
    "Objective",
    "StrategyLibrary",
    "ROLE_OBJECTIVES",
    "create_general_agent",
    "create_team",
]