"""
角色定义模块
定义狼人杀中的各种角色及其能力
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from enum import Enum


class RoleType(Enum):
    """角色类型"""
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    VILLAGER = "villager"


class RoleAbility:
    """角色能力描述"""

    def __init__(self, name: str, description: str, action_phase: str):
        self.name = name
        self.description = description
        self.action_phase = action_phase


class BaseRole(ABC):
    """角色基类"""

    def __init__(self, role_name: str, role_type: RoleType, team: str):
        self.role_name = role_name
        self.role_type = role_type
        self.team = team
        self.abilities: List[RoleAbility] = []

    @abstractmethod
    def get_strategy_hint(self) -> str:
        """获取策略提示"""
        pass

    @abstractmethod
    def get_action_space(self) -> List[str]:
        """获取行动空间"""
        pass


class WerewolfRole(BaseRole):
    """狼人角色"""

    def __init__(self):
        super().__init__("狼人", RoleType.WEREWOLF, "werewolf")
        self.abilities = [
            RoleAbility("夜间杀人", "每晚可以选择一名玩家杀害", "night")
        ]

    def get_strategy_hint(self) -> str:
        return ("作为狼人，你需要伪装成好人，混淆视听。"
                "夜间与狼队友协商击杀目标，白天要隐藏身份，"
                "必要时可以悍跳预言家或其他神职。")

    def get_action_space(self) -> List[str]:
        return ["kill", "claim_seer", "claim_witch", "claim_villager", "defend", "accuse"]


class SeerRole(BaseRole):
    """预言家角色"""

    def __init__(self):
        super().__init__("预言家", RoleType.SEER, "villager")
        self.abilities = [
            RoleAbility("查验身份", "每晚可以查验一名玩家的真实身份", "night")
        ]

    def get_strategy_hint(self) -> str:
        return ("作为预言家，你是好人的重要信息来源。"
                "夜间查验可疑玩家，白天适时跳明身份报出查验结果。"
                "注意保护自己，避免过早暴露被狼人击杀。")

    def get_action_space(self) -> List[str]:
        return ["verify", "claim_seer", "share_info", "accuse", "defend"]


class WitchRole(BaseRole):
    """女巫角色"""

    def __init__(self):
        super().__init__("女巫", RoleType.WITCH, "villager")
        self.abilities = [
            RoleAbility("解药", "可以救活被狼人杀害的玩家", "night"),
            RoleAbility("毒药", "可以毒杀一名玩家", "night")
        ]

    def get_strategy_hint(self) -> str:
        return ("作为女巫，你拥有解药和毒药。"
                "谨慎使用解药，前期可以考虑自救，后期保留给关键角色。"
                "毒药要在确认狼人身份后使用，避免误伤好人。")

    def get_action_space(self) -> List[str]:
        return ["save", "poison", "claim_witch", "accuse", "defend", "skip"]


class HunterRole(BaseRole):
    """猎人角色"""

    def __init__(self):
        super().__init__("猎人", RoleType.HUNTER, "villager")
        self.abilities = [
            RoleAbility("临终射击", "死亡时可以带走一名玩家", "death")
        ]

    def get_strategy_hint(self) -> str:
        return ("作为猎人，你死亡时可以带走一人。"
                "前期可以适当强势，通过发言找出狼人。"
                "死亡时选择最有嫌疑的狼人带走。")

    def get_action_space(self) -> List[str]:
        return ["shoot", "claim_hunter", "accuse", "defend"]


class VillagerRole(BaseRole):
    """村民角色"""

    def __init__(self):
        super().__init__("村民", RoleType.VILLAGER, "villager")
        self.abilities = []

    def get_strategy_hint(self) -> str:
        return ("作为普通村民，虽然没有特殊能力，但你的投票至关重要。"
                "认真听取发言，通过逻辑推理找出狼人。"
                "积极投票，不要弃票。")

    def get_action_space(self) -> List[str]:
        return ["vote", "claim_villager", "accuse", "defend", "analyze"]


def create_role(role_type: RoleType) -> BaseRole:
    """工厂函数：创建角色"""
    role_map = {
        RoleType.WEREWOLF: WerewolfRole,
        RoleType.SEER: SeerRole,
        RoleType.WITCH: WitchRole,
        RoleType.HUNTER: HunterRole,
        RoleType.VILLAGER: VillagerRole
    }

    if role_type not in role_map:
        raise ValueError(f"Unknown role type: {role_type}")

    return role_map[role_type]()
