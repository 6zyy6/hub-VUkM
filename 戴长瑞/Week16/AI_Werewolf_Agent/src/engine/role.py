"""角色配置"""
from dataclasses import dataclass
from typing import Dict

from ..agents.base import Role


@dataclass
class RoleConfig:
    """角色配置"""
    role: Role
    count: int
    description: str

    @staticmethod
    def default_config() -> Dict[Role, int]:
        """默认角色配置"""
        return {
            Role.WEREWOLF: 2,
            Role.SEER: 1,
            Role.WITCH: 1,
            Role.HUNTER: 1,
            Role.VILLAGER: 4,
        }

    @staticmethod
    def get_total_players() -> int:
        """获取总玩家数"""
        return sum(RoleConfig.default_config().values())  # 9人局