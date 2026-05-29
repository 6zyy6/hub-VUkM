"""
Agent工厂模块
根据角色创建对应的Agent实例
"""

from typing import Dict, Optional
from .werewolf_agent import WerewolfAgent
from .seer_agent import SeerAgent
from .witch_agent import WitchAgent
from .hunter_agent import HunterAgent
from .villager_agent import VillagerAgent


class AgentFactory:
    """Agent工厂类"""

    _agent_registry = {
        "werewolf": WerewolfAgent,
        "seer": SeerAgent,
        "witch": WitchAgent,
        "hunter": HunterAgent,
        "villager": VillagerAgent
    }

    @classmethod
    def create_agent(cls, player_id: int, name: str, role: str,
                     llm_client=None, **kwargs) -> Optional[object]:
        """
        创建Agent实例

        :param player_id: 玩家ID
        :param name: 玩家名称
        :param role: 角色类型
        :param llm_client: LLM客户端
        :param kwargs: 其他参数
        :return: Agent实例
        """
        agent_class = cls._agent_registry.get(role)
        if not agent_class:
            raise ValueError(f"Unknown role: {role}")

        return agent_class(player_id, name, llm_client, **kwargs)

    @classmethod
    def register_agent(cls, role: str, agent_class: type):
        """注册新的Agent类型"""
        cls._agent_registry[role] = agent_class

    @classmethod
    def get_supported_roles(cls) -> list:
        """获取支持的角色列表"""
        return list(cls._agent_registry.keys())
