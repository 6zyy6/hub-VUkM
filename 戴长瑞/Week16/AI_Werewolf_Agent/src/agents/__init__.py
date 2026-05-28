"""AI Agent 模块 - 狼人杀多智能体系统"""

from .base_agent import (
    BaseAgent,
    ActionResult,
    ActionType,
    Memory,
    GameContext,
)

from .general_agent import (
    GeneralAgent,
    RoleType,
    Strategy,
    Objective,
    StrategyLibrary,
    ROLE_OBJECTIVES,
    create_general_agent,
    create_team,
)

from .werewolf import WerewolfAgent, WerewolfTeam, create_werewolf_agent
from .seer import SeerAgent, create_seer_agent
from .witch import WitchAgent, create_witch_agent
from .villager import VillagerAgent, create_villager_agent

# 工厂函数
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


def create_all_agents(player_names, role_mapping, llm_client=None):
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


__all__ = [
    # 基类
    "BaseAgent",
    "ActionResult",
    "ActionType",
    "Memory",
    "GameContext",
    # 通用智能体
    "GeneralAgent",
    "RoleType",
    "Strategy",
    "Objective",
    "StrategyLibrary",
    "ROLE_OBJECTIVES",
    "create_general_agent",
    "create_team",
    # 狼人
    "WerewolfAgent",
    "WerewolfTeam",
    "create_werewolf_agent",
    # 预言家
    "SeerAgent",
    "create_seer_agent",
    # 女巫
    "WitchAgent",
    "create_witch_agent",
    # 平民
    "VillagerAgent",
    "create_villager_agent",
    # 工厂函数
    "create_agent",
    "create_all_agents",
]