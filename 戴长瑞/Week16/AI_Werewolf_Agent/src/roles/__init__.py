"""角色模块"""
from .role_def import (
    Role,
    RoleTeam,
    RoleInfo,
    GameState,
    PlayerMemory,
    ROLE_CONFIG_6P,
    DEFAULT_6P_ROLES,
    create_role_info,
    get_role_team,
    get_winner_message,
)
from .objectives import (
    WinCondition,
    RoleObjective,
    ObjectiveChecker,
    RoleStrategy,
    get_role_objective,
    get_team_objectives,
    format_strategy_for_role,
)

__all__ = [
    "Role",
    "RoleTeam",
    "RoleInfo",
    "GameState",
    "PlayerMemory",
    "WinCondition",
    "RoleObjective",
    "ObjectiveChecker",
    "RoleStrategy",
    "ROLE_CONFIG_6P",
    "DEFAULT_6P_ROLES",
    "create_role_info",
    "get_role_team",
    "get_winner_message",
    "get_role_objective",
    "get_team_objectives",
    "format_strategy_for_role",
]