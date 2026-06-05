"""
游戏配置管理
"""

from typing import List, Dict
from .models import RoleType


# 标准狼人杀角色配置（12人局）
STANDARD_12_PLAYER_CONFIG = [
    {"name": "玩家1", "role": RoleType.WEREWOLF.value, "is_ai": True},
    {"name": "玩家2", "role": RoleType.WEREWOLF.value, "is_ai": True},
    {"name": "玩家3", "role": RoleType.WEREWOLF.value, "is_ai": True},
    {"name": "玩家4", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "玩家5", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "玩家6", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "玩家7", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "玩家8", "role": RoleType.SEER.value, "is_ai": True},
    {"name": "玩家9", "role": RoleType.WITCH.value, "is_ai": True},
    {"name": "玩家10", "role": RoleType.HUNTER.value, "is_ai": True},
    {"name": "玩家11", "role": RoleType.GUARD.value, "is_ai": True},
    {"name": "玩家12", "role": RoleType.VILLAGER.value, "is_ai": True},
]

# 简化版配置（6人局）
SIMPLE_6_PLAYER_CONFIG = [
    {"name": "狼人A", "role": RoleType.WEREWOLF.value, "is_ai": True},
    {"name": "狼人B", "role": RoleType.WEREWOLF.value, "is_ai": True},
    {"name": "村民A", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "村民B", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "预言家", "role": RoleType.SEER.value, "is_ai": True},
    {"name": "女巫", "role": RoleType.WITCH.value, "is_ai": True},
]

# 快速测试配置（4人局）
QUICK_4_PLAYER_CONFIG = [
    {"name": "狼人", "role": RoleType.WEREWOLF.value, "is_ai": True},
    {"name": "村民1", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "村民2", "role": RoleType.VILLAGER.value, "is_ai": True},
    {"name": "预言家", "role": RoleType.SEER.value, "is_ai": True},
]


def get_config(config_type: str = "standard_12") -> List[Dict]:
    """
    获取游戏配置
    
    Args:
        config_type: 配置类型 ("standard_12", "simple_6", "quick_4")
    
    Returns:
        玩家配置列表
    """
    configs = {
        "standard_12": STANDARD_12_PLAYER_CONFIG,
        "simple_6": SIMPLE_6_PLAYER_CONFIG,
        "quick_4": QUICK_4_PLAYER_CONFIG,
    }
    
    return configs.get(config_type, STANDARD_12_PLAYER_CONFIG)


def create_custom_config(role_distribution: Dict[str, int]) -> List[Dict]:
    """
    创建自定义配置
    
    Args:
        role_distribution: 角色分布，如 {"werewolf": 3, "villager": 4, "seer": 1, "witch": 1}
    
    Returns:
        玩家配置列表
    """
    config = []
    player_num = 1
    
    for role, count in role_distribution.items():
        for i in range(count):
            config.append({
                "name": f"{role}_{i+1}",
                "role": role,
                "is_ai": True
            })
            player_num += 1
    
    return config
