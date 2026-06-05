"""角色定义 - 6人局狼人杀"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class Role(Enum):
    """角色枚举"""
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"


class RoleTeam(Enum):
    """阵营枚举"""
    WOLF = "wolf"      # 狼人阵营
    GOOD = "good"      # 好人阵营


@dataclass
class RoleInfo:
    """角色信息"""
    role: Role
    team: RoleTeam
    name: str
    description: str
    abilities: List[str]


# 角色配置
ROLE_CONFIG_6P = {
    Role.WEREWOLF: 2,
    Role.VILLAGER: 2,
    Role.SEER: 1,
    Role.WITCH: 1,
}


# 角色描述
ROLE_INFO = {
    Role.WEREWOLF: RoleInfo(
        role=Role.WEREWOLF,
        team=RoleTeam.WOLF,
        name="狼人",
        description="狼人每晚可以杀害一名玩家。狼人需要隐藏身份，引导舆论，消灭所有好人。",
        abilities=["杀害: 夜晚选择一名玩家杀害"]
    ),
    Role.VILLAGER: RoleInfo(
        role=Role.VILLAGER,
        team=RoleTeam.GOOD,
        name="村民",
        description="村民没有任何特殊能力，需要通过分析和推理找出狼人。",
        abilities=["投票: 白天投票放逐嫌疑人"]
    ),
    Role.SEER: RoleInfo(
        role=Role.SEER,
        team=RoleTeam.GOOD,
        name="预言家",
        description="预言家每晚可以查验一名玩家的身份是好人还是狼人。",
        abilities=["查验: 夜晚选择一名玩家查验身份"]
    ),
    Role.WITCH: RoleInfo(
        role=Role.WITCH,
        team=RoleTeam.GOOD,
        name="女巫",
        description="女巫拥有两瓶药：解药可以救活当晚被狼人杀的人，毒药可以毒死一名玩家。每瓶药只能使用一次。",
        abilities=["救人: 使用解药救活被杀的玩家", "毒人: 使用毒药毒死一名玩家"]
    ),
}


@dataclass
class PlayerMemory:
    """玩家记忆 - 实现信息隔离"""
    player_name: str
    role: Role

    # 自己可见的信息（其他人不可见）
    _private_seer_checks: Dict[str, bool] = field(default_factory=dict)  # 预言家查验记录
    _private_witch_potions: Dict[str, int] = field(default_factory=lambda: {"heal": 1, "poison": 1})  # 女巫用药
    _private_wolf_teammates: List[str] = field(default_factory=list)  # 狼人队友

    # 公共信息（所有人可见）
    deaths: List[str] = field(default_factory=list)  # 死亡玩家列表
    speeches: List[Dict[str, str]] = field(default_factory=list)  # 发言记录

    def get_private_info(self) -> Dict:
        """获取该角色私有的信息"""
        return {
            "role": self.role.value,
            "seer_checks": self._private_seer_checks.copy(),
            "witch_potions": self._private_witch_potions.copy(),
            "wolf_teammates": self._private_wolf_teammates.copy(),
        }

    def add_seer_check(self, target: str, is_wolf: bool):
        """预言家添加查验结果"""
        self._private_seer_checks[target] = is_wolf

    def use_heal_potion(self):
        """女巫使用解药"""
        self._private_witch_potions["heal"] = max(0, self._private_witch_potions["heal"] - 1)

    def use_poison_potion(self):
        """女巫使用毒药"""
        self._private_witch_potions["poison"] = max(0, self._private_witch_potions["poison"] - 1)

    def set_wolf_teammates(self, teammates: List[str]):
        """狼人设置队友"""
        self._private_wolf_teammates = teammates


@dataclass
class GameState:
    """游戏状态"""
    # 玩家信息
    players: List[str]  # 所有玩家名称
    player_roles: Dict[str, Role]  # 玩家 -> 角色 (只有自己知道)
    player_memory: Dict[str, PlayerMemory]  # 玩家记忆

    # 当前状态
    living_players: List[str] = field(default_factory=list)  # 存活玩家
    day_number: int = 1
    current_phase: str = "night"  # night / day

    # 夜晚行动
    wolf_kill_target: Optional[str] = None  # 狼人要杀的人
    seer_check_target: Optional[str] = None  # 预言家要查验的人
    seer_check_result: Optional[bool] = None  # 查验结果
    witch_heal_target: Optional[str] = None  # 女巫救的人
    witch_poison_target: Optional[str] = None  # 女巫毒的人

    # 投票
    votes: Dict[str, str] = field(default_factory=dict)  # 玩家 -> 投票目标
    vote_history: List[Dict] = field(default_factory=list)  # 投票历史

    # 死亡记录
    death_record: List[Dict] = field(default_factory=list)  # 死亡记录 [{"player": str, "cause": str, "day": int}]

    @classmethod
    def create_new(cls, players: List[str], role_distribution: Dict[Role, int]):
        """创建新游戏状态"""
        import random
        roles = []
        for role, count in role_distribution.items():
            roles.extend([role] * count)
        random.shuffle(roles)

        player_roles = {name: role for name, role in zip(players, roles)}

        player_memory = {}
        for name, role in player_roles.items():
            memory = PlayerMemory(player_name=name, role=role)

            # 狼人知道队友
            if role == Role.WEREWOLF:
                teammates = [n for n, r in player_roles.items() if r == Role.WEREWOLF and n != name]
                memory.set_wolf_teammates(teammates)

            player_memory[name] = memory

        living_players = players.copy()

        return cls(
            players=players,
            player_roles=player_roles,
            player_memory=player_memory,
            living_players=living_players,
        )

    def get_player_role(self, player_name: str) -> Role:
        """获取玩家角色（只有自己能调用）"""
        return self.player_roles[player_name]

    def get_living_players(self) -> List[str]:
        """获取存活玩家"""
        return [p for p in self.living_players if p in self.player_roles]

    def get_player_memory(self, player_name: str) -> PlayerMemory:
        """获取玩家记忆"""
        return self.player_memory[player_name]

    def eliminate_player(self, player_name: str, cause: str):
        """淘汰玩家"""
        if player_name in self.living_players:
            self.living_players.remove(player_name)
            self.death_record.append({
                "player": player_name,
                "cause": cause,
                "day": self.day_number,
            })

    def new_night(self):
        """新夜晚重置"""
        self.current_phase = "night"
        self.wolf_kill_target = None
        self.seer_check_target = None
        self.seer_check_result = None
        self.witch_heal_target = None
        self.witch_poison_target = None

    def new_day(self):
        """新白天重置"""
        self.current_phase = "day"
        self.day_number += 1
        self.votes = {}


def create_role_info(role: Role) -> RoleInfo:
    """创建角色信息"""
    return ROLE_INFO[role]


def get_role_team(role: Role) -> RoleTeam:
    """获取角色阵营"""
    return ROLE_INFO[role].team


def get_winner_message(winner: str) -> str:
    """获取胜利消息"""
    if winner == "wolf":
        return "狼人胜利！狼人消灭了所有好人。"
    elif winner == "good":
        return "好人胜利！所有狼人被放逐。"
    return "游戏结束，无结果。"


# 6人局默认配置
DEFAULT_6P_ROLES = {
    Role.WEREWOLF: 2,
    Role.VILLAGER: 2,
    Role.SEER: 1,
    Role.WITCH: 1,
}