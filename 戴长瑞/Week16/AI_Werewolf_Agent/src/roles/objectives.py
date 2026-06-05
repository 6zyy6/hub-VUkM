"""角色目标与胜利条件"""
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .role_def import Role, RoleTeam, GameState


class WinCondition(Enum):
    """胜利条件"""
    WOLF_WIN = "wolf_win"
    GOOD_WIN = "good_win"
    NO_WIN = "no_win"


@dataclass
class RoleObjective:
    """角色目标"""
    role: Role
    team: RoleTeam
    win_condition: str
    hints: List[str]


# 各角色目标定义
ROLE_OBJECTIVES = {
    Role.WEREWOLF: RoleObjective(
        role=Role.WEREWOLF,
        team=RoleTeam.WOLF,
        win_condition="狼人数量 >= 好人数量时狼人胜利",
        hints=[
            "隐藏身份，不要暴露狼人队友",
            "引导舆论，将嫌疑引向好人",
            "夜晚配合杀害关键角色（预言家、女巫）",
            "白天假装好人分析局势",
        ]
    ),
    Role.VILLAGER: RoleObjective(
        role=Role.VILLAGER,
        team=RoleTeam.GOOD,
        win_condition="所有狼人被放逐时好人胜利",
        hints=[
            "分析发言，找出可疑玩家",
            "注意狼人的逻辑漏洞",
            "不要轻易暴露自己的村民身份",
            "通过投票放逐狼人",
        ]
    ),
    Role.SEER: RoleObjective(
        role=Role.SEER,
        team=RoleTeam.GOOD,
        win_condition="所有狼人被放逐时好人胜利",
        hints=[
            "每晚查验疑似狼人的玩家",
            "根据查验结果引导好人投票",
            "平衡信息暴露与自我保护",
            "查验顺序：优先查疑似狼人的玩家",
        ]
    ),
    Role.WITCH: RoleObjective(
        role=Role.WITCH,
        team=RoleTeam.GOOD,
        win_condition="所有狼人被放逐时好人胜利",
        hints=[
            "解药留给关键角色（预言家、猎人）",
            "毒药用来消灭狼人",
            "注意狼人可能自刀骗药",
            "白天可以通过发言引导投票",
        ]
    ),
}


class ObjectiveChecker:
    """目标检查器 - 检查胜利条件"""

    @staticmethod
    def check_win_condition(state: GameState) -> WinCondition:
        """检查当前游戏状态的胜利条件"""
        living_players = state.get_living_players()

        # 统计存活玩家中的狼人和好人
        wolves = []
        goods = []

        for player_name in living_players:
            role = state.player_roles.get(player_name)
            if role == Role.WEREWOLF:
                wolves.append(player_name)
            elif role in [Role.VILLAGER, Role.SEER, Role.WITCH]:
                goods.append(player_name)

        # 胜利条件检查
        if len(wolves) == 0:
            return WinCondition.GOOD_WIN

        if len(wolves) >= len(goods):
            return WinCondition.WOLF_WIN

        return WinCondition.NO_WIN

    @staticmethod
    def get_game_progress(state: GameState) -> Dict:
        """获取游戏进度"""
        wolves = [p for p in state.living_players if state.player_roles.get(p) == Role.WEREWOLF]
        goods = [p for p in state.living_players if state.player_roles.get(p) != Role.WEREWOLF]

        return {
            "total_players": len(state.players),
            "living_count": len(state.living_players),
            "wolf_count": len(wolves),
            "good_count": len(goods),
            "death_count": len(state.players) - len(state.living_players),
            "day_number": state.day_number,
        }

    @staticmethod
    def is_game_over(state: GameState) -> bool:
        """检查游戏是否结束"""
        return ObjectiveChecker.check_win_condition(state) != WinCondition.NO_WIN


class RoleStrategy:
    """角色策略 - 生成各角色的行动提示"""

    @staticmethod
    def get_seer_night_strategy(state: GameState, seer_name: str) -> Dict:
        """预言家夜晚策略"""
        living = state.get_living_players()
        checked = state.get_player_memory(seer_name)._private_seer_checks
        unchecked = [p for p in living if p != seer_name and p not in checked]

        # 优先查验未确认的玩家
        candidates = unchecked[:3] if unchecked else living[:3]

        return {
            "candidates": candidates,
            "already_checked": list(checked.keys()),
            "strategy": "查验疑似狼人的玩家，优先选择发言可疑或逻辑矛盾的人"
        }

    @staticmethod
    def get_witch_night_strategy(state: GameState, witch_name: str) -> Dict:
        """女巫夜晚策略"""
        memory = state.get_player_memory(witch_name)
        potions = memory._private_witch_potions.copy()
        victim = state.wolf_kill_target  # 狼人今晚要杀的人

        return {
            "heal_remaining": potions["heal"],
            "poison_remaining": potions["poison"],
            "victim_to_save": victim,
            "strategy": "解药优先救预言家或猎人，毒药用来毒狼人"
        }

    @staticmethod
    def get_werewolf_night_strategy(state: GameState, wolf_name: str) -> Dict:
        """狼人夜晚策略"""
        memory = state.get_player_memory(wolf_name)
        teammates = memory._private_wolf_teammates
        living_goods = [
            p for p in state.get_living_players()
            if state.player_roles.get(p) != Role.WEREWOLF
        ]

        # 优先杀害关键好人
        key_targets = []
        for p in living_goods:
            role = state.player_roles.get(p)
            if role in [Role.SEER, Role.WITCH]:
                key_targets.append(p)

        return {
            "teammates": teammates,
            "candidates": living_goods[:5],
            "key_targets": key_targets,
            "strategy": "优先杀预言家和女巫，配合队友行动"
        }

    @staticmethod
    def get_villager_day_strategy(state: GameState, villager_name: str) -> Dict:
        """村民白天策略"""
        living = state.get_living_players()
        deaths = state.death_record

        return {
            "living_players": living,
            "recent_deaths": deaths[-3:] if deaths else [],
            "strategy": "分析发言找出狼人，注意狼人可能冒充好人引导舆论"
        }


def get_role_objective(role: Role) -> RoleObjective:
    """获取角色目标"""
    return ROLE_OBJECTIVES.get(role)


def get_team_objectives(team: RoleTeam) -> List[RoleObjective]:
    """获取阵营目标"""
    return [obj for obj in ROLE_OBJECTIVES.values() if obj.team == team]


def format_strategy_for_role(role: Role, state: GameState, player_name: str) -> str:
    """为特定角色格式化策略提示"""
    if role == Role.SEER:
        strategy = RoleStrategy.get_seer_night_strategy(state, player_name)
        already_checked_str = ', '.join([f"{k}: {'狼人' if v else '好人'}" for k, v in strategy['already_checked'].items()]) if strategy['already_checked'] else '暂无'
        return f"""你是预言家，夜晚可以查验一名玩家。

候选查验玩家: {', '.join(strategy['candidates'])}
已查验结果: {already_checked_str}

提示: {strategy['strategy']}
"""
    elif role == Role.WITCH:
        strategy = RoleStrategy.get_witch_night_strategy(state, player_name)
        return f"""你是女巫，有解药和毒药各一瓶。

解药剩余: {strategy['heal_remaining']}
毒药剩余: {strategy['poison_remaining']}
狼人今晚要杀的人: {strategy['victim_to_save'] or '未知'}

提示: {strategy['strategy']}
"""
    elif role == Role.WEREWOLF:
        strategy = RoleStrategy.get_werewolf_night_strategy(state, player_name)
        return f"""你是狼人，夜晚可以杀害一名玩家。

狼人队友: {', '.join(strategy['teammates'])}
可选目标: {', '.join(strategy['candidates'])}
关键目标（预言家/女巫）: {', '.join(strategy['key_targets']) if strategy['key_targets'] else '无'}

提示: {strategy['strategy']}
"""
    else:
        strategy = RoleStrategy.get_villager_day_strategy(state, player_name)
        return f"""你是村民，需要找出狼人。

存活玩家: {', '.join(strategy['living_players'])}
近期死亡: {', '.join([d['player'] for d in strategy['recent_deaths']]) if strategy['recent_deaths'] else '无'}

提示: {strategy['strategy']}
"""