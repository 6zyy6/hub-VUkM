"""游戏规则"""
from typing import List

from ..agents.base import Role


class GameRule:
    """游戏规则"""

    # 胜利条件
    WOLF_WIN = "wolf_win"  # 狼人胜利
    GOOD_WIN = "good_win"  # 好人胜利

    # 游戏配置
    DEFAULT_PLAYER_COUNT = 9

    @staticmethod
    def check_wolf_win(living_wolves: int, living_goods: int) -> bool:
        """狼人胜利条件：狼人数量 >= 好人数量"""
        return living_wolves >= living_goods

    @staticmethod
    def check_good_win(living_wolves: int) -> bool:
        """好人胜利条件：狼人全灭"""
        return living_wolves == 0

    @staticmethod
    def distribute_roles(player_count: int) -> dict:
        """分配角色"""
        if player_count == 9:
            return {
                Role.WEREWOLF: 2,
                Role.SEER: 1,
                Role.WITCH: 1,
                Role.HUNTER: 1,
                Role.VILLAGER: 4,
            }
        elif player_count == 6:
            return {
                Role.WEREWOLF: 2,
                Role.SEER: 1,
                Role.WITCH: 1,
                Role.VILLAGER: 2,
            }
        elif player_count == 12:
            return {
                Role.WEREWOLF: 3,
                Role.SEER: 1,
                Role.WITCH: 1,
                Role.HUNTER: 1,
                Role.VILLAGER: 6,
            }
        else:
            raise ValueError(f"Unsupported player count: {player_count}")

    @staticmethod
    def validate_player_count(count: int) -> bool:
        """验证玩家数量是否合法"""
        return count in [6, 9, 12]