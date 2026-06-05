"""游戏引擎测试"""
import pytest
from src.engine.game import WerewolfGame, GameConfig, Player
from src.engine.phase import Phase, GamePhase
from src.engine.rule import GameRule
from src.engine.role import RoleConfig
from src.agents.base import Role


class TestGamePhase:
    """阶段测试"""

    def test_phase_initial_state(self):
        phase = Phase()
        assert phase.current == GamePhase.WAITING
        assert phase.day_number == 0

    def test_phase_transitions(self):
        phase = Phase()
        phase.set_phase(GamePhase.NIGHT)
        assert phase.is_night
        assert not phase.is_day

        phase.set_phase(GamePhase.DAY)
        assert phase.is_day
        assert phase.day_number == 1


class TestGameRule:
    """游戏规则测试"""

    def test_wolf_win_condition(self):
        assert GameRule.check_wolf_win(2, 3) is False
        assert GameRule.check_wolf_win(3, 3) is True
        assert GameRule.check_wolf_win(4, 3) is True

    def test_good_win_condition(self):
        assert GameRule.check_good_win(0) is True
        assert GameRule.check_good_win(1) is False
        assert GameRule.check_good_win(2) is False

    def test_role_distribution(self):
        config = GameRule.distribute_roles(9)
        assert config[Role.WEREWOLF] == 2
        assert config[Role.SEER] == 1
        assert config[Role.WITCH] == 1
        assert config[Role.HUNTER] == 1

        total = sum(config.values())
        assert total == 9


class TestRoleConfig:
    """角色配置测试"""

    def test_default_config(self):
        config = RoleConfig.default_config()
        assert Role.VILLAGER in config
        assert Role.WEREWOLF in config
        assert config[Role.WEREWOLF] == 2

    def test_total_players(self):
        assert RoleConfig.get_total_players() == 9


class TestPlayer:
    """玩家测试"""

    def test_player_default_state(self):
        player = Player(name="测试", agent=None, role=Role.VILLAGER)
        assert player.is_alive is True
        assert player.can_speak is True
        assert player.vote_count == 0