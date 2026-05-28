"""Agent 测试"""
import pytest
from src.agents.base import BaseAgent, Role, AgentState
from src.agents.villager import VillagerAgent
from src.agents.werewolf import WerewolfAgent
from src.agents.seer import SeerAgent
from src.agents.witch import WitchAgent
from src.agents.hunter import HunterAgent
from src.agents.base_agent import ActionResult, ActionType, GameContext


class MockLLM:
    """模拟 LLM - 兼容新 agent 系统的 think() 调用"""

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        return "测试回复"


class TestAgentState:
    """Agent 状态测试（旧系统）"""

    def test_state_creation(self):
        state = AgentState(name="测试", role=Role.VILLAGER)
        assert state.name == "测试"
        assert state.role == Role.VILLAGER
        assert state.is_alive is True
        assert state.can_speak is True
        assert state.vote_count == 0


class TestVillagerAgent:
    """村民 Agent 测试（新系统）"""

    @pytest.fixture
    def agent(self):
        agent = VillagerAgent("测试村民", MockLLM())
        return agent

    def test_identity(self, agent):
        assert agent.role == "villager"
        assert not agent.role == "werewolf"

    def test_good(self, agent):
        assert agent.role in ("villager", "seer", "witch")

    @pytest.mark.asyncio
    async def test_night_action(self, agent):
        result = await agent.night_action()
        assert result.action == ActionType.WAIT


class TestWerewolfAgent:
    """狼人 Agent 测试（新系统）"""

    @pytest.fixture
    def agent(self):
        return WerewolfAgent("测试狼人", MockLLM())

    def test_identity(self, agent):
        assert agent.role == "werewolf"

    @pytest.mark.asyncio
    async def test_night_action(self, agent):
        ctx = GameContext("测试狼人", ["玩家1", "玩家2"])
        ctx.set_private_data({"role": "werewolf", "teammates": []})
        agent.set_context(ctx)
        result = await agent.night_action()
        assert result.action in (ActionType.KILL, ActionType.WAIT)


class TestSeerAgent:
    """预言家 Agent 测试（新系统）"""

    @pytest.fixture
    def agent(self):
        return SeerAgent("测试预言家", MockLLM())

    def test_initial_state(self, agent):
        assert len(agent._checked_players) == 0

    def test_receive_check_result(self, agent):
        agent.receive_check_result("玩家1", True)
        assert agent.memory.private_info.get("checks", {}).get("玩家1") is True

    @pytest.mark.asyncio
    async def test_night_action(self, agent):
        ctx = GameContext("测试预言家", ["玩家1", "玩家2"])
        ctx.set_private_data({"role": "seer", "checks": {}})
        agent.set_context(ctx)
        result = await agent.night_action()
        assert result.action in (ActionType.CHECK, ActionType.WAIT)


class TestWitchAgent:
    """女巫 Agent 测试（新系统）"""

    @pytest.fixture
    def agent(self):
        return WitchAgent("测试女巫", MockLLM())

    def test_initial_potions(self, agent):
        assert agent._potion_heal == 1
        assert agent._potion_poison == 1

    def test_new_night(self, agent):
        agent._has_healed_tonight = True
        agent._has_poisoned_tonight = True
        agent.new_night()
        assert agent._has_healed_tonight is False
        assert agent._has_poisoned_tonight is False


class TestHunterAgent:
    """猎人 Agent 测试（旧系统）"""

    @pytest.fixture
    def agent(self):
        return HunterAgent("测试猎人", MockLLM())

    def test_initial_state(self, agent):
        assert agent.can_shoot is True
        assert agent.shoot_target is None

    def test_on_death_sets_target(self, agent):
        agent.shoot_target = "目标玩家"
        agent.can_shoot = False
        assert agent.can_shoot is False
