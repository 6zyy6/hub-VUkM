"""
AI Werewolf - Streamlit Web UI
实时可视化狼人杀游戏

运行: streamlit run ui/app.py
"""

import asyncio
import sys
import time
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import streamlit as st

# 添加项目路径
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.engine.game_engine import (
    GameEngine, GameLogger, Phase,
    NightActions, Player, DeathRecord, CauseOfDeath,
)
from src.agents import (
    create_all_agents, BaseAgent, WerewolfAgent, SeerAgent, WitchAgent, VillagerAgent,
    ActionType, ActionResult, GameContext,
)
from src.llm import (
    create_llm_from_config, load_config,
)

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="AI Werewolf - 狼人杀",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 自定义样式
st.markdown("""
<style>
.game-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.player-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin: 5px;
    border-left: 4px solid #4CAF50;
}
.player-card.wolf {
    border-left-color: #f44336;
}
.player-card.dead {
    opacity: 0.5;
    background: #e9ecef;
}
.night-action {
    background: #2d3436;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.day-action {
    background: #f39c12;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.speech-bubble {
    background: #e3f2fd;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 5px 0;
    position: relative;
}
.vote-badge {
    background: #e74c3c;
    color: white;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 模拟 LLM
# ============================================================

class SimpleMockLLM:
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        await asyncio.sleep(0.1)
        return "模拟决策"


class SmartMockLLM:
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        await asyncio.sleep(0.15)

        if "杀害" in prompt or "狼人" in prompt:
            names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
            return random.choice(names)
        elif "查验" in prompt or "预言家" in prompt:
            return "Bob"
        elif "女巫" in prompt:
            if "救" in prompt.lower():
                return "救 Bob"
            return "等待"
        elif "发言" in prompt:
            speeches = [
                "我觉得场上局势比较复杂，需要仔细分析。",
                "作为好人，我会认真分析每个玩家的发言。",
                "目前还没有明确线索，需要继续观察。",
                "我认为应该重点关注那些发言模糊的人。",
            ]
            return random.choice(speeches)
        elif "投票" in prompt:
            names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
            return random.choice(names)

        return "等待"


# ============================================================
# 游戏状态管理
# ============================================================

@dataclass
class GameState:
    phase: str = "waiting"
    day: int = 0
    is_game_over: bool = False
    winner: Optional[str] = None
    players: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    speeches: List[Dict] = field(default_factory=list)
    votes: List[Dict] = field(default_factory=list)
    night_actions: Dict = field(default_factory=dict)


# ============================================================
# Session State 管理
# ============================================================

def init_session_state():
    if "game_state" not in st.session_state:
        st.session_state.game_state = GameState()
    if "game_engine" not in st.session_state:
        st.session_state.game_engine = None
    if "agents" not in st.session_state:
        st.session_state.agents = {}
    if "game_running" not in st.session_state:
        st.session_state.game_running = False


def add_event(event_type: str, data: Dict):
    st.session_state.game_state.events.append({
        "type": event_type,
        "timestamp": time.time(),
        "data": data,
    })


def add_speech(player: str, content: str, role: str, day: int = 0):
    gs = st.session_state.game_state
    gs.speeches.append({
        "player": player,
        "role": role,
        "content": content,
        "day": day or gs.day,
        "timestamp": time.time(),
    })


def add_vote(voter: str, target: str, role: str):
    st.session_state.game_state.votes.append({
        "voter": voter,
        "target": target,
        "role": role,
        "timestamp": time.time(),
    })


# ============================================================
# 游戏引擎包装
# ============================================================

class WerewolfGameUI:
    """带UI的游戏引擎"""

    PLAYER_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
    ROLE_DISTRIBUTION = {"werewolf": 2, "seer": 1, "witch": 1, "villager": 2}

    def __init__(self, use_smart_llm: bool = True, llm_client=None):
        self.llm = llm_client or (SmartMockLLM() if use_smart_llm else SimpleMockLLM())
        self.engine: Optional[GameEngine] = None
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = GameLogger("runs/logs")
        self.game_state = st.session_state.game_state
        self._day_speeches: Dict[str, str] = {}

    def init_game(self):
        """初始化游戏"""
        # 随机分配角色
        roles = []
        for role, count in self.ROLE_DISTRIBUTION.items():
            roles.extend([role] * count)
        random.shuffle(roles)

        role_mapping = {}
        for name, role in zip(self.PLAYER_NAMES, roles):
            role_mapping[name] = role

        # 创建 Agent
        raw_agents = create_all_agents(self.PLAYER_NAMES, role_mapping, self.llm)
        self.agents = {name: AgentWrapperUI(agent) for name, agent in raw_agents.items()}

        # 创建游戏引擎
        self.engine = GameEngine(
            player_names=self.PLAYER_NAMES,
            role_distribution=self.ROLE_DISTRIBUTION,
            log_dir="runs/logs",
            role_mapping=role_mapping,
        )

        # 设置AI决策
        self._setup_ai_decisions()

        # 更新游戏状态
        self.game_state.players = [
            {"name": name, "role": role, "is_alive": True, "can_speak": True}
            for name, role in role_mapping.items()
        ]
        self.game_state.phase = "night"
        self.game_state.day = 0
        self.game_state.is_game_over = False
        self.game_state.events = []
        self.game_state.speeches = []
        self.game_state.votes = []

        # 记录游戏开始
        player_info = [{"name": n, "role": role_mapping[n]} for n in self.PLAYER_NAMES]
        self.logger.log_game_start(player_info)

        add_event("game_start", {"players": role_mapping})

    def _setup_ai_decisions(self):
        """设置AI决策"""

        async def wolf_decision(player, engine):
            context = self._create_context(player.name)
            context.set_private_data({
                "role": "werewolf",
                "teammates": [n for n in self.PLAYER_NAMES if n != player.name and self.agents[n].is_wolf],
            })
            result = await self.agents[player.name].night_action(context)
            return result.target

        async def seer_decision(player, engine):
            context = self._create_context(player.name)
            context.set_private_data({
                "role": "seer",
                "checks": self.agents[player.name].seer_checks,
            })
            result = await self.agents[player.name].night_action(context)
            target = result.target
            if target:
                is_wolf = self.agents[target].is_wolf
                self.agents[player.name].agent.memory.private_info.setdefault("checks", {})[target] = is_wolf
            return (target, is_wolf if target else (None, None))

        async def witch_heal_decision(player, engine):
            context = self._create_context(player.name)
            context.set_private_data({
                "role": "witch",
                "potions": {
                    "heal": self.agents[player.name].heal_potion,
                    "poison": self.agents[player.name].poison_potion,
                },
                "tonight_victim": engine.night_actions.wolf_kill_target if hasattr(engine, 'night_actions') else None,
            })
            result = await self.agents[player.name].night_action(context)
            return result.target if result.action == ActionType.HEAL else None

        async def speech(player, engine):
            context = self._create_context(player.name)

            # 共享已收集的发言给当前说话者
            context.set_public_data({
                "speeches": dict(self._day_speeches),
                "dead_players": [p.name for p in self.engine.players.values() if not p.is_alive],
            })

            private_data = {"role": self.agents[player.name].role}
            if self.agents[player.name].is_wolf:
                private_data["teammates"] = self.agents[player.name].wolf_teammates
            elif self.agents[player.name].role == "seer":
                private_data["checks"] = self.agents[player.name].seer_checks
            elif self.agents[player.name].role == "witch":
                private_data["potions"] = {
                    "heal": self.agents[player.name].heal_potion,
                    "poison": self.agents[player.name].poison_potion,
                }
            context.set_private_data(private_data)
            content = await self.agents[player.name].speak(context)

            # 记录发言
            self._day_speeches[player.name] = content
            add_speech(player.name, content, self.agents[player.name].role, day=self.game_state.day)

            return content

        async def vote(player, engine):
            context = self._create_context(player.name)

            # 传递所有发言给投票决策
            context.set_public_data({
                "speeches": dict(self._day_speeches),
                "dead_players": [p.name for p in self.engine.players.values() if not p.is_alive],
            })

            context.set_private_data({"role": self.agents[player.name].role})
            target = await self.agents[player.name].vote(context)
            if target:
                add_vote(player.name, target, self.agents[player.name].role)
            return target

        async def poison_decision(player, engine):
            return None

        self.engine.set_ai_decision_maker("werewolf", wolf_decision)
        self.engine.set_ai_decision_maker("seer", seer_decision)
        self.engine.set_ai_decision_maker("witch", witch_heal_decision)
        self.engine.set_ai_decision_maker("witch_poison", poison_decision)
        self.engine.set_ai_decision_maker("speak", speech)
        self.engine.set_ai_decision_maker("vote", vote)

    def _create_context(self, player_name: str) -> GameContext:
        if self.engine:
            living = [p.name for p in self.engine.living_players]
        else:
            living = [p["name"] for p in self.game_state.players if p["is_alive"]]
        if not living:
            living = self.PLAYER_NAMES.copy()
        return GameContext(player_name, living)

    async def run_night_phase(self):
        """运行夜晚阶段"""
        self.game_state.day += 1
        self.game_state.phase = "night"

        add_event("night_start", {"day": self.game_state.day})

        # 夜晚行动
        night_result = await self.engine.night_phase()

        # 更新存活状态
        for p in self.engine.players.values():
            for gp in self.game_state.players:
                if gp["name"] == p.name:
                    gp["is_alive"] = p.is_alive
                    gp["can_speak"] = p.can_speak

        # 记录夜晚行动
        self.game_state.night_actions = {
            "wolf_kill": night_result.wolf_kill_target,
            "seer_check": night_result.seer_check_target,
            "seer_result": "狼人" if night_result.seer_check_result else "好人",
            "witch_heal": night_result.witch_heal_target,
            "deaths": night_result.dead_players,
        }

        add_event("night_end", {
            "wolf_kill": night_result.wolf_kill_target,
            "seer_check": f"{night_result.seer_check_target} -> {'狼人' if night_result.seer_check_result else '好人'}",
            "deaths": night_result.dead_players,
        })

    async def run_day_phase(self):
        """运行白天阶段"""
        self.game_state.phase = "day"
        self._day_speeches = {}  # 重置当天发言记录

        add_event("day_start", {"day": self.game_state.day})

        # 白天发言和投票
        executed = await self.engine.day_phase()

        # 更新存活状态
        for p in self.engine.players.values():
            for gp in self.game_state.players:
                if gp["name"] == p.name:
                    gp["is_alive"] = p.is_alive
                    gp["can_speak"] = p.can_speak

        add_event("day_end", {"executed": executed, "votes": self.game_state.votes[-6:]})

        return executed

    def check_win(self) -> bool:
        """检查胜负"""
        living_wolves = len(self.engine.living_wolf_players)
        living_goods = len(self.engine.living_good_players)

        if living_wolves == 0:
            self.game_state.is_game_over = True
            self.game_state.winner = "good"
            return True

        if living_wolves >= living_goods:
            self.game_state.is_game_over = True
            self.game_state.winner = "wolf"
            return True

        return False

    def save_log(self):
        return self.logger.save()


class AgentWrapperUI:
    """Agent包装器"""

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.name = agent.name
        self.role = agent.role

    @property
    def is_wolf(self) -> bool:
        return self.role == "werewolf"

    @property
    def is_alive(self) -> bool:
        return self.agent.memory.public_info.get("is_alive", True)

    @property
    def can_speak(self) -> bool:
        return self.agent.memory.public_info.get("can_speak", True)

    @property
    def wolf_teammates(self) -> List[str]:
        return self.agent.memory.private_info.get("teammates", [])

    @property
    def seer_checks(self) -> Dict[str, bool]:
        return self.agent.memory.private_info.get("checks", {})

    @property
    def heal_potion(self) -> int:
        return self.agent.memory.private_info.get("potions", {}).get("heal", 1)

    @property
    def poison_potion(self) -> int:
        return self.agent.memory.private_info.get("potions", {}).get("poison", 1)

    async def night_action(self, context: GameContext) -> ActionResult:
        self.agent.set_context(context)
        return await self.agent.night_action()

    async def speak(self, context: GameContext) -> str:
        self.agent.set_context(context)
        result = await self.agent.speak()
        return result.content or ""

    async def vote(self, context: GameContext) -> str:
        self.agent.set_context(context)
        result = await self.agent.vote()
        return result.target or ""


# ============================================================
# UI 组件
# ============================================================

def render_header():
    """渲染标题栏"""
    st.markdown("""
    <div class="game-header">
        <h1>🐺 AI Werewolf</h1>
        <p>全自动狼人杀多智能体对战系统 | 6人局</p>
    </div>
    """, unsafe_allow_html=True)


def render_game_info():
    """渲染游戏信息"""
    gs = st.session_state.game_state

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        phase_emoji = "🌙" if gs.phase == "night" else "☀️"
        st.metric("当前阶段", f"{phase_emoji} {gs.phase.upper()}")

    with col2:
        st.metric("当前天数", gs.day)

    with col3:
        alive = sum(1 for p in gs.players if p["is_alive"])
        st.metric("存活人数", alive)

    with col4:
        wolves = sum(1 for p in gs.players if p["is_alive"] and p["role"] == "werewolf")
        st.metric("狼人数量", wolves)


def render_players():
    """渲染玩家列表"""
    gs = st.session_state.game_state

    st.subheader("👥 玩家状态")

    cols = st.columns(3)
    for i, player in enumerate(gs.players):
        with cols[i % 3]:
            is_alive = player["is_alive"]
            is_wolf = player["role"] == "werewolf"

            css_class = "player-card"
            if is_wolf:
                css_class += " wolf"
            if not is_alive:
                css_class += " dead"

            status = "🟢 存活" if is_alive else "🔴 死亡"
            role_emoji = "🐺" if is_wolf else "😊"

            st.markdown(f"""
            <div class="{css_class}">
                <h4>{role_emoji} {player['name']}</h4>
                <p><b>身份:</b> {player['role']}</p>
                <p><b>状态:</b> {status}</p>
            </div>
            """, unsafe_allow_html=True)


def render_night_actions():
    """渲染夜晚行动"""
    gs = st.session_state.game_state
    actions = gs.night_actions

    if not actions:
        return

    st.subheader("🌙 夜晚行动")

    with st.container():
        if actions.get("wolf_kill"):
            st.info(f"🐺 狼人选择杀害: {actions['wolf_kill']}")
        else:
            st.info("🐺 狼人今晚未杀人")

        if actions.get("seer_check"):
            st.success(f"👁️ 预言家查验: {actions['seer_check']} -> {actions.get('seer_result', '未知')}")
        else:
            st.info("👁️ 预言家今晚未查验")

        if actions.get("witch_heal"):
            st.success(f"🧪 女巫使用解药救: {actions['witch_heal']}")

        if actions.get("deaths"):
            for death in actions["deaths"]:
                st.error(f"💀 {death} 死亡")
        else:
            st.success("今晚平安夜，无人死亡")


def render_speeches():
    """渲染发言记录（按轮次分组展示）"""
    gs = st.session_state.game_state

    st.subheader("💬 发言记录")

    if not gs.speeches:
        st.info("暂无发言记录")
        return

    with st.container():
        # 按天分组显示
        current_day = None
        for speech in gs.speeches:
            day = speech.get("day", gs.day)
            if day != current_day:
                current_day = day
                st.caption(f"——— 第 {day} 天 ———")

            role_emoji = "🐺" if speech["role"] == "werewolf" else \
                         "👁️" if speech["role"] == "seer" else \
                         "🧪" if speech["role"] == "witch" else "😊"
            st.markdown(f"""
            <div class="speech-bubble">
                <b>{role_emoji} {speech['player']}</b> <small>({speech['role']})</small>: {speech['content']}
            </div>
            """, unsafe_allow_html=True)


def render_votes():
    """渲染投票记录"""
    gs = st.session_state.game_state

    st.subheader("🗳️ 投票记录")

    if not gs.votes:
        st.info("暂无投票记录")
        return

    # 显示投票统计
    vote_counts: Dict[str, int] = {}
    for vote in gs.votes:
        target = vote["target"]
        vote_counts[target] = vote_counts.get(target, 0) + 1

    for target, count in sorted(vote_counts.items(), key=lambda x: -x[1]):
        st.progress(count / 6, text=f"{target}: {count}票")

    # 显示详细投票
    with st.expander("查看详细投票"):
        for vote in gs.votes[-6:]:
            st.text(f"{vote['voter']} -> {vote['target']}")


def render_events():
    """渲染事件日志"""
    gs = st.session_state.game_state

    st.subheader("📋 事件日志")

    if not gs.events:
        return

    with st.container():
        for event in reversed(gs.events[-10:]):
            event_type = event["type"]
            data = event.get("data", {})

            if event_type == "night_start":
                st.write(f"🌙 第{data.get('day', '?')}夜开始")
            elif event_type == "night_end":
                st.write(f"  夜晚死亡: {data.get('deaths', [])}")
            elif event_type == "day_start":
                st.write(f"☀️ 第{data.get('day', '?')}天开始")
            elif event_type == "day_end":
                executed = data.get('executed', '无')
                st.write(f"  处决: {executed}")
            elif event_type == "game_start":
                st.write("🎮 游戏开始！")


def render_game_over():
    """渲染游戏结束"""
    gs = st.session_state.game_state

    if not gs.is_game_over:
        return

    st.markdown("---")
    st.subheader("🏆 游戏结束")

    if gs.winner == "good":
        st.success("🎉 好人胜利！狼人全部被放逐！")
    elif gs.winner == "wolf":
        st.error("🐺 狼人胜利！狼人数量已占优势！")
    else:
        st.warning("游戏结束，无结果")

    # 显示最终状态
    st.subheader("📊 最终状态")
    for player in gs.players:
        status = "存活" if player["is_alive"] else "死亡"
        st.text(f"{player['name']}: {player['role']} - {status}")


def render_sidebar():
    """渲染侧边栏控制"""
    st.sidebar.title("🎮 游戏控制")

    # 开始游戏按钮
    if not st.session_state.game_running:
        if st.sidebar.button("🚀 开始新游戏", type="primary", use_container_width=True):
            # 根据选择创建 LLM
            cfg = load_config()
            llm_provider = cfg.get("llm", {}).get("provider", "mock")
            llm_client = create_llm_from_config()

            if llm_client is not None:
                # 真实 LLM
                game = WerewolfGameUI(use_smart_llm=False, llm_client=llm_client)
            else:
                # mock 模式
                game = WerewolfGameUI(use_smart_llm=cfg.get("game", {}).get("mode", "smart") == "smart")
            st.session_state.game = game
            st.session_state.game_running = True
            game.init_game()
            st.rerun()

    # 继续游戏按钮
    if st.session_state.game_running and not st.session_state.game_state.is_game_over:
        if st.sidebar.button("▶️ 继续下一步", type="primary", use_container_width=True):
            game = st.session_state.game

            async def run_step():
                if game.game_state.phase == "night":
                    await game.run_night_phase()
                    if game.check_win():
                        return
                    await game.run_day_phase()
                    game.check_win()

            asyncio.run(run_step())
            st.rerun()

        # 自动运行按钮
        if st.sidebar.button("⚡ 自动运行", type="secondary", use_container_width=True):
            game = st.session_state.game

            async def run_auto():
                for _ in range(15):
                    if game.game_state.is_game_over:
                        break
                    if game.game_state.phase == "night":
                        await game.run_night_phase()
                        if game.check_win():
                            break
                    await game.run_day_phase()
                    game.check_win()
                    if game.check_win():
                        break
                    await asyncio.sleep(0.5)

            asyncio.run(run_auto())
            st.rerun()

    # 重置按钮
    if st.sidebar.button("🔄 重置游戏", use_container_width=True):
        st.session_state.game_running = False
        st.session_state.game_state = GameState()
        st.rerun()

    # 游戏说明
    # LLM 配置选择
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 LLM 配置")

    llm_provider = st.sidebar.selectbox(
        "LLM 提供商",
        options=["mock (模拟)", "claude (Claude API)", "openai (OpenAI API)"],
        index=0,
        help="选择 AI 决策所用的 LLM。mock 使用内置规则，claude/openai 调用真实 API",
    )

    if "claude" in llm_provider:
        config = load_config()
        model = config.get("llm", {}).get("model", "claude-sonnet-4-20250514")
        api_key = config.get("llm", {}).get("api_key", "")
        if not api_key:
            st.sidebar.warning("⚠️ config.toml 中未配置 api_key", icon="⚠️")
        st.sidebar.info(f"模型: {model}", icon="🧠")

    if "openai" in llm_provider:
        config = load_config()
        model = config.get("llm", {}).get("model", "gpt-4o")
        st.sidebar.info(f"模型: {model}", icon="🧠")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🎯 游戏规则

    **角色配置：**
    - 🐺 狼人 x2
    - 👁️ 预言家 x1
    - 🧪 女巫 x1
    - 😊 村民 x2

    **胜利条件：**
    - 好人：狼人全灭
    - 狼人：狼人数量 >= 好人数量

    ### ⏰ 游戏流程

    **夜晚：**
    1. 狼人选择杀害目标
    2. 预言家查验身份
    3. 女巫决定是否用药

    **白天：**
    1. 宣布夜晚死亡
    2. 存活玩家发言
    3. 投票处决
    """)


# ============================================================
# 主界面
# ============================================================

def main():
    init_session_state()

    # 渲染标题
    render_header()

    # 渲染侧边栏控制
    render_sidebar()

    # 渲染游戏信息
    render_game_info()

    # 渲染玩家状态
    render_players()

    # 渲染夜晚行动
    if st.session_state.game_running:
        render_night_actions()

        # 渲染发言和投票
        col1, col2 = st.columns(2)
        with col1:
            render_speeches()
        with col2:
            render_votes()

        # 渲染事件日志
        render_events()

        # 渲染游戏结束
        render_game_over()

        # 保存日志
        if st.session_state.game_state.is_game_over:
            if st.button("💾 保存日志"):
                game = st.session_state.game
                log_path = game.save_log()
                st.success(f"日志已保存: {log_path}")

    else:
        st.info("👈 点击左侧「开始新游戏」按钮启动AI狼人杀对战")

        # 显示游戏说明
        st.markdown("""
        ---
        ## 🐺 AI Werewolf - 全自动狼人杀

        这是一个**多智能体AI狼人杀对战系统**，6个AI Agent将自动进行完整的狼人杀游戏。

        ### 功能特性

        - 🤖 **纯AI对战**：6个独立的AI Agent扮演不同角色
        - 🔒 **信息隔离**：每个Agent只能看到自己的私有信息
        - 🧠 **自主决策**：基于LLM和记忆系统做出决策
        - 📊 **实时可视化**：实时显示游戏进展
        - 📝 **结构化日志**：完整记录每局游戏

        ### 如何运行

        1. 点击左侧「开始新游戏」
        2. 点击「继续下一步」单步执行
        3. 或点击「自动运行」快速完成
        """)


if __name__ == "__main__":
    main()