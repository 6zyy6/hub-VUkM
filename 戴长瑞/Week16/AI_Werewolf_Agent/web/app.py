"""Streamlit 主应用"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game import WerewolfGame, GameConfig
from src.agents.base import Role
from src.agents import VillagerAgent, WerewolfAgent, SeerAgent, WitchAgent, HunterAgent
from src.llm.openai_llm import OpenAILLM
from .components.game_board import render_game_board
from .components.chat import render_chat_panel


st.set_page_config(
    page_title="AI Werewolf - 狼人杀",
    page_icon="🐺",
    layout="wide",
)


def init_session_state():
    """初始化会话状态"""
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "game" not in st.session_state:
        st.session_state.game = None
    if "players" not in st.session_state:
        st.session_state.players = []
    if "logs" not in st.session_state:
        st.session_state.logs = []


def create_agents(player_names: list, llm) -> list:
    """创建 Agent"""
    # 默认配置：2狼人、1预言家、1女巫、1猎人、4村民 = 9人
    roles = [
        Role.WEREWOLF, Role.WEREWOLF,
        Role.SEER, Role.WITCH, Role.HUNTER,
        Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
    ]

    agents = []
    for name, role in zip(player_names, roles):
        if role == Role.VILLAGER:
            agent = VillagerAgent(name, llm)
        elif role == Role.WEREWOLF:
            agent = WerewolfAgent(name, llm)
        elif role == Role.SEER:
            agent = SeerAgent(name, llm)
        elif role == Role.WITCH:
            agent = WitchAgent(name, llm)
        elif role == Role.HUNTER:
            agent = HunterAgent(name, llm)
        else:
            agent = VillagerAgent(name, llm)
        agents.append(agent)

    return agents


def main():
    """主函数"""
    st.title("🐺 AI Werewolf - 狼人杀多智能体博弈")

    init_session_state()

    # 侧边栏配置
    with st.sidebar:
        st.header("游戏配置")

        player_count = st.selectbox("玩家数量", [6, 9, 12], index=1)

        player_names = []
        for i in range(player_count):
            name = st.text_input(f"玩家 {i+1} 名称", value=f"玩家{i+1}")
            player_names.append(name)

        model = st.selectbox(
            "LLM 模型",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "claude-3-5-sonnet-20241002"],
        )

        if st.button("开始游戏", type="primary"):
            if all(player_names):
                llm = OpenAILLM(model=model)
                agents = create_agents(player_names, llm)

                config = GameConfig(player_names=player_names)
                game = WerewolfGame(config, agents)

                st.session_state.game = game
                st.session_state.players = player_names
                st.session_state.game_started = True
                st.rerun()
            else:
                st.error("请填写所有玩家名称")

    # 主内容区
    if st.session_state.game_started:
        tab1, tab2 = st.tabs(["游戏面板", "聊天记录"])

        with tab1:
            render_game_board(st.session_state.game)

        with tab2:
            render_chat_panel(st.session_state.logs)
    else:
        st.info("👈 请在侧边栏配置游戏并点击开始")


if __name__ == "__main__":
    main()