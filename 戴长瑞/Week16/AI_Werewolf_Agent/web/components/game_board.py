"""游戏面板组件"""
import streamlit as st
from typing import Optional

from src.engine.game import WerewolfGame
from src.agents.base import Role
from src.utils.helpers import format_role


def render_player_card(name: str, role: Role, is_alive: bool, can_speak: bool = True):
    """渲染玩家卡片"""
    status = "🟢 存活" if is_alive else "🔴 死亡"
    speak_status = "💬 可发言" if can_speak else "🔇 禁言"

    color = "green" if is_alive else "red"

    st.markdown(f"""
    <div style="padding: 10px; border: 1px solid {color}; border-radius: 8px; margin: 5px;">
        <h4>{name}</h4>
        <p>{format_role(role)}</p>
        <p>{status} | {speak_status}</p>
    </div>
    """, unsafe_allow_html=True)


def render_phase_info(game: WerewolfGame):
    """渲染阶段信息"""
    col1, col2, col3 = st.columns(3)

    with col1:
        phase = game.phase.current.value
        st.metric("当前阶段", phase)

    with col2:
        living = len(game.living_players)
        st.metric("存活人数", living)

    with col3:
        wolves = len([p for p in game.living_players if p.role == Role.WEREWOLF])
        st.metric("狼人数量", wolves)


def render_game_board(game: WerewolfGame):
    """渲染游戏面板"""
    st.header("当前游戏状态")

    render_phase_info(game)

    st.divider()

    # 存活玩家
    st.subheader("存活玩家")
    living_cols = st.columns(3)

    for i, player in enumerate(game.living_players):
        with living_cols[i % 3]:
            render_player_card(
                player.name,
                player.role,
                player.is_alive,
                player.can_speak,
            )

    # 死亡玩家
    dead_players = [p for p in game.players if not p.is_alive]
    if dead_players:
        st.divider()
        st.subheader("死亡玩家")
        dead_cols = st.columns(3)

        for i, player in enumerate(dead_players):
            with dead_cols[i % 3]:
                render_player_card(
                    player.name,
                    player.role,
                    player.is_alive,
                    False,
                )

    # 操作按钮
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        if st.button("进入夜晚阶段", type="primary"):
            st.info("夜晚阶段处理中...")

    with col2:
        if st.button("进入白天阶段"):
            st.info("白天阶段处理中...")