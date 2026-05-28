"""聊天组件"""
import streamlit as st
from typing import List, Dict


def render_message(player: str, content: str, timestamp: str = ""):
    """渲染消息"""
    st.markdown(f"""
    <div style="padding: 8px; border-left: 3px solid #4CAF50; margin: 5px 0; background: #f5f5f5; border-radius: 4px;">
        <strong>{player}:</strong> {content}
    </div>
    """, unsafe_allow_html=True)


def render_chat_panel(logs: List[Dict]):
    """渲染聊天面板"""
    st.header("聊天记录")

    if not logs:
        st.info("暂无聊天记录")
        return

    for log in logs:
        if log.get("type") == "speech":
            render_message(
                log.get("player", "未知"),
                log.get("content", ""),
                log.get("timestamp", ""),
            )
        elif log.get("type") == "event":
            st.info(log.get("message", ""))

    # 滚动到底部
    # st.session_state.scroll_to_bottom = True