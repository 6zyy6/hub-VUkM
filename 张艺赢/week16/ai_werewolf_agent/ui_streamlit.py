from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="AI Werewolf Observer", layout="wide")
st.title("AI Werewolf Agent Team Observer")

log_dir = Path(st.sidebar.text_input("Log directory", "logs"))
summary_path = log_dir / "summary.json"

if not summary_path.exists():
    st.info("Run `python main.py --games 3` first, then refresh this page.")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
st.metric("Games", summary["games"])

cols = st.columns(len(summary["leaderboard"]) or 1)
for col, row in zip(cols, summary["leaderboard"]):
    col.metric(row["team"], f"{row['wins']} wins", f"{row['win_rate']} win rate")

selected = st.selectbox("Game", summary["results"], format_func=lambda item: f"Game {item['game']} - {item['winner']}")
st.subheader("Result")
st.json(selected)

log_path = Path(selected["log"])
if log_path.exists():
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    st.subheader("Timeline")
    for record in records:
        with st.expander(f"Day {record['day']} | {record['phase']} | {record['type']}"):
            st.json(record["data"])
