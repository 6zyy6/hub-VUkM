# Week 16 - AI Werewolf Agent Team

This homework implements a multi-agent Werewolf system for information-asymmetric
gameplay. Each agent receives role-limited observations, reasons with its own
goal, speaks, votes, and takes night actions. The game engine controls phase
transitions, death resolution, victory checks, and structured logging.

## Features

- Multi-agent roles: werewolf, seer, witch, guard, hunter, villager
- Information isolation: wolves know teammates; seer knows only checked results
- Full game loop: night actions, day speech, voting, execution, win judgment
- JSONL observability: setup, speeches, votes, night actions, and final result
- Evaluation and replay: tournament summary, win rate, average days, review notes
- Optional observer UI with Streamlit

## Run

```bash
python main.py --games 3 --seed 42 --log-dir logs
```

Optional UI:

```bash
pip install streamlit
streamlit run ui_streamlit.py
```

## Advanced Direction

This project chooses direction 2: evaluation and replay. The `evaluation.py`
module runs multiple games, builds a simple leaderboard, and writes replay
notes that explain what should be improved for future agent versions.
