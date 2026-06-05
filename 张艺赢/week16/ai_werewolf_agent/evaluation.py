from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from engine import WerewolfGameEngine
from models import Team


def run_tournament(games: int, seed: int, log_dir: str | Path) -> dict[str, Any]:
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    results = []
    win_counter: Counter[str] = Counter()
    duration_by_winner: defaultdict[str, list[int]] = defaultdict(list)

    for index in range(games):
        game_log = log_root / f"game_{index + 1:03d}.jsonl"
        engine = WerewolfGameEngine(seed=seed + index, log_path=game_log)
        result = engine.run()
        results.append(
            {
                "game": index + 1,
                "winner": result.winner.value,
                "days": result.days,
                "survivors": result.survivors,
                "reason": result.reason,
                "log": str(game_log),
            }
        )
        win_counter[result.winner.value] += 1
        duration_by_winner[result.winner.value].append(result.days)

    leaderboard = [
        {
            "team": team,
            "wins": wins,
            "win_rate": round(wins / games, 3),
            "avg_days": round(sum(duration_by_winner[team]) / wins, 2),
        }
        for team, wins in win_counter.most_common()
    ]

    summary = {
        "games": games,
        "leaderboard": leaderboard,
        "results": results,
        "review": build_review(results),
    }
    summary_path = log_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_review(results: list[dict[str, Any]]) -> dict[str, Any]:
    quick_games = [item for item in results if item["days"] <= 2]
    long_games = [item for item in results if item["days"] >= 4]
    return {
        "process_evaluation": {
            "quick_finish_count": len(quick_games),
            "long_game_count": len(long_games),
            "average_days": round(sum(item["days"] for item in results) / len(results), 2),
        },
        "result_evaluation": {
            "villager_wins": sum(1 for item in results if item["winner"] == Team.VILLAGERS.value),
            "werewolf_wins": sum(1 for item in results if item["winner"] == Team.WEREWOLVES.value),
        },
        "replay_notes": [
            "If wolves win too quickly, improve village speech and voting coordination.",
            "If villagers win too often, improve wolf deception and night target planning.",
            "Use JSONL logs to inspect every night action, speech, vote, and execution.",
        ],
    }
