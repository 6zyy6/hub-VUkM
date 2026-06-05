from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation import run_tournament


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Werewolf Agent Team homework demo.")
    parser.add_argument("--games", type=int, default=3, help="Number of games to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-dir", default="logs", help="Directory for JSONL logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_tournament(args.games, args.seed, Path(args.log_dir))
    print(json.dumps(summary["leaderboard"], ensure_ascii=False, indent=2))
    print(f"Logs written to: {Path(args.log_dir).resolve()}")


if __name__ == "__main__":
    main()
