"""游戏日志"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class GameLogger:
    """游戏日志记录器"""

    def __init__(self, log_dir: str = "runs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"game_{self.game_id}.json"

        self.events: List[Dict] = []

    def log_game_start(self, players: List[str]):
        """记录游戏开始"""
        event = {
            "type": "game_start",
            "timestamp": datetime.now().isoformat(),
            "players": players,
        }
        self.events.append(event)
        self._flush()

    def log_phase(self, phase: str):
        """记录阶段切换"""
        event = {
            "type": "phase",
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
        }
        self.events.append(event)
        self._flush()

    def log_speech(self, player: str, content: str):
        """记录发言"""
        event = {
            "type": "speech",
            "timestamp": datetime.now().isoformat(),
            "player": player,
            "content": content,
        }
        self.events.append(event)

    def log_vote(self, votes: Dict[str, str]):
        """记录投票"""
        event = {
            "type": "vote",
            "timestamp": datetime.now().isoformat(),
            "votes": votes,
        }
        self.events.append(event)
        self._flush()

    def log_death(self, player: str, cause: str):
        """记录死亡"""
        event = {
            "type": "death",
            "timestamp": datetime.now().isoformat(),
            "player": player,
            "cause": cause,
        }
        self.events.append(event)
        self._flush()

    def log_wolf_action(self, actions: Dict):
        """记录狼人行动"""
        event = {
            "type": "wolf_action",
            "timestamp": datetime.now().isoformat(),
            "actions": actions,
        }
        self.events.append(event)
        self._flush()

    def log_seer_action(self, actions: Dict):
        """记录预言家行动"""
        event = {
            "type": "seer_action",
            "timestamp": datetime.now().isoformat(),
            "actions": actions,
        }
        self.events.append(event)
        self._flush()

    def log_witch_action(self, actions: Dict):
        """记录女巫行动"""
        event = {
            "type": "witch_action",
            "timestamp": datetime.now().isoformat(),
            "actions": actions,
        }
        self.events.append(event)
        self._flush()

    def log_event(self, message: str):
        """记录通用事件"""
        event = {
            "type": "event",
            "timestamp": datetime.now().isoformat(),
            "message": message,
        }
        self.events.append(event)
        self._flush()

    def log_win(self, winner: str):
        """记录胜利"""
        event = {
            "type": "win",
            "timestamp": datetime.now().isoformat(),
            "winner": winner,
        }
        self.events.append(event)
        self._flush()

    def log_game_end(self, result: str):
        """记录游戏结束"""
        event = {
            "type": "game_end",
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }
        self.events.append(event)
        self._flush()

    def _flush(self):
        """写入文件"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump({
                "game_id": self.game_id,
                "events": self.events,
            }, f, ensure_ascii=False, indent=2)

    def get_log_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)