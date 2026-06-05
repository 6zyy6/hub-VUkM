"""辅助函数"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from ..agents.base import Role


def format_role(role: Role) -> str:
    """格式化角色名称"""
    role_names = {
        Role.VILLAGER: "村民",
        Role.WEREWOLF: "狼人",
        Role.SEER: "预言家",
        Role.WITCH: "女巫",
        Role.HUNTER: "猎人",
    }
    return role_names.get(role, role.value)


def load_game_log(log_path: str) -> Optional[Dict]:
    """加载游戏日志"""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_game_log(log_path: str, data: Dict) -> bool:
    """保存游戏日志"""
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def get_winner_message(winner: str) -> str:
    """获取胜利消息"""
    if winner == "good":
        return "好人胜利！狼人全部被放逐。"
    elif winner == "wolf":
        return "狼人胜利！狼人数量已占优势。"
    else:
        return "游戏结束，无结果。"