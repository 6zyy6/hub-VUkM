"""
配置加载模块
从 config.toml 加载 LLM 和游戏配置
"""

import tomllib
from pathlib import Path
from typing import Optional


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.toml"


def load_config(config_path: Optional[Path] = None) -> dict:
    """加载配置文件"""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return _default_config()
    with open(path, "rb") as f:
        return tomllib.load(f)


def _default_config() -> dict:
    """默认配置"""
    return {
        "llm": {
            "provider": "mock",
            "api_key": "",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 512,
            "temperature": 0.7,
        },
        "game": {
            "mode": "smart",
            "max_days": 15,
        },
    }


def get_llm_config(config: Optional[dict] = None) -> dict:
    """获取 LLM 配置"""
    cfg = config or load_config()
    return cfg.get("llm", _default_config()["llm"])


def get_game_config(config: Optional[dict] = None) -> dict:
    """获取游戏配置"""
    cfg = config or load_config()
    return cfg.get("game", _default_config()["game"])
