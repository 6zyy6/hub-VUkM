"""
日志系统模块
提供结构化的游戏日志输出
"""

import sys
from typing import Optional
from datetime import datetime

try:
    from loguru import logger
except ImportError:
    logger = None


class GameLogger:
    """游戏日志记录器"""

    def __init__(self, log_file: Optional[str] = None):
        if logger:
            logger.remove()

            logger.add(
                sys.stderr,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
                level="INFO"
            )

            if log_file:
                logger.add(
                    log_file,
                    rotation="10 MB",
                    encoding="utf-8",
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
                )

        self._logger = logger

    def info(self, message: str):
        """信息日志"""
        if self._logger:
            self._logger.info(message)
        else:
            print(f"[INFO] {message}")

    def success(self, message: str):
        """成功日志"""
        if self._logger:
            self._logger.success(message)
        else:
            print(f"[SUCCESS] {message}")

    def warning(self, message: str):
        """警告日志"""
        if self._logger:
            self._logger.warning(message)
        else:
            print(f"[WARNING] {message}")

    def error(self, message: str):
        """错误日志"""
        if self._logger:
            self._logger.error(message)
        else:
            print(f"[ERROR] {message}")

    def debug(self, message: str):
        """调试日志"""
        if self._logger:
            self._logger.debug(message)
        else:
            print(f"[DEBUG] {message}")
