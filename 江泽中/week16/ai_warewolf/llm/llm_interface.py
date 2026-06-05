"""
LLM接口抽象模块
定义统一的LLM调用接口
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMInterface(ABC):
    """LLM接口基类"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本

        :param prompt: 用户提示
        :param system_prompt: 系统提示
        :return: 生成的文本
        """
        pass

    @abstractmethod
    def chat(self, messages: list) -> str:
        """
        对话聊天

        :param messages: 消息列表
        :return: 回复文本
        """
        pass
