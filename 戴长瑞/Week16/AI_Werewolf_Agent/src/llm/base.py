"""LLM 基类"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMBase(ABC):
    """LLM 接口基类"""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本"""
        pass

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """聊天"""
        pass

    @abstractmethod
    def set_temperature(self, temperature: float):
        """设置温度"""
        pass

    @abstractmethod
    def set_max_tokens(self, max_tokens: int):
        """设置最大 token 数"""
        pass