"""
DeepSeek LLM实现
"""

from .llm_interface import LLMInterface
from typing import Optional
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class DeepSeekLLM(LLMInterface):
    """DeepSeek LLM客户端"""

    def __init__(self, api_key: Optional[str] = None,
                 model: str = "deepseek-v4-flash",
                 temperature: float = 0.7,
                 max_tokens: int = 500):
        if OpenAI is None:
            raise ImportError("Please install openai package: pip install openai")

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """生成文本"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return self._fallback_response(prompt)

    def chat(self, messages: list) -> str:
        """对话聊天"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API error: {e}")
            return "抱歉，我暂时无法回应。"

    def _fallback_response(self, prompt: str) -> str:
        """降级响应"""
        return '{"action": "skip", "reason": "API unavailable, using fallback"}'
