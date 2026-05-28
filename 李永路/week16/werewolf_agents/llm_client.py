"""
LLM 客户端封装
支持多种大语言模型（OpenAI、Qwen、DeepSeek 等）
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """LLM 客户端基类"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen3.6-flash"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API Key 未设置，请通过参数或环境变量 OPENAI_API_KEY 提供")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error]: {str(e)}"


class QwenClient(BaseLLMClient):
    """通义千问 API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen3.6-plus"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("DashScope API Key 未设置，请通过参数或环境变量 DASHSCOPE_API_KEY 提供")
        
        try:
            import dashscope
            dashscope.api_key = self.api_key
            self.dashscope = dashscope
        except ImportError:
            raise ImportError("请安装 dashscope 包: pip install dashscope")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> str:
        from dashscope import Generation
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_length=max_tokens,
                result_format='message'
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content.strip()
            else:
                return f"[LLM Error]: {response.message}"
        except Exception as e:
            return f"[LLM Error]: {str(e)}"


class DeepSeekClient(BaseLLMClient):
    """DeepSeek API 客户端"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未设置，请通过参数或环境变量 DEEPSEEK_API_KEY 提供")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1"
            )
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: int = 500, **kwargs) -> str:
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error]: {str(e)}"


class MockLLMClient(BaseLLMClient):
    """模拟 LLM 客户端（用于测试，无需 API Key）"""
    
    def __init__(self):
        self.responses = {
            "night_action": "我选择行动",
            "day_speech": "我认为我们需要更多信息来判断局势。",
            "vote": "我投票给可疑的玩家。"
        }
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # 简单的基于关键词的响应
        if "夜间" in prompt or "kill" in prompt.lower():
            return "1"  # 返回目标玩家ID
        elif "发言" in prompt or "speech" in prompt.lower():
            return "我觉得当前局势还不明朗，需要大家多分享信息。"
        elif "投票" in prompt or "vote" in prompt.lower():
            return "2"  # 返回投票目标ID
        else:
            return "我需要更多信息来做决定。"


def create_llm_client(provider: str = "mock", **kwargs) -> BaseLLMClient:
    """
    工厂函数：创建 LLM 客户端
    
    Args:
        provider: LLM 提供商 ("openai", "qwen", "deepseek", "mock")
        **kwargs: 传递给客户端的参数
        
    Returns:
        LLM 客户端实例
    """
    providers = {
        "openai": OpenAIClient(api_key=""),
        "qwen": QwenClient,
        "deepseek": DeepSeekClient,
        "mock": MockLLMClient,
    }
    
    client_class = providers.get(provider.lower())
    if not client_class:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")
    
    return client_class(**kwargs)
