"""
LLM 接口模块
提供统一的 LLM 调用接口，支持 mock / claude / openai 三种模式
"""

from .base import LLMBase
from .openai_llm import OpenAILLM
from .config import load_config, get_llm_config


def create_llm_from_config(config_path=None):
    """
    根据配置文件创建 LLM 实例
    延迟导入以避免缺失依赖导致启动失败

    Returns:
        LLMBase or None: LLM 实例，mock 模式返回 None
    """
    cfg = load_config(config_path)
    llm_cfg = get_llm_config(cfg)
    provider = llm_cfg.get("provider", "mock")

    if provider == "claude":
        from .claude_llm import ClaudeLLM
        return ClaudeLLM(
            model=llm_cfg.get("model", "claude-sonnet-4-20250514"),
            api_key=llm_cfg.get("api_key"),
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 512),
        )
    elif provider == "openai":
        return OpenAILLM(
            model=llm_cfg.get("model", "gpt-4o"),
            api_key=llm_cfg.get("api_key"),
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 512),
        )
    else:
        # mock — 不返回真实 LLM，main.py/ui 会使用 SmartMockLLM
        return None


__all__ = [
    "LLMBase",
    "OpenAILLM",
    "create_llm_from_config",
    "load_config",
    "get_llm_config",
]
