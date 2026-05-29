"""
Agent基类模块
实现基于LLM的智能Agent核心逻辑
"""

from typing import Optional, List, Dict
from abc import ABC, abstractmethod
import json


class BaseAgent(ABC):
    """智能Agent基类"""

    def __init__(self, player_id: int, name: str, role: str,
                 llm_client=None, enable_memory: bool = True,
                 memory_size: int = 50):
        self.player_id = player_id
        self.name = name
        self.role = role
        self.llm_client = llm_client
        self.enable_memory = enable_memory
        self.memory_size = memory_size
        self.memory: List[Dict] = []
        self.strategy_notes: str = ""

    def add_to_memory(self, event: Dict):
        """添加事件到记忆"""
        if self.enable_memory:
            self.memory.append(event)
            if len(self.memory) > self.memory_size:
                self.memory = self.memory[-self.memory_size:]

    def get_context(self, game_state: Dict) -> str:
        """构建上下文信息"""
        context = f"你是{self.name}，扮演{self.role}角色。\n\n"
        context += "当前游戏状态：\n"
        context += f"- 轮次：{game_state.get('current_round', 0)}\n"
        context += f"- 阶段：{game_state.get('current_phase', 'unknown')}\n"
        context += f"- 存活玩家：{len(game_state.get('alive_players', []))}\n\n"

        if self.memory:
            context += "你的记忆：\n"
            for i, mem in enumerate(self.memory[-5:], 1):
                context += f"{i}. {mem.get('description', '')}\n"
            context += "\n"

        if self.strategy_notes:
            context += f"你的策略笔记：{self.strategy_notes}\n\n"

        return context

    def call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """调用LLM"""
        if self.llm_client:
            return self.llm_client.generate(prompt, system_prompt)
        else:
            return self.default_response(prompt)

    def parse_json_response(self, response: str) -> Dict:
        """解析JSON响应"""
        try:
            return json.loads(response)
        except:
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    return json.loads(json_str)
            except:
                pass

        return {"error": "Failed to parse JSON", "raw": response}

    @abstractmethod
    def night_action(self, game_state: Dict, **kwargs) -> Dict:
        """夜间行动"""
        pass

    @abstractmethod
    def day_speech(self, game_state: Dict, **kwargs) -> str:
        """白天发言"""
        pass

    @abstractmethod
    def voting_decision(self, game_state: Dict, **kwargs) -> int:
        """投票决策"""
        pass

    def default_response(self, prompt: str) -> str:
        """默认响应"""
        return '{"action": "skip", "reason": "No LLM client configured"}'

    def update_strategy(self, analysis: str):
        """更新策略笔记"""
        self.strategy_notes = analysis
