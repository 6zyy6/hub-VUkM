"""
狼人Agent实现
"""

from .base_agent import BaseAgent
from typing import Dict, List
import json


class WerewolfAgent(BaseAgent):
    """狼人智能Agent"""

    def __init__(self, player_id: int, name: str, llm_client=None, **kwargs):
        super().__init__(player_id, name, "werewolf", llm_client, **kwargs)
        self.teammates: List[int] = []

    def set_teammates(self, teammate_ids: List[int]):
        """设置狼队友"""
        self.teammates = teammate_ids

    def night_action(self, game_state: Dict, **kwargs) -> Dict:
        """狼人夜间行动：选择击杀目标"""
        context = self.get_context(game_state)
        alive_players = game_state.get('alive_players', [])
        candidates = [p for p in alive_players if p['player_id'] not in self.teammates]

        prompt = f"""
{context}
现在是夜晚，你需要和狼队友商量击杀目标。

可选目标：
{json.dumps([{'id': p['player_id'], 'name': p['name']} for p in candidates], ensure_ascii=False)}

狼队友：{self.teammates}

请分析局势并选择击杀目标，以JSON格式返回：
{{
    "target_id": <目标玩家ID>,
    "target_name": "<目标玩家名称>",
    "reason": "<选择理由>"
}}
"""

        response = self.call_llm(prompt)
        result = self.parse_json_response(response)

        self.add_to_memory({
            "phase": "night",
            "action": "kill",
            "target": result.get("target_id"),
            "reason": result.get("reason", "")
        })

        return result

    def day_speech(self, game_state: Dict, **kwargs) -> str:
        """狼人白天发言"""
        context = self.get_context(game_state)
        speech_history = kwargs.get('speech_history', [])

        prompt = f"""
{context}
现在是白天发言阶段。你需要伪装成好人，混淆视听。

发言历史：
{json.dumps(speech_history[-3:], ensure_ascii=False, indent=2)}

请发表你的发言，目标是：
1. 隐藏狼人身份
2. 误导好人投票
3. 保护狼队友（但不要过于明显）

请直接返回发言内容（不要JSON格式）：
"""

        speech = self.call_llm(prompt)

        self.add_to_memory({
            "phase": "day",
            "action": "speech",
            "content": speech
        })

        return speech

    def voting_decision(self, game_state: Dict, **kwargs) -> int:
        """狼人投票决策"""
        context = self.get_context(game_state)
        candidates = kwargs.get('candidates', [])

        prompt = f"""
{context}
现在是投票阶段。你需要投票淘汰一名玩家。

候选玩家：
{json.dumps(candidates, ensure_ascii=False)}

策略：
- 优先投票给暴露风险高的好人
- 如果狼队友被怀疑，可以投票给其他好人转移视线
- 避免连续多轮投票给同一阵营

请返回JSON格式：
{{
    "vote_target": <投票目标ID>,
    "reason": "<投票理由>"
}}
"""

        response = self.call_llm(prompt)
        result = self.parse_json_response(response)

        vote_target = result.get("vote_target")
        if vote_target and vote_target in [c['player_id'] for c in candidates]:
            return vote_target

        return candidates[0]['player_id'] if candidates else -1
