"""
村民Agent实现
"""

from .base_agent import BaseAgent
from typing import Dict
import json


class VillagerAgent(BaseAgent):
    """村民智能Agent"""

    def __init__(self, player_id: int, name: str, llm_client=None, **kwargs):
        super().__init__(player_id, name, "villager", llm_client, **kwargs)

    def night_action(self, game_state: Dict, **kwargs) -> Dict:
        """村民夜间无行动"""
        return {"action": "sleep", "reason": "村民夜间无行动"}

    def day_speech(self, game_state: Dict, **kwargs) -> str:
        """村民白天发言"""
        context = self.get_context(game_state)
        speech_history = kwargs.get('speech_history', [])

        prompt = f"""
{context}
现在是白天发言阶段。作为普通村民，你需要通过逻辑推理找出狼人。

发言历史：
{json.dumps(speech_history[-3:], ensure_ascii=False, indent=2)}

策略建议：
- 认真分析每个玩家的发言
- 注意发言矛盾和逻辑漏洞
- 积极提供自己的观点
- 不要轻易相信所有人的说法

请发表你的发言：
"""

        speech = self.call_llm(prompt)

        self.add_to_memory({
            "phase": "day",
            "action": "speech",
            "content": speech
        })

        return speech

    def voting_decision(self, game_state: Dict, **kwargs) -> int:
        """村民投票决策"""
        context = self.get_context(game_state)
        candidates = kwargs.get('candidates', [])

        prompt = f"""
{context}
现在是投票阶段。作为村民，你的投票至关重要。

候选玩家：
{json.dumps(candidates, ensure_ascii=False)}

请根据你的分析和判断进行投票，返回JSON格式：
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
