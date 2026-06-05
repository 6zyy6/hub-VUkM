"""
猎人Agent实现
"""

from .base_agent import BaseAgent
from typing import Dict, Optional
import json


class HunterAgent(BaseAgent):
    """猎人智能Agent"""

    def __init__(self, player_id: int, name: str, llm_client=None, **kwargs):
        super().__init__(player_id, name, "hunter", llm_client, **kwargs)

    def night_action(self, game_state: Dict, **kwargs) -> Dict:
        """猎人夜间无行动"""
        return {"action": "sleep", "reason": "猎人夜间无行动"}

    def day_speech(self, game_state: Dict, **kwargs) -> str:
        """猎人白天发言"""
        context = self.get_context(game_state)
        speech_history = kwargs.get('speech_history', [])

        prompt = f"""
{context}
现在是白天发言阶段。作为猎人，你可以适当强势一些。

发言历史：
{json.dumps(speech_history[-3:], ensure_ascii=False, indent=2)}

策略建议：
- 可以通过强势发言试探其他玩家反应
- 注意观察谁在针对你
- 死亡时可以带走一人，这是你的威慑力

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
        """猎人投票决策"""
        context = self.get_context(game_state)
        candidates = kwargs.get('candidates', [])

        prompt = f"""
{context}
现在是投票阶段。

候选玩家：
{json.dumps(candidates, ensure_ascii=False)}

请根据你的判断进行投票，返回JSON格式：
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

    def on_death(self, game_state: Dict) -> Optional[Dict]:
        """猎人死亡时的射击技能"""
        context = self.get_context(game_state)
        alive_players = game_state.get('alive_players', [])

        if not alive_players:
            return None

        prompt = f"""
{context}
你已被淘汰！作为猎人，你可以带走一名玩家。

存活玩家：
{json.dumps([{'id': p['player_id'], 'name': p['name']} for p in alive_players], ensure_ascii=False)}

请选择你要带走的玩家，返回JSON格式：
{{
    "shoot_target": <目标ID，-1表示不射击>,
    "reason": "<选择理由>"
}}
"""

        response = self.call_llm(prompt)
        result = self.parse_json_response(response)

        shoot_target = result.get("shoot_target", -1)
        valid_targets = [p['player_id'] for p in alive_players]

        if shoot_target in valid_targets:
            self.add_to_memory({
                "phase": "death",
                "action": "shoot",
                "target": shoot_target,
                "reason": result.get("reason", "")
            })
            return result

        return {"shoot_target": -1, "reason": "放弃射击"}
