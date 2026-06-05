"""
预言家Agent实现
"""

from .base_agent import BaseAgent
from typing import Dict, List
import json


class SeerAgent(BaseAgent):
    """预言家智能Agent"""

    def __init__(self, player_id: int, name: str, llm_client=None, **kwargs):
        super().__init__(player_id, name, "seer", llm_client, **kwargs)
        self.verified_results: Dict[int, str] = {}

    def night_action(self, game_state: Dict, **kwargs) -> Dict:
        """预言家夜间行动：查验身份"""
        context = self.get_context(game_state)
        alive_players = game_state.get('alive_players', [])

        verified_ids = list(self.verified_results.keys())
        unverified = [p for p in alive_players if p['player_id'] not in verified_ids]

        prompt = f"""
{context}
现在是夜晚，你需要查验一名玩家的身份。

未查验的玩家：
{json.dumps([{'id': p['player_id'], 'name': p['name']} for p in unverified], ensure_ascii=False)}

已查验结果：
{json.dumps(self.verified_results, ensure_ascii=False)}

请分析并选择查验目标，以JSON格式返回：
{{
    "target_id": <目标玩家ID>,
    "target_name": "<目标玩家名称>",
    "reason": "<查验理由>"
}}
"""

        response = self.call_llm(prompt)
        result = self.parse_json_response(response)

        self.add_to_memory({
            "phase": "night",
            "action": "verify",
            "target": result.get("target_id"),
            "reason": result.get("reason", "")
        })

        return result

    def receive_verification_result(self, target_id: int, is_werewolf: bool):
        """接收查验结果"""
        self.verified_results[target_id] = "werewolf" if is_werewolf else "good"
        self.add_to_memory({
            "phase": "night_result",
            "action": "verification_result",
            "target": target_id,
            "result": "werewolf" if is_werewolf else "good"
        })

    def day_speech(self, game_state: Dict, **kwargs) -> str:
        """预言家白天发言"""
        context = self.get_context(game_state)
        speech_history = kwargs.get('speech_history', [])
        current_round = game_state.get('current_round', 1)

        prompt = f"""
{context}
现在是白天发言阶段。作为预言家，你需要适当透露身份信息。

查验结果：
{json.dumps(self.verified_results, ensure_ascii=False)}

发言历史：
{json.dumps(speech_history[-3:], ensure_ascii=False, indent=2)}

当前轮次：{current_round}

策略建议：
- 如果查验到狼人，可以跳明身份报出查验结果
- 如果都是好人，可以先隐藏身份观察
- 第2-3轮是跳身份的较好时机
- 注意保护自己，避免被狼人优先击杀

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
        """预言家投票决策"""
        context = self.get_context(game_state)
        candidates = kwargs.get('candidates', [])

        verified_werewolves = [
            pid for pid, result in self.verified_results.items()
            if result == "werewolf" and pid in [c['player_id'] for c in candidates]
        ]

        prompt = f"""
{context}
现在是投票阶段。

已查验的狼人：{verified_werewolves}
候选玩家：
{json.dumps(candidates, ensure_ascii=False)}

策略：
- 优先投票给已查验的狼人
- 如果没有查验到狼人，投票给发言可疑的玩家
- 避免投票给已查验的好人

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

        if verified_werewolves:
            return verified_werewolves[0]

        return candidates[0]['player_id'] if candidates else -1
