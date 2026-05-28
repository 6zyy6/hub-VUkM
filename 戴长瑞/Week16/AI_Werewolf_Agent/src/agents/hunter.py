"""猎人 Agent"""
from typing import Dict

from .base import BaseAgent, Role
from ..prompts.templates import hunter_prompt


class HunterAgent(BaseAgent):
    """猎人 Agent - 死亡时带走一人"""

    def __init__(self, name: str, llm, log_dir: str = "runs/logs"):
        super().__init__(name, Role.HUNTER, llm, log_dir)
        self.can_shoot: bool = True
        self.shoot_target: Optional[str] = None

    async def night_phase(self, game_context: Dict) -> Dict:
        """猎人夜晚准备开枪"""
        return {"action": "wait", "target": None}

    async def day_phase(self, game_context: Dict) -> Dict:
        """猎人白天分析局势"""
        prompt = hunter_prompt(
            player_name=self.name,
            game_context=game_context,
            can_shoot=self.can_shoot,
        )
        speech = await self.speak(prompt)
        return {"action": "speak", "content": speech}

    async def vote(self, game_context: Dict) -> str:
        """猎人投票"""
        living = game_context.get("living_players", [])
        return living[0] if living else ""

    async def on_death(self, game_context: Dict) -> Optional[str]:
        """死亡时带走一人"""
        if not self.can_shoot:
            return None

        living = game_context.get("living_players", [])
        prompt = hunter_prompt(
            player_name=self.name,
            game_context=game_context,
            can_shoot=True,
            is_dying=True,
        )
        target = await self.llm.generate(prompt)
        self.shoot_target = target.strip()
        self.can_shoot = False
        return self.shoot_target