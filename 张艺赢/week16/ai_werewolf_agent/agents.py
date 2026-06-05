from __future__ import annotations

import random
from dataclasses import dataclass

from models import Observation, Player, Role, Team, Vote


@dataclass
class AgentDecision:
    action: str
    target_id: int | None = None
    reason: str = ""
    speech: str = ""


class BaseAgent:
    def __init__(self, player: Player, seed: int = 0) -> None:
        self.player = player
        self.random = random.Random(seed + player.id * 17)

    def night_action(self, observation: Observation) -> AgentDecision:
        return AgentDecision(action="sleep", reason="No night ability.")

    def speak(self, observation: Observation) -> str:
        target = self._most_suspicious(observation)
        if target is None:
            return f"{self.player.name}: I need more information before making a hard claim."
        return (
            f"{self.player.name}: I suspect {target.name}. "
            "Their behavior does not help the village find wolves."
        )

    def vote(self, observation: Observation) -> Vote:
        target = self._most_suspicious(observation) or self._random_alive_other(observation)
        assert target is not None
        return Vote(self.player.id, target.id, "Highest current suspicion.")

    def _random_alive_other(self, observation: Observation) -> Player | None:
        candidates = [p for p in observation.alive_players if p.id != self.player.id]
        return self.random.choice(candidates) if candidates else None

    def _most_suspicious(self, observation: Observation) -> Player | None:
        candidates = [p for p in observation.alive_players if p.id != self.player.id]
        if not candidates:
            return None

        scored: list[tuple[float, Player]] = []
        for candidate in candidates:
            score = 0.0
            if candidate.id in observation.known_wolves:
                score += 2.0
            if candidate.id in observation.seer_checks:
                score += 1.5 if observation.seer_checks[candidate.id] == Team.WEREWOLVES else -1.0
            score += self.random.uniform(-0.2, 0.2)
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]


class WerewolfAgent(BaseAgent):
    def night_action(self, observation: Observation) -> AgentDecision:
        candidates = [p for p in observation.alive_players if p.team != Team.WEREWOLVES]
        special_priority = {Role.SEER: 3, Role.WITCH: 2, Role.GUARD: 1, Role.HUNTER: 1}
        candidates.sort(
            key=lambda p: (special_priority.get(p.role, 0), self.random.random()),
            reverse=True,
        )
        target = candidates[0]
        return AgentDecision("kill", target.id, f"Remove high-value village role {target.name}.")

    def speak(self, observation: Observation) -> str:
        target = self._random_alive_villager(observation)
        if target is None:
            return f"{self.player.name}: I am following the public information."
        return (
            f"{self.player.name}: {target.name} feels suspicious to me. "
            "I suggest we compare voting patterns carefully."
        )

    def vote(self, observation: Observation) -> Vote:
        target = self._random_alive_villager(observation) or self._random_alive_other(observation)
        assert target is not None
        return Vote(self.player.id, target.id, "Wolf team wants to remove a villager.")

    def _random_alive_villager(self, observation: Observation) -> Player | None:
        candidates = [p for p in observation.alive_players if p.team != Team.WEREWOLVES]
        return self.random.choice(candidates) if candidates else None


class SeerAgent(BaseAgent):
    def __init__(self, player: Player, seed: int = 0) -> None:
        super().__init__(player, seed)
        self.checks: dict[int, Team] = {}

    def night_action(self, observation: Observation) -> AgentDecision:
        unchecked = [
            p for p in observation.alive_players
            if p.id != self.player.id and p.id not in self.checks
        ]
        target = unchecked[0] if unchecked else self._random_alive_other(observation)
        assert target is not None
        return AgentDecision("check", target.id, f"Check {target.name}'s team.")

    def remember_check(self, target_id: int, team: Team) -> None:
        self.checks[target_id] = team

    def speak(self, observation: Observation) -> str:
        wolves = [pid for pid, team in self.checks.items() if team == Team.WEREWOLVES]
        villagers = [pid for pid, team in self.checks.items() if team == Team.VILLAGERS]
        if wolves:
            return f"{self.player.name}: I found a wolf: player {wolves[-1]}."
        if villagers:
            return f"{self.player.name}: Player {villagers[-1]} looks good from my information."
        return f"{self.player.name}: I am collecting information tonight."

    def vote(self, observation: Observation) -> Vote:
        wolves = [
            p for p in observation.alive_players
            if self.checks.get(p.id) == Team.WEREWOLVES
        ]
        if wolves:
            return Vote(self.player.id, wolves[0].id, "Seer check found wolf.")
        return super().vote(observation)


class WitchAgent(BaseAgent):
    def __init__(self, player: Player, seed: int = 0) -> None:
        super().__init__(player, seed)
        self.has_save = True
        self.has_poison = True

    def decide_save(self, victim: Player | None) -> bool:
        if victim is None or not self.has_save:
            return False
        if victim.role in {Role.SEER, Role.WITCH, Role.GUARD}:
            self.has_save = False
            return True
        return False

    def night_action(self, observation: Observation) -> AgentDecision:
        if not self.has_poison:
            return AgentDecision("hold", reason="Poison already used.")
        target = self._most_suspicious(observation)
        if target and target.team == Team.WEREWOLVES:
            self.has_poison = False
            return AgentDecision("poison", target.id, f"Poison suspected wolf {target.name}.")
        return AgentDecision("hold", reason="No strong poison target.")


class GuardAgent(BaseAgent):
    def night_action(self, observation: Observation) -> AgentDecision:
        priority = [Role.SEER, Role.WITCH, Role.GUARD]
        candidates = [p for p in observation.alive_players if p.team == Team.VILLAGERS]
        candidates.sort(key=lambda p: (priority.index(p.role) if p.role in priority else 99, p.id))
        target = candidates[0]
        return AgentDecision("protect", target.id, f"Protect important role {target.name}.")


class HunterAgent(BaseAgent):
    pass


class VillagerAgent(BaseAgent):
    pass


def build_agent(player: Player, seed: int) -> BaseAgent:
    if player.role == Role.WEREWOLF:
        return WerewolfAgent(player, seed)
    if player.role == Role.SEER:
        return SeerAgent(player, seed)
    if player.role == Role.WITCH:
        return WitchAgent(player, seed)
    if player.role == Role.GUARD:
        return GuardAgent(player, seed)
    if player.role == Role.HUNTER:
        return HunterAgent(player, seed)
    return VillagerAgent(player, seed)
