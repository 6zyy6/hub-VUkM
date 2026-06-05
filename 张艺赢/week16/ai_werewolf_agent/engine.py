from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

from agents import BaseAgent, SeerAgent, WitchAgent, build_agent
from logger import JsonlLogger
from models import GameResult, Observation, Phase, Player, PublicEvent, Role, Team


DEFAULT_ROLES = [
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.SEER,
    Role.WITCH,
    Role.GUARD,
    Role.HUNTER,
    Role.VILLAGER,
    Role.VILLAGER,
]


class WerewolfGameEngine:
    def __init__(
        self,
        roles: list[Role] | None = None,
        seed: int = 42,
        log_path: str | Path = "logs/game.jsonl",
    ) -> None:
        self.seed = seed
        self.random = random.Random(seed)
        self.day = 0
        self.public_events: list[PublicEvent] = []
        self.logger = JsonlLogger(log_path)
        self.players = self._create_players(roles or DEFAULT_ROLES)
        self.agents: dict[int, BaseAgent] = {
            player.id: build_agent(player, seed) for player in self.players
        }

    def _create_players(self, roles: list[Role]) -> list[Player]:
        shuffled = list(roles)
        self.random.shuffle(shuffled)
        return [
            Player(id=index + 1, name=f"P{index + 1}", role=role)
            for index, role in enumerate(shuffled)
        ]

    def run(self, max_days: int = 8) -> GameResult:
        self.logger.log(0, "setup", "roles_assigned", roles=self.public_roles(hidden=True))
        while self.day < max_days:
            self.day += 1
            self.run_night()
            result = self.check_winner()
            if result:
                return result
            self.run_day()
            result = self.check_winner()
            if result:
                return result

        return self._result(Team.WEREWOLVES, "Reached max days; wolves win by chaos.")

    def run_night(self) -> None:
        phase = Phase.NIGHT
        self._public(phase, "night_start", f"Night {self.day} begins.")
        for player in self.players:
            player.protected = False
            player.poisoned = False

        wolf_target = self._wolf_kill_target()
        guarded = self._guard_target()
        seer_check = self._seer_check()
        victim = self._player(wolf_target) if wolf_target else None
        saved = self._witch_save(victim)
        poisoned = self._witch_poison()

        deaths: list[Player] = []
        if victim and victim.id != guarded and not saved:
            deaths.append(victim)
        if poisoned:
            poison_target = self._player(poisoned)
            if poison_target and poison_target.alive and poison_target not in deaths:
                deaths.append(poison_target)

        for death in deaths:
            death.alive = False

        self.logger.log(
            self.day,
            phase,
            "night_actions",
            wolf_target=wolf_target,
            guarded=guarded,
            seer_check=seer_check,
            witch_saved=saved,
            poisoned=poisoned,
            deaths=[p.id for p in deaths],
        )
        message = "No one died last night." if not deaths else "Night deaths: " + ", ".join(p.name for p in deaths)
        self._public(phase, "night_result", message, deaths=[p.id for p in deaths])

    def run_day(self) -> None:
        self._public(Phase.DAY, "day_start", f"Day {self.day} discussion begins.")
        speeches: list[dict[str, str | int]] = []
        for player in self.alive_players():
            speech = self.agents[player.id].speak(self._observe(player))
            speeches.append({"player_id": player.id, "speech": speech})
            self.logger.log(self.day, Phase.SPEECH, "speech", player_id=player.id, speech=speech)

        votes = [self.agents[p.id].vote(self._observe(p)) for p in self.alive_players()]
        for vote in votes:
            self.logger.log(
                self.day,
                Phase.VOTE,
                "vote",
                voter_id=vote.voter_id,
                target_id=vote.target_id,
                reason=vote.reason,
            )
        target_id = self._vote_result([vote.target_id for vote in votes])
        executed = self._player(target_id)
        if executed:
            executed.alive = False
            self._public(
                Phase.VOTE,
                "execution",
                f"{executed.name} was voted out.",
                player_id=executed.id,
                role=executed.role.value,
            )

    def _wolf_kill_target(self) -> int | None:
        wolves = [p for p in self.alive_players() if p.role == Role.WEREWOLF]
        if not wolves:
            return None
        targets = []
        for wolf in wolves:
            decision = self.agents[wolf.id].night_action(self._observe(wolf))
            if decision.target_id:
                targets.append(decision.target_id)
        return self._vote_result(targets)

    def _guard_target(self) -> int | None:
        guard = self._alive_role(Role.GUARD)
        if not guard:
            return None
        decision = self.agents[guard.id].night_action(self._observe(guard))
        if decision.target_id:
            protected = self._player(decision.target_id)
            if protected:
                protected.protected = True
            return decision.target_id
        return None

    def _seer_check(self) -> int | None:
        seer = self._alive_role(Role.SEER)
        if not seer:
            return None
        agent = self.agents[seer.id]
        decision = agent.night_action(self._observe(seer))
        if decision.target_id:
            target = self._player(decision.target_id)
            if target and isinstance(agent, SeerAgent):
                agent.remember_check(target.id, target.team)
                self.logger.log(
                    self.day,
                    Phase.NIGHT,
                    "seer_result",
                    seer_id=seer.id,
                    target_id=target.id,
                    team=target.team.value,
                )
            return decision.target_id
        return None

    def _witch_save(self, victim: Player | None) -> bool:
        witch = self._alive_role(Role.WITCH)
        if not witch:
            return False
        agent = self.agents[witch.id]
        return isinstance(agent, WitchAgent) and agent.decide_save(victim)

    def _witch_poison(self) -> int | None:
        witch = self._alive_role(Role.WITCH)
        if not witch:
            return None
        agent = self.agents[witch.id]
        decision = agent.night_action(self._observe(witch))
        return decision.target_id if decision.action == "poison" else None

    def _vote_result(self, target_ids: list[int]) -> int | None:
        valid = [target for target in target_ids if self._player(target) and self._player(target).alive]
        if not valid:
            return None
        counts = Counter(valid)
        top_count = max(counts.values())
        tied = [target for target, count in counts.items() if count == top_count]
        return sorted(tied)[0]

    def _observe(self, player: Player) -> Observation:
        alive = [self._copy_public_player(p, player) for p in self.alive_players()]
        known_wolves: list[int] = []
        seer_checks: dict[int, Team] = {}
        private_facts = [f"Your role is {player.role.value}."]

        if player.role == Role.WEREWOLF:
            known_wolves = [p.id for p in self.players if p.role == Role.WEREWOLF]
            private_facts.append(f"Wolf teammates: {known_wolves}.")

        agent = self.agents[player.id]
        if isinstance(agent, SeerAgent):
            seer_checks = dict(agent.checks)
            private_facts.append(f"Your checks: {seer_checks}.")

        return Observation(
            self_player=player,
            alive_players=alive,
            public_events=list(self.public_events),
            private_facts=private_facts,
            known_wolves=known_wolves,
            seer_checks=seer_checks,
        )

    def _copy_public_player(self, player: Player, observer: Player) -> Player:
        visible_role = player.role if player.id == observer.id or observer.role == Role.WEREWOLF else Role.VILLAGER
        return Player(id=player.id, name=player.name, role=visible_role, alive=player.alive)

    def _public(self, phase: Phase, event_type: str, message: str, **data: object) -> None:
        event = PublicEvent(self.day, phase, event_type, message, dict(data))
        self.public_events.append(event)
        self.logger.log(self.day, phase, event_type, message=message, **data)

    def check_winner(self) -> GameResult | None:
        alive = self.alive_players()
        wolves = [p for p in alive if p.role == Role.WEREWOLF]
        villagers = [p for p in alive if p.role != Role.WEREWOLF]
        if not wolves:
            return self._result(Team.VILLAGERS, "All wolves were eliminated.")
        if len(wolves) >= len(villagers):
            return self._result(Team.WEREWOLVES, "Wolves reached parity with villagers.")
        return None

    def _result(self, winner: Team, reason: str) -> GameResult:
        result = GameResult(
            winner=winner,
            days=self.day,
            survivors=[p.id for p in self.alive_players()],
            reason=reason,
            metrics={
                "alive_wolves": len([p for p in self.alive_players() if p.role == Role.WEREWOLF]),
                "alive_villagers": len([p for p in self.alive_players() if p.role != Role.WEREWOLF]),
                "total_events": len(self.public_events),
            },
        )
        self.logger.log(self.day, Phase.GAME_OVER, "game_over", winner=winner.value, reason=reason)
        return result

    def alive_players(self) -> list[Player]:
        return [player for player in self.players if player.alive]

    def _alive_role(self, role: Role) -> Player | None:
        return next((p for p in self.alive_players() if p.role == role), None)

    def _player(self, player_id: int | None) -> Player | None:
        if player_id is None:
            return None
        return next((p for p in self.players if p.id == player_id), None)

    def public_roles(self, hidden: bool = False) -> dict[int, str]:
        return {
            player.id: ("hidden" if hidden else player.role.value)
            for player in self.players
        }
