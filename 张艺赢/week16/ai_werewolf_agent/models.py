from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"
    VILLAGER = "villager"


class Team(str, Enum):
    WEREWOLVES = "werewolves"
    VILLAGERS = "villagers"


class Phase(str, Enum):
    NIGHT = "night"
    DAY = "day"
    SPEECH = "speech"
    VOTE = "vote"
    GAME_OVER = "game_over"


ROLE_TEAM = {
    Role.WEREWOLF: Team.WEREWOLVES,
    Role.SEER: Team.VILLAGERS,
    Role.WITCH: Team.VILLAGERS,
    Role.HUNTER: Team.VILLAGERS,
    Role.GUARD: Team.VILLAGERS,
    Role.VILLAGER: Team.VILLAGERS,
}


@dataclass
class Player:
    id: int
    name: str
    role: Role
    alive: bool = True
    protected: bool = False
    poisoned: bool = False

    @property
    def team(self) -> Team:
        return ROLE_TEAM[self.role]


@dataclass
class PublicEvent:
    day: int
    phase: Phase
    type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivateMemory:
    player_id: int
    facts: list[str] = field(default_factory=list)
    suspicions: dict[int, float] = field(default_factory=dict)

    def add_fact(self, fact: str) -> None:
        self.facts.append(fact)

    def adjust_suspicion(self, player_id: int, delta: float) -> None:
        current = self.suspicions.get(player_id, 0.0)
        self.suspicions[player_id] = max(-1.0, min(1.0, current + delta))


@dataclass
class Observation:
    self_player: Player
    alive_players: list[Player]
    public_events: list[PublicEvent]
    private_facts: list[str]
    known_wolves: list[int] = field(default_factory=list)
    seer_checks: dict[int, Team] = field(default_factory=dict)


@dataclass
class NightActions:
    wolf_kill: int | None = None
    seer_check: int | None = None
    guard_protect: int | None = None
    witch_save: bool = False
    witch_poison: int | None = None


@dataclass
class Vote:
    voter_id: int
    target_id: int
    reason: str


@dataclass
class GameResult:
    winner: Team
    days: int
    survivors: list[int]
    reason: str
    metrics: dict[str, Any]
