"""
狼人杀对局引擎 - Game Engine
支持 12 人标准局：4狼 + 4神(预言家/女巫/猎人/守卫) + 4村民
回合流转、信息隔离、胜负裁决、结构化日志
"""

import random
import time
import json
import enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict


# ============ 枚举定义 ============

class Role(enum.Enum):
    WEREWOLF = "狼人"
    SEER = "预言家"
    WITCH = "女巫"
    HUNTER = "猎人"
    GUARD = "守卫"
    VILLAGER = "村民"

class Team(enum.Enum):
    GOOD = "好人阵营"
    EVIL = "狼人阵营"

class Phase(enum.Enum):
    NIGHT = "夜晚"
    DAY = "白天"
    VOTE = "投票"
    GAME_OVER = "结束"

class GameEvent(enum.Enum):
    GAME_START = "game_start"
    NIGHT_START = "night_start"
    WEREWOLF_KILL = "werewolf_kill"
    SEER_CHECK = "seer_check"
    WITCH_SAVE = "witch_save"
    WITCH_POISON = "witch_poison"
    GUARD_PROTECT = "guard_protect"
    NIGHT_RESULT = "night_result"
    DAY_START = "day_start"
    DISCUSSION = "discussion"
    VOTE_START = "vote_start"
    VOTE_CAST = "vote_cast"
    VOTE_RESULT = "vote_result"
    PLAYER_ELIMINATED = "player_eliminated"
    HUNTER_SHOOT = "hunter_shoot"
    SPEECH = "speech"
    GAME_OVER = "game_over"


# ============ 数据类 ============

@dataclass
class PlayerState:
    player_id: int
    role: Role
    team: Team
    alive: bool = True
    seat_number: int = 0
    name: str = ""
    # 状态标记
    is_sheriff: bool = False
    is_killed: bool = False
    is_poisoned: bool = False
    is_protected: bool = False
    is_lover: bool = False
    # 夜间被投票
    night_votes: int = 0

@dataclass
class GameLog:
    round_number: int = 0
    phase: Phase = Phase.NIGHT
    event: GameEvent = GameEvent.GAME_START
    timestamp: float = field(default_factory=time.time)
    player_id: Optional[int] = None
    target_id: Optional[int] = None
    content: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "round": self.round_number,
            "phase": self.phase.value,
            "event": self.event.value,
            "timestamp": self.timestamp,
            "player_id": self.player_id,
            "target_id": self.target_id,
            "content": self.content,
            "extra": self.extra
        }

@dataclass
class VoteRecord:
    voter_id: int
    target_id: int

# ============ 回调类型 ============
LogCallback = Callable[[GameLog], None]
EventCallback = Callable[[dict], None]


# ============ 游戏引擎 ============

class WerewolfGameEngine:
    """狼人杀对局引擎：管理回合流转、信息隔离、胜负裁决"""

    ROLE_CONFIG_12 = [
        Role.WEREWOLF, Role.WEREWOLF, Role.WEREWOLF, Role.WEREWOLF,
        Role.SEER, Role.WITCH, Role.HUNTER, Role.GUARD,
        Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER,
    ]

    def __init__(self, player_names: Optional[List[str]] = None,
                 log_callback: Optional[LogCallback] = None,
                 event_callback: Optional[EventCallback] = None):
        self.players: Dict[int, PlayerState] = {}
        self.round_number: int = 0
        self.phase: Phase = Phase.NIGHT
        self.logs: List[GameLog] = []
        self.log_callback = log_callback
        self.event_callback = event_callback

        # 夜间行动结果
        self.werewolf_target: Optional[int] = None
        self.seer_check_target: Optional[int] = None
        self.guard_target: Optional[int] = None
        self.witch_save_used: bool = False
        self.witch_poison_used: bool = False
        self.witch_poison_target: Optional[int] = None
        self.witch_save_target: Optional[int] = None
        self.last_guard_target: Optional[int] = None  # 守卫不能连续守同一人

        # 投票系统
        self.current_votes: Dict[int, int] = {}  # voter_id -> target_id
        self.has_voted: Set[int] = set()
        self.sheriff_id: Optional[int] = None
        self.sheriff_votes: Dict[int, int] = {}

        # 猎人
        self.hunter_can_shoot: bool = True

        # 初始化玩家
        names = player_names or [f"Player_{i}" for i in range(12)]
        roles = list(self.ROLE_CONFIG_12)
        random.shuffle(roles)
        for i, role in enumerate(roles):
            pid = i + 1
            team = Team.EVIL if role == Role.WEREWOLF else Team.GOOD
            self.players[pid] = PlayerState(
                player_id=pid, role=role, team=team,
                seat_number=i + 1, name=names[i] if i < len(names) else f"玩家{pid}"
            )

    # ========== 日志与事件 ==========

    def _log(self, event: GameEvent, player_id: Optional[int] = None,
             target_id: Optional[int] = None, content: str = "", **extra):
        log = GameLog(
            round_number=self.round_number,
            phase=self.phase,
            event=event,
            player_id=player_id,
            target_id=target_id,
            content=content,
            extra=extra
        )
        self.logs.append(log)
        if self.log_callback:
            self.log_callback(log)
        return log

    def _emit(self, event_type: str, data: dict):
        if self.event_callback:
            self.event_callback({"type": event_type, **data})

    # ========== 查询接口 ==========

    def get_alive_players(self) -> List[PlayerState]:
        return [p for p in self.players.values() if p.alive]

    def get_alive_ids(self) -> List[int]:
        return [p.player_id for p in self.get_alive_players()]

    def get_players_by_role(self, role: Role) -> List[PlayerState]:
        return [p for p in self.players.values() if p.role == role]

    def get_werewolves(self) -> List[PlayerState]:
        return [p for p in self.players.values() if p.role == Role.WEREWOLF]

    def get_good_team(self) -> List[PlayerState]:
        return [p for p in self.players.values() if p.team == Team.GOOD]

    def get_player(self, pid: int) -> Optional[PlayerState]:
        return self.players.get(pid)

    def check_win(self) -> Optional[Team]:
        """检查胜负：狼人全灭→好人胜；好人数≤狼人数→狼人胜"""
        alive_wolves = [p for p in self.get_werewolves() if p.alive]
        alive_good = [p for p in self.get_good_team() if p.alive]
        if len(alive_wolves) == 0:
            return Team.GOOD
        if len(alive_wolves) >= len(alive_good):
            return Team.EVIL
        return None

    def game_state_dict(self) -> dict:
        return {
            "round": self.round_number,
            "phase": self.phase.value,
            "players": [
                {
                    "id": p.player_id,
                    "name": p.name,
                    "seat": p.seat_number,
                    "alive": p.alive,
                    "role": p.role.value if not p.alive else "???",
                    "team": p.team.value if not p.alive else "???",
                    "is_sheriff": p.is_sheriff,
                }
                for p in sorted(self.players.values(), key=lambda x: x.player_id)
            ],
            "alive_count": len(self.get_alive_ids()),
            "winning_team": None,
            "sheriff_id": self.sheriff_id,
        }

    def public_state_dict(self, player_id: int) -> dict:
        """返回特定玩家视角的公开状态（信息隔离）"""
        player = self.players.get(player_id)
        result = self.game_state_dict()
        if player:
            result["my_role"] = player.role.value
            result["my_team"] = player.team.value
        return result

    # ========== 游戏流程 ==========

    def start_game(self) -> List[GameLog]:
        """开始新游戏"""
        self.round_number = 0
        self.phase = Phase.NIGHT

        # 竞选警长
        self._elect_sheriff()

        self._log(GameEvent.GAME_START, content="游戏开始！天黑请闭眼。")
        self._emit("game_state", self.game_state_dict())

        return self.logs

    def _elect_sheriff(self):
        """竞选警长：随机选一位活着的玩家"""
        candidates = self.get_alive_ids()
        if candidates:
            self.sheriff_id = random.choice(candidates)
            self.players[self.sheriff_id].is_sheriff = True
            self._log(GameEvent.GAME_START, player_id=self.sheriff_id,
                      content=f"玩家{self.sheriff_id}当选警长，投票权重+1")

    def run_night_phase(self) -> dict:
        """执行夜晚阶段：狼人刀人→预言家查验→女巫用药→守卫守护"""
        self.round_number += 1
        self.phase = Phase.NIGHT
        self._log(GameEvent.NIGHT_START, content=f"第{self.round_number}天夜晚降临")

        # 重置夜间状态
        for p in self.players.values():
            p.is_killed = False
            p.is_poisoned = False
            p.is_protected = False

        self.werewolf_target = None
        self.seer_check_target = None
        self.witch_poison_target = None
        self.witch_save_target = None
        self.guard_target = None

        night_actions = {
            "werewolf_target": self.werewolf_target,
            "guard_target": self.guard_target,
            "seer_check_target": self.seer_check_target,
            "witch_actions": {"save": self.witch_save_target, "poison": self.witch_poison_target},
        }
        self._emit("night_phase", {"round": self.round_number, "actions_needed": night_actions})

        return night_actions

    def set_werewolf_kill(self, target_id: int):
        """狼人选择击杀目标"""
        if self.phase != Phase.NIGHT:
            return
        self.werewolf_target = target_id
        self._log(GameEvent.WEREWOLF_KILL, target_id=target_id,
                  content=f"狼人决定击杀玩家{target_id}")

    def set_seer_check(self, target_id: int):
        """预言家查验"""
        if self.phase != Phase.NIGHT:
            return
        self.seer_check_target = target_id
        target = self.players.get(target_id)
        if target:
            is_wolf = target.role == Role.WEREWOLF
            self._log(GameEvent.SEER_CHECK, target_id=target_id,
                      content=f"预言家查验玩家{target_id}：{'狼人' if is_wolf else '好人'}")
            return is_wolf
        return None

    def set_guard_protect(self, target_id: int):
        """守卫守护"""
        if self.phase != Phase.NIGHT:
            return
        if target_id == self.last_guard_target:
            return False  # 不能连续守护同一人
        self.guard_target = target_id
        self.last_guard_target = target_id
        self._log(GameEvent.GUARD_PROTECT, target_id=target_id,
                  content=f"守卫守护玩家{target_id}")
        return True

    def set_witch_save(self, target_id: int):
        """女巫使用解药"""
        if self.phase != Phase.NIGHT or self.witch_save_used:
            return False
        self.witch_save_used = True
        self.witch_save_target = target_id
        self._log(GameEvent.WITCH_SAVE, target_id=target_id,
                  content=f"女巫使用解药救活玩家{target_id}")
        return True

    def set_witch_poison(self, target_id: int):
        """女巫使用毒药"""
        if self.phase != Phase.NIGHT or self.witch_poison_used:
            return False
        self.witch_poison_used = True
        self.witch_poison_target = target_id
        self._log(GameEvent.WITCH_POISON, target_id=target_id,
                  content=f"女巫使用毒药毒杀玩家{target_id}")
        return True

    def resolve_night(self) -> List[int]:
        """结算夜晚结果，返回死亡玩家列表"""
        dead_players: Set[int] = set()

        # 狼人刀人
        if self.werewolf_target:
            target = self.players.get(self.werewolf_target)
            if target and target.alive:
                # 守卫守护判定
                if self.guard_target == self.werewolf_target:
                    self._log(GameEvent.NIGHT_RESULT, target_id=self.werewolf_target,
                              content=f"守卫成功守护玩家{self.werewolf_target}")
                elif self.witch_save_target == self.werewolf_target:
                    # 女巫救活
                    self._log(GameEvent.NIGHT_RESULT, target_id=self.werewolf_target,
                              content=f"女巫救活玩家{self.werewolf_target}")
                else:
                    target.is_killed = True

        # 女巫毒药
        if self.witch_poison_target:
            target = self.players.get(self.witch_poison_target)
            if target and target.alive:
                target.is_poisoned = True

        # 收集死亡
        for p in self.get_alive_players():
            if p.is_killed:
                dead_players.add(p.player_id)
            if p.is_poisoned:
                dead_players.add(p.player_id)

        # 执行死亡
        night_dead = []
        for pid in dead_players:
            self.players[pid].alive = False
            night_dead.append(pid)
            self._log(GameEvent.PLAYER_ELIMINATED, target_id=pid,
                      content=f"玩家{pid}（{self.players[pid].role.value}）在夜晚死亡")

        self._emit("night_result", {
            "dead_players": night_dead,
            "round": self.round_number,
        })

        return night_dead

    def start_day_phase(self, night_dead: List[int]) -> dict:
        """开始白天阶段：公布死者，进入讨论"""
        self.phase = Phase.DAY
        self._log(GameEvent.DAY_START, content=f"天亮了。死亡玩家：{night_dead if night_dead else '无'}")

        winner = self.check_win()
        if winner:
            self.phase = Phase.GAME_OVER
            self._log(GameEvent.GAME_OVER, content=f"游戏结束！{winner.value}获胜！")
            self._emit("game_over", {"winner": winner.value})
            return {"phase": "game_over", "winner": winner.value}

        # 猎人死亡触发技能
        for pid in night_dead:
            player = self.players[pid]
            if player.role == Role.HUNTER and self.hunter_can_shoot:
                self._log(GameEvent.HUNTER_SHOOT, player_id=pid,
                          content=f"猎人{pid}死亡，可以开枪")

        self._emit("day_phase", {"round": self.round_number, "dead": night_dead})
        return {"phase": "day", "dead": night_dead}

    def hunter_shoot(self, hunter_id: int, target_id: int) -> bool:
        """猎人开枪"""
        hunter = self.players.get(hunter_id)
        target = self.players.get(target_id)
        if not hunter or hunter.role != Role.HUNTER or not self.hunter_can_shoot:
            return False
        if not target or not target.alive:
            return False
        target.alive = False
        self.hunter_can_shoot = False
        self._log(GameEvent.HUNTER_SHOOT, player_id=hunter_id, target_id=target_id,
                  content=f"猎人开枪带走玩家{target_id}（{target.role.value}）")
        self._log(GameEvent.PLAYER_ELIMINATED, target_id=target_id,
                  content=f"玩家{target_id}被猎人带走")
        return True

    def start_vote(self):
        """开始投票阶段"""
        self.phase = Phase.VOTE
        self.current_votes.clear()
        self.has_voted.clear()
        self._log(GameEvent.VOTE_START, content="开始投票放逐")
        self._emit("vote_start", {"round": self.round_number})

    def cast_vote(self, voter_id: int, target_id: int) -> bool:
        """玩家投票"""
        if self.phase != Phase.VOTE:
            return False
        if voter_id in self.has_voted:
            return False
        voter = self.players.get(voter_id)
        target = self.players.get(target_id)
        if not voter or not voter.alive:
            return False
        if not target or not target.alive:
            return False
        # 警长投票权重
        weight = 2 if voter.is_sheriff else 1
        self.current_votes[voter_id] = target_id
        self.has_voted.add(voter_id)
        self._log(GameEvent.VOTE_CAST, player_id=voter_id, target_id=target_id,
                  content=f"玩家{voter_id}投票给玩家{target_id}", extra={"weight": weight})

        self._emit("vote_update", {
            "voter_id": voter_id,
            "target_id": target_id,
            "weight": weight,
            "votes": self._tally_votes()
        })
        return True

    def all_voted(self) -> bool:
        alive = set(self.get_alive_ids())
        return alive.issubset(self.has_voted)

    def _tally_votes(self) -> Dict[int, int]:
        """统计投票权重"""
        tally = defaultdict(int)
        for voter_id, target_id in self.current_votes.items():
            voter = self.players.get(voter_id)
            weight = 2 if (voter and voter.is_sheriff) else 1
            tally[target_id] += weight
        return dict(tally)

    def resolve_vote(self) -> Optional[int]:
        """结算投票，返回被放逐的玩家ID"""
        tally = self._tally_votes()
        if not tally:
            return None

        max_votes = max(tally.values())
        top_candidates = [pid for pid, v in tally.items() if v == max_votes]

        eliminated = None
        if len(top_candidates) == 1:
            eliminated = top_candidates[0]
            # 警长平票时警长决定
        elif len(top_candidates) > 1 and self.sheriff_id:
            eliminated = random.choice(top_candidates)

        if eliminated:
            self.players[eliminated].alive = False
            self._log(GameEvent.VOTE_RESULT, target_id=eliminated,
                      content=f"投票结果：玩家{eliminated}（{self.players[eliminated].role.value}）被放逐",
                      extra={"tally": tally})
            self._log(GameEvent.PLAYER_ELIMINATED, target_id=eliminated,
                      content=f"玩家{eliminated}被放逐出局")

        self._emit("vote_result", {
            "eliminated": eliminated,
            "tally": tally,
            "role_reveal": self.players[eliminated].role.value if eliminated else None,
        })

        # 被放逐的是猎人
        if eliminated and self.players[eliminated].role == Role.HUNTER and self.hunter_can_shoot:
            self._log(GameEvent.HUNTER_SHOOT, player_id=eliminated,
                      content=f"猎人{eliminated}被放逐，可以开枪")

        return eliminated

    def end_round(self):
        """结束当前回合"""
        winner = self.check_win()
        if winner:
            self.phase = Phase.GAME_OVER
            self._log(GameEvent.GAME_OVER, content=f"游戏结束！{winner.value}获胜！")
            self._emit("game_over", {"winner": winner.value})
            return winner
        return None

    def get_logs_json(self) -> str:
        return json.dumps([log.to_dict() for log in self.logs], ensure_ascii=False)

    def get_summary(self) -> dict:
        return {
            "total_rounds": self.round_number,
            "winner": self.check_win(),
            "players": [
                {
                    "id": p.player_id, "name": p.name, "role": p.role.value,
                    "team": p.team.value, "alive": p.alive
                }
                for p in self.players.values()
            ],
            "events_count": len(self.logs),
        }
