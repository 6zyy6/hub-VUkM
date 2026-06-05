"""
狼人杀对局引擎 - Game Engine
完整回合状态机 + 夜晚行动流程 + 白天公投 + 死亡判定 + 胜负裁决 + 结构化日志
"""

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable
from pathlib import Path


# ============================================================
# 枚举定义
# ============================================================

class Phase(Enum):
    """游戏阶段"""
    WAITING = auto()          # 等待开始
    NIGHT_START = auto()      # 夜晚开始
    WOLF_KILL = auto()        # 狼人杀人
    SEER_CHECK = auto()       # 预言家验人
    WITCH_ACTION = auto()     # 女巫用药
    DAY_START = auto()        # 白天开始
    SPEECH = auto()           # 发言阶段
    VOTE = auto()             # 投票阶段
    EXECUTION = auto()        # 处决阶段
    GAME_OVER = auto()         # 游戏结束


class ActionResult(Enum):
    """行动结果"""
    WAIT = "wait"            # 等待
    KILL = "kill"            # 杀害
    CHECK = "check"          # 查验
    HEAL = "heal"            # 救人
    POISON = "poison"        # 毒人
    SPEAK = "speak"          # 发言
    VOTE = "vote"            # 投票
    EXECUTE = "execute"      # 处决


class CauseOfDeath(Enum):
    """死亡原因"""
    WOLF_KILL = "wolf_kill"      # 狼人杀害
    VOTE = "vote"                # 投票处决
    WITCH_POISON = "witch_poison" # 女巫毒杀
    HUNTER_SHOOT = "hunter_shoot" # 猎人开枪


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Player:
    """玩家"""
    id: str
    name: str
    role: 'Role'  # 来自 roles.role_def
    is_alive: bool = True
    can_speak: bool = True
    vote_count: int = 0
    last_word: str = ""

    # 女巫专属
    heal_potion: int = 1
    poison_potion: int = 1
    has_healed_tonight: bool = False
    has_poisoned_tonight: bool = False

    # 预言家专属
    seer_checks: Dict[str, bool] = field(default_factory=dict)  # name -> is_wolf

    # 狼人专属
    wolf_teammates: List[str] = field(default_factory=list)

    # 猎人专属
    can_shoot: bool = True
    shoot_target: Optional[str] = None

    @property
    def is_wolf(self) -> bool:
        return self.role.value == "werewolf"

    @property
    def is_good(self) -> bool:
        return not self.is_wolf

    def new_night(self):
        """新夜晚重置"""
        self.has_healed_tonight = False
        self.has_poisoned_tonight = False

    def reset_vote(self):
        """重置投票"""
        self.vote_count = 0

    def speak(self, content: str):
        """发言"""
        self.last_word = content


@dataclass
class DeathRecord:
    """死亡记录"""
    player: str
    cause: CauseOfDeath
    day: int
    phase: Phase
    killer: Optional[str] = None  # 杀手（狼人/女巫/猎人）


@dataclass
class ActionRecord:
    """行动记录"""
    player: str
    action: ActionResult
    target: Optional[str] = None
    result: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NightActions:
    """夜晚行动结果"""
    wolf_kill_target: Optional[str] = None
    wolf_kill_decided: bool = False

    seer_check_target: Optional[str] = None
    seer_check_result: Optional[bool] = None
    seer_check_decided: bool = False

    witch_heal_target: Optional[str] = None
    witch_poison_target: Optional[str] = None
    witch_heal_decided: bool = False
    witch_poison_decided: bool = False

    dead_players: List[str] = field(default_factory=list)
    death_causes: Dict[str, CauseOfDeath] = field(default_factory=dict)
    vote_map: Dict[str, str] = field(default_factory=dict)  # voter -> target


# ============================================================
# 结构化日志系统
# ============================================================

class GameLogger:
    """游戏日志记录器 - 可观测、可复盘"""

    def __init__(self, log_dir: str = "runs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.game_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()

        self.events: List[Dict] = []
        self.phase_history: List[Dict] = []
        self.action_history: List[ActionRecord] = []
        self.death_history: List[DeathRecord] = []

    def _add_event(self, event_type: str, data: Dict):
        """添加事件"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "game_time": self._get_game_time(),
            **data
        }
        self.events.append(event)

    def _get_game_time(self) -> str:
        """获取游戏内时间"""
        elapsed = datetime.now() - self.start_time
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def log_game_start(self, players: List[Dict]):
        """记录游戏开始"""
        self._add_event("game_start", {
            "game_id": self.game_id,
            "player_count": len(players),
            "players": players,
        })

    def log_phase_change(self, old_phase: Phase, new_phase: Phase, day: int):
        """记录阶段切换"""
        event = {
            "type": "phase_change",
            "from_phase": old_phase.name,
            "to_phase": new_phase.name,
            "day": day,
        }
        self.phase_history.append(event)
        self._add_event("phase_change", event)

    def log_night_start(self, day: int):
        """记录夜晚开始"""
        self._add_event("night_start", {"day": day})

    def log_night_end(self, night_actions: NightActions):
        """记录夜晚结束"""
        self._add_event("night_end", {
            "wolf_kill": night_actions.wolf_kill_target,
            "seer_check": f"{night_actions.seer_check_target} -> {'狼人' if night_actions.seer_check_result else '好人'}",
            "witch_heal": night_actions.witch_heal_target,
            "witch_poison": night_actions.witch_poison_target,
            "deaths": [f"{p} ({night_actions.death_causes.get(p, '').value})" for p in night_actions.dead_players],
        })

    def log_day_start(self, day: int, deaths: List[str]):
        """记录白天开始"""
        self._add_event("day_start", {
            "day": day,
            "night_deaths": deaths,
        })

    def log_speech(self, player: str, content: str, day: int):
        """记录发言"""
        event = {
            "type": "speech",
            "player": player,
            "content": content,
            "day": day,
        }
        self.action_history.append(ActionRecord(player, ActionResult.SPEAK, result=content))
        self._add_event("speech", event)

    def log_vote(self, voter: str, target: str, day: int):
        """记录投票"""
        event = {
            "type": "vote",
            "voter": voter,
            "target": target,
            "day": day,
        }
        self.action_history.append(ActionRecord(voter, ActionResult.VOTE, target=target))
        self._add_event("vote", event)

    def log_execution(self, executed: str, vote_counts: Dict[str, int], day: int):
        """记录处决"""
        self._add_event("execution", {
            "executed": executed,
            "vote_counts": vote_counts,
            "day": day,
        })

    def log_death(self, record: DeathRecord):
        """记录死亡"""
        self.death_history.append(record)
        self._add_event("death", {
            "player": record.player,
            "cause": record.cause.value,
            "day": record.day,
            "killer": record.killer,
        })

    def log_game_over(self, winner: str, reason: str):
        """记录游戏结束"""
        self._add_event("game_over", {
            "winner": winner,
            "reason": reason,
            "total_days": len(set(e.get("day") for e in self.phase_history if "day" in e)),
        })

    def log_event(self, message: str):
        """记录通用事件"""
        self._add_event("event", {"message": message})

    def log_win(self, winner: str):
        """记录胜利"""
        self._add_event("win", {"winner": winner})

    def save(self):
        """保存日志到文件"""
        log_file = self.log_dir / f"game_{self.game_id}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "game_id": self.game_id,
                "start_time": self.start_time.isoformat(),
                "events": self.events,
                "phase_history": self.phase_history,
                "deaths": [
                    {"player": d.player, "cause": d.cause.value, "day": d.day, "killer": d.killer}
                    for d in self.death_history
                ],
            }, f, ensure_ascii=False, indent=2)
        return str(log_file)

    def get_summary(self) -> Dict:
        """获取游戏摘要"""
        return {
            "game_id": self.game_id,
            "total_events": len(self.events),
            "total_deaths": len(self.death_history),
            "phases": len(self.phase_history),
        }


# ============================================================
# 游戏引擎
# ============================================================

class GameEngine:
    """
    狼人杀对局引擎

    状态机流程：
    WAITING → NIGHT_START → WOLF_KILL → SEER_CHECK → WITCH_ACTION
           → DAY_START → SPEECH → VOTE → EXECUTION
           → (循环) 或 GAME_OVER
    """

    def __init__(
        self,
        player_names: List[str],
        role_distribution: Optional[Dict] = None,
        log_dir: str = "runs/logs",
        callback: Optional[Callable] = None,
        role_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        初始化游戏引擎

        Args:
            player_names: 玩家名称列表
            role_distribution: 角色分配 {"werewolf": 2, "villager": 2, "seer": 1, "witch": 1}
            log_dir: 日志目录
            callback: 状态更新回调函数
            role_mapping: 可选，预定义的角色分配 {player_name: role}，如提供则忽略 role_distribution
        """
        self.player_names = player_names
        self.role_distribution = role_distribution or {
            "werewolf": 2,
            "villager": 2,
            "seer": 1,
            "witch": 1,
        }
        self._role_mapping = role_mapping

        self.callback = callback  # 状态更新回调
        self.logger = GameLogger(log_dir)

        # 游戏状态
        self.players: Dict[str, Player] = {}
        self.phase = Phase.WAITING
        self.day = 0
        self.night_actions = NightActions()
        self.is_game_over = False
        self.winner: Optional[str] = None

        # 初始化玩家
        self._init_players()

    def _init_players(self):
        """初始化玩家"""
        # 如果提供了 role_mapping，直接使用它
        if self._role_mapping:
            for name in self.player_names:
                role_str = self._role_mapping.get(name, "villager")
                player = Player(
                    id=str(uuid.uuid4())[:8],
                    name=name,
                    role=self._create_role(role_str),
                )
                self.players[name] = player

                # 狼人知道队友
                if player.is_wolf:
                    player.wolf_teammates = [
                        p.name for p in self.players.values()
                        if p.is_wolf and p.name != name
                    ]

            # 记录游戏开始
            player_info = [
                {"name": p.name, "role": p.role.value}
                for p in self.players.values()
            ]
            self.logger.log_game_start(player_info)
            return

        # 否则从 role_distribution 随机分配
        roles = []
        for role, count in self.role_distribution.items():
            roles.extend([role] * count)

        # 如果玩家数不够，补充村民
        while len(roles) < len(self.player_names):
            roles.append("villager")

        random.shuffle(roles)

        for name, role in zip(self.player_names, roles):
            player = Player(
                id=str(uuid.uuid4())[:8],
                name=name,
                role=self._create_role(role),
            )
            self.players[name] = player

            # 狼人知道队友
            if player.is_wolf:
                player.wolf_teammates = [
                    p.name for p in self.players.values()
                    if p.is_wolf and p.name != name
                ]

        # 记录游戏开始
        player_info = [
            {"name": p.name, "role": p.role.value}
            for p in self.players.values()
        ]
        self.logger.log_game_start(player_info)

    def _create_role(self, role_str: str):
        """创建角色对象"""
        # 简单实现，直接返回角色字符串
        # 实际项目中可以从 roles.role_def 导入 Role
        class Role:
            def __init__(self, value):
                self.value = value
            def __eq__(self, other):
                return self.value == other
        return Role(role_str)

    @property
    def living_players(self) -> List[Player]:
        """获取存活玩家"""
        return [p for p in self.players.values() if p.is_alive]

    @property
    def living_good_players(self) -> List[Player]:
        """获取存活好人"""
        return [p for p in self.living_players if p.is_good]

    @property
    def living_wolf_players(self) -> List[Player]:
        """获取存活狼人"""
        return [p for p in self.living_players if p.is_wolf]

    @property
    def speaking_players(self) -> List[Player]:
        """获取可发言玩家"""
        return [p for p in self.living_players if p.can_speak]

    def _notify(self, event_type: str, data: Dict):
        """通知状态更新"""
        if self.callback:
            self.callback(event_type, data)

    def _set_phase(self, new_phase: Phase):
        """设置阶段"""
        old_phase = self.phase
        self.phase = new_phase
        self.logger.log_phase_change(old_phase, new_phase, self.day)
        self._notify("phase_change", {
            "old": old_phase.name,
            "new": new_phase.name,
            "day": self.day,
        })

    # ============================================================
    # 夜晚行动流程
    # ============================================================

    async def night_phase(self) -> NightActions:
        """
        执行夜晚阶段

        流程：狼人刀人 → 预言家验人 → 女巫用药

        Returns:
            NightActions: 夜晚行动结果
        """
        self.day += 1
        self._set_phase(Phase.NIGHT_START)
        self.logger.log_night_start(self.day)

        # 重置女巫夜晚状态
        for p in self.players.values():
            p.new_night()

        self.night_actions = NightActions()

        # 阶段 1: 狼人杀人
        await self._wolf_kill_phase()

        # 阶段 2: 预言家验人
        await self._seer_check_phase()

        # 阶段 3: 女巫用药
        await self._witch_action_phase()

        # 结算死亡
        self._resolve_night_deaths()

        self._set_phase(Phase.DAY_START)
        self.logger.log_night_end(self.night_actions)

        return self.night_actions

    async def _wolf_kill_phase(self):
        """狼人杀人阶段"""
        self._set_phase(Phase.WOLF_KILL)
        wolves = self.living_wolf_players

        if not wolves:
            return

        # 狼人按顺序行动（如果有多个狼人）
        for wolf in wolves:
            if not wolf.is_alive or not wolf.can_speak:
                continue

            target = await self._get_wolf_decision(wolf)
            if target and target in [p.name for p in self.living_players]:
                # 验证目标不是狼人队友
                target_player = self.players.get(target)
                if target_player and not target_player.is_wolf:
                    self.night_actions.wolf_kill_target = target
                    self.night_actions.wolf_kill_decided = True

        self._notify("wolf_kill", {
            "target": self.night_actions.wolf_kill_target,
            "decided": self.night_actions.wolf_kill_decided,
        })

    async def _seer_check_phase(self):
        """预言家验人阶段"""
        self._set_phase(Phase.SEER_CHECK)
        seers = [p for p in self.living_players if p.role.value == "seer"]

        if not seers:
            return

        seer = seers[0]
        target, result = await self._get_seer_decision(seer)

        if target:
            is_wolf = target in [p.name for p in self.living_wolf_players]
            self.night_actions.seer_check_target = target
            self.night_actions.seer_check_result = is_wolf
            self.night_actions.seer_check_decided = True

            # 记录到预言家个人查验记录
            seer.seer_checks[target] = is_wolf

        self._notify("seer_check", {
            "target": self.night_actions.seer_check_target,
            "result": "wolf" if self.night_actions.seer_check_result else "good",
        })

    async def _witch_action_phase(self):
        """女巫用药阶段"""
        self._set_phase(Phase.WITCH_ACTION)
        witches = [p for p in self.living_players if p.role.value == "witch"]

        if not witches:
            return

        witch = witches[0]

        # 如果有狼人杀人目标，女巫决定是否救
        if self.night_actions.wolf_kill_target and witch.heal_potion > 0:
            heal_target = await self._get_witch_heal_decision(witch)
            if heal_target:
                self.night_actions.witch_heal_target = heal_target
                self.night_actions.witch_heal_decided = True
                witch.heal_potion -= 1
                witch.has_healed_tonight = True

        # 女巫决定是否毒人
        poison_target = await self._get_witch_poison_decision(witch)
        if poison_target:
            self.night_actions.witch_poison_target = poison_target
            self.night_actions.witch_poison_decided = True
            witch.poison_potion -= 1
            witch.has_poisoned_tonight = True

        self._notify("witch_action", {
            "heal_target": self.night_actions.witch_heal_target,
            "poison_target": self.night_actions.witch_poison_target,
        })

    def _resolve_night_deaths(self):
        """结算夜晚死亡"""
        dead = []

        # 狼人杀的人（如果没被女巫救）
        if self.night_actions.wolf_kill_target:
            victim = self.night_actions.wolf_kill_target
            if victim != self.night_actions.witch_heal_target:
                dead.append(victim)
                self.night_actions.death_causes[victim] = CauseOfDeath.WOLF_KILL

        # 女巫毒的人
        if self.night_actions.witch_poison_target:
            poison_victim = self.night_actions.witch_poison_target
            if poison_victim not in dead:
                dead.append(poison_victim)
                self.night_actions.death_causes[poison_victim] = CauseOfDeath.WITCH_POISON

        # 执行死亡
        for name in dead:
            if name in self.players:
                player = self.players[name]
                player.is_alive = False
                player.can_speak = False

                record = DeathRecord(
                    player=name,
                    cause=self.night_actions.death_causes[name],
                    day=self.day,
                    phase=self.phase,
                    killer=self.night_actions.wolf_kill_target if self.night_actions.death_causes[name] == CauseOfDeath.WOLF_KILL else None,
                )
                self.logger.log_death(record)

        self.night_actions.dead_players = dead

    # ============================================================
    # 白天流程
    # ============================================================

    async def day_phase(self) -> str:
        """
        执行白天阶段

        流程：宣布夜晚死亡 → 发言 → 投票 → 处决

        Returns:
            str: 被处决的玩家，空字符串表示平票无人被处决
        """
        # 如果有夜晚死亡，宣布死亡
        night_deaths = self.night_actions.dead_players.copy()

        self._set_phase(Phase.DAY_START)
        self.logger.log_day_start(self.day, night_deaths)
        self._notify("day_start", {"deaths": night_deaths})

        # 检查胜利条件
        if self._check_win_condition():
            return ""

        # 发言阶段
        await self._speech_phase()

        # 投票阶段
        executed = await self._vote_phase()

        return executed

    async def _speech_phase(self):
        """发言阶段"""
        self._set_phase(Phase.SPEECH)

        speakers = self.speaking_players
        for player in speakers:
            if not player.is_alive or not player.can_speak:
                continue

            speech = await self._get_speech(player)
            player.speak(speech)
            self.logger.log_speech(player.name, speech, self.day)
            self._notify("speech", {"player": player.name, "content": speech})

    async def _vote_phase(self) -> str:
        """投票阶段"""
        self._set_phase(Phase.VOTE)

        votes = {}
        speakers = self.speaking_players

        for player in speakers:
            if not player.is_alive or not player.can_speak:
                continue

            target = await self._get_vote(player)
            if target:
                votes[player.name] = target
                self.logger.log_vote(player.name, target, self.day)

        # 统计票数
        self.night_actions.vote_map = votes
        vote_counts: Dict[str, int] = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        for name, count in vote_counts.items():
            if name in self.players:
                self.players[name].vote_count = count

        # 找出最高票
        executed = self._resolve_vote(vote_counts)

        # 重置票数
        for p in self.players.values():
            p.reset_vote()

        return executed

    def _resolve_vote(self, vote_counts: Dict[str, int]) -> str:
        """结算投票，处决最高票"""
        if not vote_counts:
            return ""

        max_votes = max(vote_counts.values())
        candidates = [name for name, count in vote_counts.items() if count == max_votes]

        # 平票，无人处决
        if len(candidates) > 1:
            self.logger.log_event(f"平票，无人出局: {candidates}")
            return ""

        # 处决
        executed = candidates[0]
        if executed in self.players:
            self.players[executed].is_alive = False
            self.players[executed].can_speak = False

            record = DeathRecord(
                player=executed,
                cause=CauseOfDeath.VOTE,
                day=self.day,
                phase=self.phase,
            )
            self.logger.log_death(record)

        self.logger.log_execution(executed, vote_counts, self.day)
        self._notify("execution", {"executed": executed, "votes": vote_counts})

        return executed

    # ============================================================
    # 胜负判定
    # ============================================================

    def _check_win_condition(self) -> bool:
        """检查胜负条件"""
        living_wolves = len(self.living_wolf_players)
        living_goods = len(self.living_good_players)

        if living_wolves == 0:
            self._end_game("good", "所有狼人被放逐")
            return True

        if living_wolves >= living_goods:
            self._end_game("wolf", "狼人数量已占优势")
            return True

        return False

    def _end_game(self, winner: str, reason: str):
        """结束游戏"""
        self.is_game_over = True
        self.winner = winner
        self._set_phase(Phase.GAME_OVER)
        self.logger.log_game_over(winner, reason)
        self._notify("game_over", {"winner": winner, "reason": reason})

    def get_winner(self) -> Optional[str]:
        """获取胜利者"""
        return self.winner

    # ============================================================
    # AI 决策接口（需要子类或外部实现）
    # ============================================================

    async def _get_wolf_decision(self, wolf: Player) -> Optional[str]:
        """获取狼人杀人决策 - 子类重写或外部传入"""
        # 默认：杀第一个活着的非狼人
        for player in self.living_players:
            if not player.is_wolf:
                return player.name
        return None

    async def _get_seer_decision(self, seer: Player) -> tuple:
        """获取预言家验人决策 - 子类重写或外部传入"""
        # 默认：验第一个活着的非预言家
        for player in self.living_players:
            if player.role.value != "seer":
                return player.name, player.is_wolf
        return None, None

    async def _get_witch_heal_decision(self, witch: Player) -> Optional[str]:
        """获取女巫救人决策 - 子类重写或外部传入"""
        # 默认：救狼人要杀的人
        if self.night_actions.wolf_kill_target and witch.heal_potion > 0:
            return self.night_actions.wolf_kill_target
        return None

    async def _get_witch_poison_decision(self, witch: Player) -> Optional[str]:
        """获取女巫毒人决策 - 子类重写或外部传入"""
        # 默认：毒狼人（如果有信息）
        return None

    async def _get_speech(self, player: Player) -> str:
        """获取发言 - 子类重写或外部传入"""
        return f"我是{player.name}，现在进入发言阶段。"

    async def _get_vote(self, player: Player) -> Optional[str]:
        """获取投票决策 - 子类重写或外部传入"""
        # 默认：投第一个活着的非自己
        for p in self.living_players:
            if p.name != player.name:
                return p.name
        return None

    # ============================================================
    # 辅助方法
    # ============================================================

    def get_state(self) -> Dict:
        """获取游戏状态快照"""
        return {
            "phase": self.phase.name,
            "day": self.day,
            "is_game_over": self.is_game_over,
            "winner": self.winner,
            "players": [
                {
                    "name": p.name,
                    "role": p.role.value,
                    "is_alive": p.is_alive,
                    "can_speak": p.can_speak,
                }
                for p in self.players.values()
            ],
            "living_count": len(self.living_players),
            "wolf_count": len(self.living_wolf_players),
            "good_count": len(self.living_good_players),
        }

    def get_player_info(self, player_name: str) -> Optional[Dict]:
        """获取玩家信息（带信息隔离）"""
        if player_name not in self.players:
            return None

        player = self.players[player_name]
        info = {
            "name": player.name,
            "role": player.role.value,
            "is_alive": player.is_alive,
        }

        # 根据角色添加私有信息
        if player.is_wolf:
            info["teammates"] = player.wolf_teammates
        elif player.role.value == "seer":
            info["checks"] = player.seer_checks
        elif player.role.value == "witch":
            info["heal_potion"] = player.heal_potion
            info["poison_potion"] = player.poison_potion

        return info

    def set_ai_decision_maker(
        self,
        role: str,
        func: Callable[[Player, 'GameEngine'], str]
    ):
        """设置 AI 决策函数"""
        if role == "werewolf":
            self._get_wolf_decision = lambda p: func(p, self)
        elif role == "seer":
            self._get_seer_decision = lambda p: func(p, self)
        elif role == "witch":
            self._get_witch_heal_decision = lambda p: func(p, self)
        elif role == "witch_poison":
            self._get_witch_poison_decision = lambda p: func(p, self)
        elif role == "speak":
            self._get_speech = lambda p: func(p, self)
        elif role == "vote":
            self._get_vote = lambda p: func(p, self)

    def save_log(self) -> str:
        """保存日志"""
        return self.logger.save()


# ============================================================
# 游戏运行器
# ============================================================

class GameRunner:
    """游戏运行器"""

    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.max_days = 30

    async def run(self) -> str:
        """运行完整游戏"""
        print(f"🐺 游戏开始！{len(self.engine.players)}名玩家")
        print(f"角色分配: {self.engine.role_distribution}")
        print("-" * 50)

        day = 0
        while not self.engine.is_game_over and day < self.max_days:
            day += 1

            # 夜晚阶段
            night_result = await self.engine.night_phase()
            self._print_night_result(night_result)

            # 检查游戏结束
            if self.engine.is_game_over:
                break

            # 白天阶段
            executed = await self.engine.day_phase()
            self._print_day_result(executed)

        # 打印结果
        winner = self.engine.get_winner()
        if winner == "good":
            print("🏆 好人胜利！狼人全部被放逐。")
        elif winner == "wolf":
            print("🐺 狼人胜利！")
        else:
            print("⚖️ 游戏结束，无结果。")

        log_path = self.engine.save_log()
        print(f"\n📝 日志已保存: {log_path}")

        return winner or "draw"

    def _print_night_result(self, result: NightActions):
        """打印夜晚结果"""
        print(f"\n🌙 第 {self.engine.day} 夜 结束")
        if result.dead_players:
            for player, cause in result.death_causes.items():
                print(f"  ☠️ {player} 死亡 ({cause.value})")
        else:
            print("  😴 今晚无人死亡")

    def _print_day_result(self, executed: str):
        """打印白天结果"""
        print(f"\n☀️ 第 {self.engine.day} 天 结束")
        if executed:
            print(f"  ⚔️ {executed} 被投票处决")
        else:
            print("  🗳️ 平票，无人出局")


# ============================================================
# 模拟 AI 实现（用于测试）
# ============================================================

class MockAI:
    """模拟 AI - 用于测试"""

    @staticmethod
    async def wolf_decision(wolf: Player, engine: GameEngine) -> str:
        """狼人决策：随机杀好人"""
        goods = [p.name for p in engine.living_good_players]
        return random.choice(goods) if goods else None

    @staticmethod
    async def seer_decision(seer: Player, engine: GameEngine) -> tuple:
        """预言家决策：随机验人"""
        living = [p.name for p in engine.living_players if p.name != seer.name and not p.role.value == "seer"]
        target = random.choice(living) if living else None
        if target:
            is_wolf = target in [p.name for p in engine.living_wolf_players]
            return target, is_wolf
        return None, None

    @staticmethod
    async def witch_heal_decision(witch: Player, engine: GameEngine) -> Optional[str]:
        """女巫决策：总是救人"""
        if engine.night_actions.wolf_kill_target and witch.heal_potion > 0:
            return engine.night_actions.wolf_kill_target
        return None

    @staticmethod
    async def witch_poison_decision(witch: Player, engine: GameEngine) -> Optional[str]:
        """女巫决策：不毒人"""
        return None

    @staticmethod
    async def speech(player: Player, engine: GameEngine) -> str:
        """发言"""
        role_descriptions = {
            "werewolf": "我觉得场上好人居多，需要仔细分析",
            "villager": "我是村民，大家要相信我",
            "seer": "我查验了一些人，情况不太明朗",
            "witch": "我观察局势中，药还没用",
        }
        return role_descriptions.get(player.role.value, "发言中...")

    @staticmethod
    async def vote(player: Player, engine: GameEngine) -> str:
        """投票：随机投"""
        living = [p.name for p in engine.speaking_players if p.name != player.name]
        return random.choice(living) if living else None


# ============================================================
# 主入口
# ============================================================

async def main():
    """主入口"""
    # 玩家名称（6人局）
    players = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

    # 创建游戏引擎
    engine = GameEngine(
        player_names=players,
        role_distribution={
            "werewolf": 2,
            "villager": 2,
            "seer": 1,
            "witch": 1,
        },
        callback=lambda e, d: print(f"[EVENT] {e}: {d}"),
    )

    # 设置 AI
    engine.set_ai_decision_maker("werewolf", MockAI.wolf_decision)
    engine.set_ai_decision_maker("seer", MockAI.seer_decision)
    engine.set_ai_decision_maker("witch", MockAI.witch_heal_decision)
    engine.set_ai_decision_maker("speak", MockAI.speech)
    engine.set_ai_decision_maker("vote", MockAI.vote)

    # 运行游戏
    runner = GameRunner(engine)
    winner = await runner.run()

    # 打印最终状态
    print("\n📊 游戏状态:")
    state = engine.get_state()
    for p in state["players"]:
        status = "存活" if p["is_alive"] else "死亡"
        print(f"  {p['name']}: {p['role']} - {status}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())