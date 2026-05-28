# 对局引擎文档

## 概述

`game_engine.py` 实现了完整的狼人杀对局引擎，包含：
- 状态机管理
- 夜晚/白天流程
- 死亡判定
- 胜负裁决
- 结构化日志

## 核心类

### GameEngine

主游戏引擎类。

```python
from src.engine import GameEngine

engine = GameEngine(
    player_names=["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
    role_distribution={
        "werewolf": 2,
        "seer": 1,
        "witch": 1,
        "villager": 2,
    },
    log_dir="runs/logs",
)
```

#### 主要方法

```python
# 执行夜晚阶段
night_result = await engine.night_phase()

# 执行白天阶段
executed = await engine.day_phase()

# 检查胜负
winner = engine.check_win_condition()

# 获取状态
state = engine.get_state()

# 设置 AI 决策
engine.set_ai_decision_maker("werewolf", my_wolf_decision_func)
```

### GameRunner

游戏运行器。

```python
from src.engine import GameRunner

runner = GameRunner(engine)
winner = await runner.run()
```

## 游戏阶段

### Phase 枚举

```python
class Phase(Enum):
    WAITING = auto()          # 等待开始
    NIGHT_START = auto()      # 夜晚开始
    WOLF_KILL = auto()         # 狼人杀人
    SEER_CHECK = auto()        # 预言家验人
    WITCH_ACTION = auto()      # 女巫用药
    DAY_START = auto()         # 白天开始
    SPEECH = auto()            # 发言
    VOTE = auto()              # 投票
    EXECUTION = auto()          # 处决
    GAME_OVER = auto()          # 游戏结束
```

## 状态机流程

```
WAITING
  ↓
NIGHT_START → WOLF_KILL → SEER_CHECK → WITCH_ACTION
  ↓                                           ↓
DAY_START ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←
  ↓
SPEECH → VOTE → EXECUTION
  ↓
[循环或 GAME_OVER]
```

## 夜晚流程

```python
async def night_phase(self):
    self.day += 1
    self._set_phase(Phase.NIGHT_START)

    # 重置女巫状态
    for p in self.players.values():
        p.new_night()

    # 阶段 1: 狼人杀人
    await self._wolf_kill_phase()

    # 阶段 2: 预言家验人
    await self._seer_check_phase()

    # 阶段 3: 女巫用药
    await self._witch_action_phase()

    # 结算死亡
    self._resolve_night_deaths()

    return self.night_actions
```

### 狼人杀人

```python
async def _wolf_kill_phase(self):
    wolves = self.living_wolf_players
    for wolf in wolves:
        if wolf.can_speak:
            target = await self._get_wolf_decision(wolf)
            if target:
                self.night_actions.wolf_kill_target = target
```

### 预言家验人

```python
async def _seer_check_phase(self):
    seers = [p for p in self.living_players if p.role.value == "seer"]
    if seers:
        seer = seers[0]
        target, is_wolf = await self._get_seer_decision(seer)
        if target:
            self.night_actions.seer_check_target = target
            self.night_actions.seer_check_result = is_wolf
```

### 女巫用药

```python
async def _witch_action_phase(self):
    witches = [p for p in self.living_players if p.role.value == "witch"]
    if witches:
        witch = witches[0]
        # 救人
        if self.night_actions.wolf_kill_target and witch.heal_potion > 0:
            heal = await self._get_witch_heal_decision(witch)
            if heal:
                self.night_actions.witch_heal_target = heal
        # 毒人
        poison = await self._get_witch_poison_decision(witch)
        if poison:
            self.night_actions.witch_poison_target = poison
```

## 白天流程

```python
async def day_phase(self) -> str:
    self._set_phase(Phase.DAY_START)

    # 发言阶段
    await self._speech_phase()

    # 投票阶段
    executed = await self._vote_phase()

    return executed
```

## 数据结构

### NightActions

夜晚行动结果。

```python
@dataclass
class NightActions:
    wolf_kill_target: Optional[str] = None
    wolf_kill_decided: bool = False

    seer_check_target: Optional[str] = None
    seer_check_result: Optional[bool] = None
    seer_check_decided: bool = False

    witch_heal_target: Optional[str] = None
    witch_poison_target: Optional[str] = None

    dead_players: List[str] = field(default_factory=list)
    death_causes: Dict[str, CauseOfDeath] = field(default_factory=dict)
```

### CauseOfDeath

死亡原因枚举。

```python
class CauseOfDeath(Enum):
    WOLF_KILL = "wolf_kill"       # 狼人杀害
    VOTE = "vote"                  # 投票处决
    WITCH_POISON = "witch_poison"  # 女巫毒杀
    HUNTER_SHOOT = "hunter_shoot"  # 猎人开枪
```

## AI 决策接口

通过 `set_ai_decision_maker` 设置各角色的 AI 决策函数。

```python
async def my_wolf_decision(player, engine):
    """自定义狼人决策"""
    return "Bob"  # 返回要杀的人

async def my_seer_decision(player, engine):
    """自定义预言家决策"""
    return ("Bob", True)  # 返回 (目标, 是否狼人)

async def my_speech(player, engine):
    """自定义发言"""
    return "我觉得场上局势很复杂..."

engine.set_ai_decision_maker("werewolf", my_wolf_decision)
engine.set_ai_decision_maker("seer", my_seer_decision)
engine.set_ai_decision_maker("speak", my_speech)
```

## 日志系统

### GameLogger

结构化日志记录器。

```python
from src.engine import GameLogger

logger = GameLogger("runs/logs")
logger.log_game_start(players)
logger.log_phase_change(old_phase, new_phase, day)
logger.log_night_start(day)
logger.log_night_end(night_actions)
logger.log_speech(player, content, day)
logger.log_vote(voter, target, day)
logger.log_execution(executed, vote_counts, day)
logger.log_death(DeathRecord(...))
logger.log_game_over(winner, reason)

# 保存日志
log_path = logger.save()  # -> runs/logs/game_xxx.json
```

### 日志格式

```json
{
  "game_id": "abc123",
  "events": [
    {"type": "game_start", "timestamp": "...", "players": [...]},
    {"type": "phase", "timestamp": "...", "phase": "night"},
    {"type": "speech", "player": "Alice", "content": "..."},
    {"type": "vote", "voter": "Bob", "target": "Charlie"},
    {"type": "death", "player": "Charlie", "cause": "vote"},
    {"type": "game_over", "winner": "wolf"}
  ]
}
```

## 胜负判定

```python
def _check_win_condition(self) -> bool:
    living_wolves = len(self.living_wolf_players)
    living_goods = len(self.living_good_players)

    if living_wolves == 0:
        self._end_game("good", "所有狼人被放逐")
        return True

    if living_wolves >= living_goods:
        self._end_game("wolf", "狼人数量已占优势")
        return True

    return False
```

## 属性

```python
# 存活玩家
engine.living_players       # List[Player]
engine.living_good_players  # List[Player]
engine.living_wolf_players  # List[Player]

# 阶段信息
engine.phase                # Phase
engine.day                 # int
engine.is_game_over        # bool
engine.winner              # Optional[str]
```