# API 参考文档

## 导入方式

```python
# Agent 模块
from src.agents import (
    BaseAgent,
    GeneralAgent,
    WerewolfAgent,
    SeerAgent,
    WitchAgent,
    VillagerAgent,
    RoleType,
    ActionResult,
    ActionType,
    Memory,
    GameContext,
    create_agent,
    create_all_agents,
    create_general_agent,
    create_team,
)

# 引擎模块
from src.engine import (
    GameEngine,
    GameRunner,
    GameLogger,
    Phase,
    ActionType,
    CauseOfDeath,
    NightActions,
    Player,
    DeathRecord,
    MockAI,
)

# 角色模块
from src.roles import (
    Role,
    RoleTeam,
    RoleInfo,
    GameState,
    PlayerMemory,
    WinCondition,
    ObjectiveChecker,
    RoleStrategy,
    DEFAULT_6P_ROLES,
    create_role_info,
    get_role_team,
    get_winner_message,
)
```

## Agent 模块

### BaseAgent

```python
class BaseAgent(ABC):
    def __init__(self, name: str, role: str, llm_client=None)
    def set_context(self, context: GameContext)
    async def night_action(self) -> ActionResult
    async def speak(self) -> ActionResult
    async def vote(self) -> ActionResult
    def remember_speech(self, content: str, day: int)
    def remember_vote(self, target: str, day: int, reason: str = "")
    def remember_night_action(self, action: str, target: str, day: int)
    def update_suspicion(self, player: str, is_suspicious: bool, reason: str = "")
    async def think(self, prompt: str, system_prompt: str = None) -> str
    def get_system_prompt(self) -> str
    def get_decision_context(self) -> str
    @property
    def memory(self) -> Memory
    @property
    def context(self) -> Optional[GameContext]
```

### GeneralAgent

```python
class GeneralAgent(BaseAgent):
    def __init__(self, name: str, role: RoleType, llm_client=None, log_dir="runs/logs")
    def switch_role(self, new_role: RoleType)
    def set_behavior_mode(self, mode: str)  # "normal", "aggressive", "defensive", "conservative"
    def add_strategy(self, strategy: Strategy)
    def remove_strategy(self, strategy_name: str)
    def set_custom_prompt(self, action: str, prompt: str)
    def set_action_hook(self, action: str, hook: Callable)
    def get_state(self) -> Dict
    @property
    def role_type(self) -> RoleType
    @property
    def objective(self) -> Optional[Objective]
    @property
    def strategies(self) -> List[Strategy]
```

### RoleType

```python
class RoleType(Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    GUARD = "guard"
```

### ActionResult

```python
@dataclass
class ActionResult:
    action: ActionType
    target: Optional[str] = None
    content: Optional[str] = None
    reasoning: str = ""
    confidence: float = 0.5
```

### ActionType

```python
class ActionType(Enum):
    WAIT = "wait"
    KILL = "kill"
    CHECK = "check"
    HEAL = "heal"
    POISON = "poison"
    SPEAK = "speak"
    VOTE = "vote"
```

### Memory

```python
@dataclass
class Memory:
    player_name: str
    role: str
    private_info: Dict[str, Any]
    public_info: Dict[str, Any]
    speech_history: List[Dict]
    vote_history: List[Dict]
    night_action_history: List[Dict]
    suspicions: Dict[str, bool]
    def add_speech(self, content: str, day: int)
    def add_vote(self, target: str, day: int, reason: str = "")
    def add_night_action(self, action: str, target: str, day: int)
    def update_suspicion(self, player: str, is_suspicious: bool, reason: str = "")
    def get_private_data(self) -> Dict
```

### GameContext

```python
class GameContext:
    def __init__(self, player_name: str, living_players: List[str])
    def set_private_data(self, data: Dict)
    def set_public_data(self, data: Dict)
    @property
    def my_name(self) -> str
    @property
    def alive_players(self) -> List[str]
    def i_am(self) -> str
    def my_teammates(self) -> List[str]
    def my_checks(self) -> Dict[str, bool]
    def my_potions(self) -> Dict[str, int]
    def other_players(self) -> List[str]
    def get_others_speech(self, player_name: str) -> Optional[str]
    def get_others_vote(self, player_name: str) -> Optional[str]
    def get_dead_players(self) -> List[str]
    def get_recent_events(self) -> List[Dict]
```

### 工厂函数

```python
def create_agent(name: str, role: str, llm_client=None) -> BaseAgent
def create_all_agents(player_names, role_mapping, llm_client=None) -> Dict[str, BaseAgent]
def create_general_agent(name: str, role: RoleType, llm_client=None, behavior_mode="normal") -> GeneralAgent
def create_team(player_names: List[str], role_mapping: Dict[str, RoleType], llm_client=None) -> Dict[str, GeneralAgent]
```

## 引擎模块

### GameEngine

```python
class GameEngine:
    def __init__(
        self,
        player_names: List[str],
        role_distribution: Optional[Dict] = None,
        log_dir: str = "runs/logs",
        callback: Optional[Callable] = None,
    )

    # 属性
    @property
    def players(self) -> Dict[str, Player]
    @property
    def living_players(self) -> List[Player]
    @property
    def living_good_players(self) -> List[Player]
    @property
    def living_wolf_players(self) -> List[Player]
    @property
    def speaking_players(self) -> List[Player]
    @property
    def phase(self) -> Phase
    @property
    def day(self) -> int
    @property
    def is_game_over(self) -> bool
    @property
    def winner(self) -> Optional[str]

    # 方法
    async def night_phase(self) -> NightActions
    async def day_phase(self) -> str
    async def run(self) -> str
    def _check_win_condition(self) -> bool
    def _end_game(self, winner: str, reason: str)
    def get_state(self) -> Dict
    def get_player_info(self, player_name: str) -> Optional[Dict]
    def set_ai_decision_maker(self, role: str, func: Callable)
    def save_log(self) -> str
```

### GameRunner

```python
class GameRunner:
    def __init__(self, engine: GameEngine)
    async def run(self) -> str
```

### GameLogger

```python
class GameLogger:
    def __init__(self, log_dir: str = "runs/logs")
    def log_game_start(self, players: List[Dict])
    def log_phase_change(self, old_phase: Phase, new_phase: Phase, day: int)
    def log_night_start(self, day: int)
    def log_night_end(self, night_actions: NightActions)
    def log_day_start(self, day: int, deaths: List[str])
    def log_speech(self, player: str, content: str, day: int)
    def log_vote(self, voter: str, target: str, day: int)
    def log_execution(self, executed: str, vote_counts: Dict[str, int], day: int)
    def log_death(self, record: DeathRecord)
    def log_game_over(self, winner: str, reason: str)
    def save(self) -> str
    def get_summary(self) -> Dict
```

### Phase

```python
class Phase(Enum):
    WAITING = auto()
    NIGHT_START = auto()
    WOLF_KILL = auto()
    SEER_CHECK = auto()
    WITCH_ACTION = auto()
    DAY_START = auto()
    SPEECH = auto()
    VOTE = auto()
    EXECUTION = auto()
    GAME_OVER = auto()
```

### NightActions

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
    witch_heal_decided: bool = False
    witch_poison_decided: bool = False
    dead_players: List[str] = field(default_factory=list)
    death_causes: Dict[str, CauseOfDeath] = field(default_factory=dict)
```

### CauseOfDeath

```python
class CauseOfDeath(Enum):
    WOLF_KILL = "wolf_kill"
    VOTE = "vote"
    WITCH_POISON = "witch_poison"
    HUNTER_SHOOT = "hunter_shoot"
```

### Player

```python
@dataclass
class Player:
    id: str
    name: str
    role: Role
    is_alive: bool = True
    can_speak: bool = True
    vote_count: int = 0
    last_word: str = ""
    heal_potion: int = 1
    poison_potion: int = 1
    has_healed_tonight: bool = False
    has_poisoned_tonight: bool = False
    seer_checks: Dict[str, bool] = field(default_factory=dict)
    wolf_teammates: List[str] = field(default_factory=list)
    can_shoot: bool = True
    shoot_target: Optional[str] = None
    def new_night(self)
    def reset_vote(self)
    def speak(self, content: str)
    @property
    def is_wolf(self) -> bool
    @property
    def is_good(self) -> bool
```

### DeathRecord

```python
@dataclass
class DeathRecord:
    player: str
    cause: CauseOfDeath
    day: int
    phase: Phase
    killer: Optional[str] = None
```

## 角色模块

### Role

```python
class Role(Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"
```

### RoleTeam

```python
class RoleTeam(Enum):
    WOLF = "wolf"
    GOOD = "good"
```

### WinCondition

```python
class WinCondition(Enum):
    WOLF_WIN = "wolf_win"
    GOOD_WIN = "good_win"
    NO_WIN = "no_win"
```

### ObjectiveChecker

```python
class ObjectiveChecker:
    @staticmethod
    def check_win_condition(state: GameState) -> WinCondition
    @staticmethod
    def get_game_progress(state: GameState) -> Dict
    @staticmethod
    def is_game_over(state: GameState) -> bool
```

### RoleStrategy

```python
class RoleStrategy:
    @staticmethod
    def get_seer_night_strategy(state: GameState, seer_name: str) -> Dict
    @staticmethod
    def get_witch_night_strategy(state: GameState, witch_name: str) -> Dict
    @staticmethod
    def get_werewolf_night_strategy(state: GameState, wolf_name: str) -> Dict
    @staticmethod
    def get_villager_day_strategy(state: GameState, villager_name: str) -> Dict
```

### 工厂函数

```python
def create_role_info(role: Role) -> RoleInfo
def get_role_team(role: Role) -> RoleTeam
def get_winner_message(winner: str) -> str
```

## 常量

```python
# 默认6人局配置
DEFAULT_6P_ROLES = {
    Role.WEREWOLF: 2,
    Role.VILLAGER: 2,
    Role.SEER: 1,
    Role.WITCH: 1,
}
```