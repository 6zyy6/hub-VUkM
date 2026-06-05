# 角色系统文档

## 概述

角色系统定义了狼人杀游戏中的各个角色，包含：
- 角色配置
- 信息隔离规则
- 胜利条件
- 角色目标

## 文件结构

```
roles/
├── __init__.py
├── role_def.py    # 角色定义
└── objectives.py  # 角色目标
```

## 角色配置

### 6人局配置

```python
from src.roles import DEFAULT_6P_ROLES

# 默认6人局
DEFAULT_6P_ROLES = {
    Role.WEREWOLF: 2,   # 狼人 x2
    Role.VILLAGER: 2,   # 村民 x2
    Role.SEER: 1,       # 预言家 x1
    Role.WITCH: 1,     # 女巫 x1
}
```

### Role 枚举

```python
class Role(Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"
```

### RoleTeam 枚举

```python
class RoleTeam(Enum):
    WOLF = "wolf"    # 狼人阵营
    GOOD = "good"    # 好人阵营
```

## 信息隔离

每个角色只能访问自己的私有信息，不能访问其他角色的私有信息。

### 信息访问规则

| 角色 | 可访问的私有信息 | 隐藏的信息 |
|------|-----------------|-----------|
| 狼人 | 队友列表 | 其他狼人的杀人目标 |
| 预言家 | 查验记录 | 其他人的查验结果 |
| 女巫 | 用药状态 | 其他女巫的决策 |
| 村民 | 无 | 所有角色的私有信息 |

### PlayerMemory

玩家记忆类实现信息隔离。

```python
@dataclass
class PlayerMemory:
    player_name: str
    role: Role

    # 私有信息（只有自己能访问）
    _private_seer_checks: Dict[str, bool] = {}   # 预言家查验
    _private_witch_potions: Dict[str, int] = {}  # 女巫用药
    _private_wolf_teammates: List[str] = []      # 狼人队友
```

### GameState

游戏状态类。

```python
@dataclass
class GameState:
    players: List[str]                    # 所有玩家
    player_roles: Dict[str, Role]        # 玩家角色映射
    player_memory: Dict[str, PlayerMemory]  # 玩家记忆

    living_players: List[str]             # 存活玩家
    day_number: int                       # 当前天数
    current_phase: str                    # 当前阶段

    # 夜晚行动
    wolf_kill_target: Optional[str]       # 狼人要杀的人
    seer_check_target: Optional[str]      # 预言家要查的人
```

## 角色详解

### 狼人 Werewolf

**阵营**：狼人

**能力**：每晚可以杀害一名玩家

**信息**：知道所有狼人队友的身份

**胜利条件**：狼人数量 >= 好人数量

**策略**：
- 隐藏狼人身份
- 引导舆论
- 优先击杀关键角色（预言家、女巫）

### 村民 Villager

**阵营**：好人

**能力**：无特殊能力

**信息**：无私有信息，只能通过发言分析

**胜利条件**：所有狼人被放逐

**策略**：
- 分析发言
- 找出狼人
- 投票处决

### 预言家 Seer

**阵营**：好人

**能力**：每晚可以查验一名玩家的身份

**信息**：知道自己的查验结果

**胜利条件**：所有狼人被放逐

**策略**：
- 每晚查验可疑玩家
- 根据情况决定是否跳身份
- 引导好人投票

### 女巫 Witch

**阵营**：好人

**能力**：
- 解药：救狼人杀的人
- 毒药：毒死一名玩家

每瓶药只能用一次。

**信息**：知道自己的用药状态

**胜利条件**：所有狼人被放逐

**策略**：
- 解药优先救关键角色
- 注意狼人自刀骗药
- 合理规划用药

## 胜利条件

### 好人胜利

条件：所有狼人被放逐（狼人数量 = 0）

### 狼人胜利

条件：狼人数量 >= 好人数量

## 角色目标

### Objective 类

```python
@dataclass
class RoleObjective:
    role: Role
    team: RoleTeam
    win_condition: str
    hints: List[str]
```

### 各角色目标

```python
ROLE_OBJECTIVES = {
    Role.WEREWOLF: RoleObjective(
        role=Role.WEREWOLF,
        team=RoleTeam.WOLF,
        win_condition="狼人数量 >= 好人数量",
        hints=["隐藏身份", "引导舆论", "优先杀预言家"]
    ),
    Role.VILLAGER: RoleObjective(
        role=Role.VILLAGER,
        team=RoleTeam.GOOD,
        win_condition="所有狼人被放逐",
        hints=["分析发言", "找狼人", "不要盲目跟风"]
    ),
    Role.SEER: RoleObjective(
        role=Role.SEER,
        team=RoleTeam.GOOD,
        win_condition="所有狼人被放逐",
        hints=["优先查可疑玩家", "根据情况跳身份", "保护自己"]
    ),
    Role.WITCH: RoleObjective(
        role=Role.WITCH,
        team=RoleTeam.GOOD,
        win_condition="所有狼人被放逐",
        hints=["解药救关键角色", "毒药灭狼人", "注意自刀骗药"]
    ),
}
```

## ObjectiveChecker

目标检查器。

```python
from src.roles import ObjectiveChecker

# 检查胜负
result = ObjectiveChecker.check_win_condition(game_state)
# -> WinCondition.GOOD_WIN / WOLF_WIN / NO_WIN

# 获取游戏进度
progress = ObjectiveChecker.get_game_progress(game_state)
# -> {"total_players": 6, "living_count": 4, "wolf_count": 1, ...}

# 检查游戏是否结束
is_over = ObjectiveChecker.is_game_over(game_state)
```

## WinCondition 枚举

```python
class WinCondition(Enum):
    WOLF_WIN = "wolf_win"
    GOOD_WIN = "good_win"
    NO_WIN = "no_win"
```

## RoleStrategy

角色策略生成器。

```python
from src.roles import RoleStrategy, Role

# 预言家夜晚策略
strategy = RoleStrategy.get_seer_night_strategy(state, seer_name)
# -> {"candidates": [...], "already_checked": {...}, "strategy": "..."}

# 女巫夜晚策略
strategy = RoleStrategy.get_witch_night_strategy(state, witch_name)
# -> {"heal_remaining": 1, "poison_remaining": 1, ...}
```

## 工厂函数

```python
from src.roles import create_role_info, get_role_team, get_winner_message

# 获取角色信息
role_info = create_role_info(Role.SEER)

# 获取角色阵营
team = get_role_team(Role.WEREWOLF)
# -> RoleTeam.WOLF

# 获取胜利消息
msg = get_winner_message("wolf")
# -> "狼人胜利！狼人消灭了所有好人。"
```