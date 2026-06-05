# Agent 模块文档

## 概述

Agent 模块实现了狼人杀游戏中各个角色的 AI 智能体，每个 Agent 具备：
- 自主决策能力
- 记忆系统
- 信息隔离
- 动态适配

## 文件结构

```
agents/
├── __init__.py           # 导出接口
├── base_agent.py          # 基类 BaseAgent
├── general_agent.py        # 通用智能体 GeneralAgent
├── werewolf.py            # 狼人 Agent
├── seer.py                # 预言家 Agent
├── witch.py               # 女巫 Agent
└── villager.py           # 村民 Agent
```

## BaseAgent

所有 Agent 的基类。

### 主要组件

#### Memory（记忆系统）
```python
@dataclass
class Memory:
    player_name: str       # 玩家名称
    role: str              # 角色
    private_info: Dict     # 私有信息
    public_info: Dict      # 公共信息
    speech_history: List   # 发言历史
    vote_history: List     # 投票历史
    night_action_history: List  # 夜晚行动历史
    suspicions: Dict       # 推理结论
```

#### GameContext（游戏上下文）
```python
class GameContext:
    player_name: str           # 当前玩家
    living_players: List        # 存活玩家

    # 私有信息访问（角色特定）
    my_name() -> str           # 我的名字
    my_teammates() -> List     # 狼人队友
    my_checks() -> Dict       # 预言家查验
    my_potions() -> Dict       # 女巫用药

    # 公共信息访问
    alive_players() -> List     # 存活玩家
    other_players() -> List     # 其他玩家
    get_dead_players() -> List # 死亡玩家
```

#### ActionResult（行动结果）
```python
@dataclass
class ActionResult:
    action: ActionType    # 行动类型
    target: Optional[str]  # 目标
    content: Optional[str] # 发言内容
    reasoning: str        # 推理过程
    confidence: float     # 置信度
```

### 抽象方法

```python
class BaseAgent(ABC):
    async def night_action(self) -> ActionResult:
        """夜晚行动"""
        pass

    async def speak(self) -> ActionResult:
        """白天发言"""
        pass

    async def vote(self) -> ActionResult:
        """投票"""
        pass
```

## GeneralAgent

通用智能体，一个 Agent 可以扮演任何角色。

### 特性

1. **动态适配**：运行时切换角色
2. **策略库**：预定义各角色策略
3. **行为模式**：支持激进/保守等模式
4. **可扩展**：支持自定义策略和钩子

### 使用示例

```python
from src.agents import GeneralAgent, RoleType

# 创建狼人
agent = GeneralAgent("Alice", RoleType.WEREWOLF)

# 执行夜晚行动
result = await agent.night_action()

# 切换角色
agent.switch_role(RoleType.SEER)

# 设置行为模式
agent.set_behavior_mode("aggressive")
```

### RoleType 枚举

```python
class RoleType(Enum):
    WEREWOLF = "werewolf"    # 狼人
    VILLAGER = "villager"    # 村民
    SEER = "seer"            # 预言家
    WITCH = "witch"          # 女巫
    HUNTER = "hunter"        # 猎人
    GUARD = "guard"          # 守卫
```

### 策略管理

```python
# 添加自定义策略
agent.add_strategy(Strategy(
    name="激进策略",
    description="...",
    priority=10,
    conditions=["day_phase"],
    actions=["vote_aggressive"],
    prompt_template="..."
))

# 设置自定义提示词
agent.set_custom_prompt("speak", "你是激进派...")

# 设置行动钩子
agent.set_action_hook("night_action", my_hook_function)
```

## 各角色 Agent

### WerewolfAgent（狼人）

**特性**：
- 知道队友身份
- 夜晚协作杀人
- 白天隐藏身份

**决策逻辑**：
```python
# 夜晚杀人
- 优先杀预言家、女巫
- 不杀狼人队友

# 白天发言
- 伪装好人发言
- 不暴露队友关系

# 投票
- 投给好人
```

### SeerAgent（预言家）

**特性**：
- 夜晚验人
- 持有私有查验记录
- 可以选择跳身份

**决策逻辑**：
```python
# 夜晚验人
- 优先查可疑玩家
- 避免重复查验

# 白天发言
- 根据情况决定是否跳身份
- 可以暗示性发言
```

### WitchAgent（女巫）

**特性**：
- 有解药和毒药
- 每瓶药用一次
- 信息隔离

**决策逻辑**：
```python
# 解药
- 优先救预言家、猎人
- 注意狼人自刀骗药

# 毒药
- 确认狼人后使用
- 不要浪费
```

### VillagerAgent（村民）

**特性**：
- 无特殊能力
- 纯分析推理

**决策逻辑**：
```python
# 分析发言
- 找逻辑矛盾
- 识别狼人伪装

# 投票
- 基于分析结果
```

## 工厂函数

```python
from src.agents import create_agent, create_all_agents, create_general_agent, create_team

# 创建单个 Agent
agent = create_agent("Alice", "werewolf", llm_client)

# 创建所有 Agent
role_mapping = {
    "Alice": "werewolf",
    "Bob": "seer",
    ...
}
agents = create_all_agents(names, role_mapping, llm_client)

# 使用 GeneralAgent
agent = create_general_agent("Alice", RoleType.WEREWOLF, behavior_mode="aggressive")

# 创建团队
team = create_team(names, role_mapping, llm_client)
```

## 信息隔离规则

| 角色 | 可见信息 | 不可见信息 |
|------|---------|-----------|
| 狼人 | 队友列表 | 其他狼人的具体目标 |
| 预言家 | 自己的查验记录 | 其他人的查验结果 |
| 女巫 | 自己用药状态 | 其他女巫的决策 |
| 村民 | 公共发言 | 任何私有信息 |

## 继承关系

```
BaseAgent (ABC)
├── WerewolfAgent
├── SeerAgent
├── WitchAgent
├── VillagerAgent
└── GeneralAgent (可扮演任意角色)
```