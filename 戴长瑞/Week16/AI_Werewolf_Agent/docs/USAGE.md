# 使用指南

## 基本用法

### 1. CLI 模式

```bash
# 默认运行（智能模式）
python main.py

# 输出示例
============================================================
AI Werewolf - 全自动狼人杀多智能体对战系统
============================================================

角色分配:
  Alice: werewolf
  Bob: seer
  ...

第 1 天
  [夜晚] 狼人选择杀害: Charlie
  [白天] Charlie 被投票处决

游戏结束!
好人胜利
```

### 2. Web 界面模式

```bash
streamlit run ui/app.py

# 浏览器打开 http://localhost:8501
```

Web 界面功能：
- 🎮 **开始新游戏** — 初始化一局新的狼人杀
- ▶️ **继续下一步** — 单步执行（夜晚 → 白天发言 → 投票）
- ⚡ **自动运行** — 快速完成整局游戏
- 💬 **发言记录** — 实时显示所有存活玩家的发言内容（按天分组，带角色标识）
- 🗳️ **投票统计** — 展示投票分布和详细投票记录
- 🤖 **LLM 配置** — 侧边栏选择 LLM 提供商（mock / Claude / OpenAI）

### 3. 集成到代码

```python
import asyncio
from src.engine import GameEngine
from src.agents import create_all_agents

# 创建游戏
player_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
role_mapping = {
    "Alice": "werewolf",
    "Bob": "werewolf",
    "Charlie": "seer",
    "Diana": "witch",
    "Eve": "villager",
    "Frank": "villager",
}

# 创建 Agent
agents = create_all_agents(player_names, role_mapping, llm_client)

# 创建引擎
engine = GameEngine(player_names=player_names, role_distribution={...})

# 运行游戏
async def run():
    while not engine.is_game_over:
        await engine.night_phase()
        await engine.day_phase()

asyncio.run(run())
```

## 配置文件

项目使用 `config.toml` 管理 LLM 和游戏配置：

```toml
[llm]
provider = "mock"                    # mock | claude | openai
api_key = ""                         # Claude/OpenAI API Key
model = "claude-sonnet-4-20250514"   # 模型名称
max_tokens = 512
temperature = 0.7

[game]
mode = "smart"                       # smart | simple
max_days = 15
```

### LLM 提供商切换

**Mock 模式**（默认，无需 API Key）：
```toml
[llm]
provider = "mock"
```

**Claude 模式**（需安装 `anthropic` 包）：
```bash
pip install anthropic>=0.40.0
```
```toml
[llm]
provider = "claude"
api_key = "sk-ant-..."        # 或设置 ANTHROPIC_API_KEY 环境变量
model = "claude-sonnet-4-20250514"
```

**OpenAI 模式**：
```toml
[llm]
provider = "openai"
api_key = "sk-..."             # 或设置 OPENAI_API_KEY 环境变量
model = "gpt-4o"
```

CLI 模式下，config.toml 的 `[llm]` 配置控制使用哪个提供商。Web 界面可以在侧边栏中实时切换。

## 完整发言系统

游戏引擎实现了一整套发言机制，让 AI Agent 能够进行有意义的交流：

### 发言流程

```
白天开始
  ↓
玩家1 发言（参考公共信息 + 自己身份）
  ↓   ← 发言内容存入 `_day_speeches`
玩家2 发言（参考玩家1的发言 + 公共信息）
  ↓   ← 发言内容存入 `_day_speeches`
玩家3 发言（参考玩家1、2的发言 + 公共信息）
  ↓
...   ← 依次类推
  ↓
投票阶段（所有玩家参考全部发言记录后投票）
```

### 信息共享机制

每个 Agent 通过 `GameContext` 访问其他玩家的发言：

```python
# 获取所有玩家的发言
all_speeches = context.get_all_speeches()
# -> {"Alice": "我觉得...", "Bob": "我同意..."}

# 获取指定玩家的发言
alice_speech = context.get_others_speech("Alice")
```

### 各角色发言策略

| 角色 | 发言目标 | 策略 |
|------|---------|------|
| 狼人 | 隐藏身份，引导投票 | 参考前面玩家的发言，假装好人分析，把怀疑引向好人的方向 |
| 预言家 | 传递查验信息 | 决定是否跳身份，报查验结果，引导好人投票方向 |
| 女巫 | 隐藏身份，分析局势 | 参考夜间信息（谁被刀了），结合发言分析找出可疑的人 |
| 村民 | 找出狼人 | 分析每个发言的逻辑矛盾、立场变化、可疑关注点 |

### 投票决策

投票基于完整的发言分析：

```python
# 在 vote() 中，Agent 会分析所有发言：
# 1. 谁发言逻辑矛盾
# 2. 谁在转移焦点
# 3. 谁在附和别人没有主见
# 4. 谁的分析最像好人/狼人
# 5. 结合自己的私有信息（查验结果、夜间信息等）
```

## 自定义 AI 决策

### 设置狼人决策

```python
async def my_wolf_decision(player, engine):
    """自定义狼人杀人逻辑"""
    # 返回要杀的人
    return "预言家"  # 或其他逻辑

engine.set_ai_decision_maker("werewolf", my_wolf_decision)
```

### 设置预言家决策

```python
async def my_seer_decision(player, engine):
    """自定义预言家查验逻辑"""
    # 返回 (目标, 是否狼人)
    return ("Alice", True)  # 或其他逻辑

engine.set_ai_decision_maker("seer", my_seer_decision)
```

### 设置女巫决策

```python
async def my_witch_heal_decision(player, engine):
    """自定义女巫救人逻辑"""
    # 返回要救的人或 None
    return "Bob"

async def my_witch_poison_decision(player, engine):
    """自定义女巫毒人逻辑"""
    # 返回要毒的人或 None
    return "Charlie"

engine.set_ai_decision_maker("witch", my_witch_heal_decision)
```

### 设置发言

```python
async def my_speech(player, engine):
    """自定义发言逻辑"""
    return "我觉得场上局势很复杂，需要仔细分析。"

engine.set_ai_decision_maker("speak", my_speech)
```

### 设置投票

```python
async def my_vote(player, engine):
    """自定义投票逻辑"""
    return "Bob"

engine.set_ai_decision_maker("vote", my_vote)
```

## 使用 GeneralAgent

### 创建通用智能体

```python
from src.agents import GeneralAgent, RoleType, create_general_agent

# 方式1：直接创建
agent = GeneralAgent("Alice", RoleType.WEREWOLF)

# 方式2：使用工厂函数
agent = create_general_agent("Alice", RoleType.SEER, behavior_mode="aggressive")
```

### 动态切换角色

```python
# 创建狼人
agent = GeneralAgent("Alice", RoleType.WEREWOLF)
await agent.night_action()  # 狼人杀人

# 切换为预言家
agent.switch_role(RoleType.SEER)
await agent.night_action()  # 预言家验人
```

### 自定义策略

```python
from src.agents import Strategy

# 添加自定义策略
agent.add_strategy(Strategy(
    name="激进策略",
    description="优先攻击可疑玩家",
    priority=10,
    conditions=["day_phase", "vote_phase"],
    actions=["vote_aggressive"],
    prompt_template="你是激进派..."
))

# 设置行为模式
agent.set_behavior_mode("aggressive")  # 激进
agent.set_behavior_mode("defensive")   # 保守
agent.set_behavior_mode("conservative")  # 谨慎
```

### 角色特定数据

```python
# 狼人：设置队友
agent.role_data["wolf_teammates"] = ["Bob", "Charlie"]

# 预言家：查看查验记录
checks = agent.role_data["checked_players"]
# -> {"Alice": True, "Bob": False}

# 女巫：查看用药状态
potions = agent.role_data["potions"]
# -> {"heal": 1, "poison": 1}
```

## 创建团队

```python
from src.agents import create_team, RoleType

names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
role_mapping = {
    "Alice": RoleType.WEREWOLF,
    "Bob": RoleType.WEREWOLF,
    "Charlie": RoleType.SEER,
    "Diana": RoleType.WITCH,
    "Eve": RoleType.VILLAGER,
    "Frank": RoleType.VILLAGER,
}

team = create_team(names, role_mapping, llm_client)

# 使用团队
for name, agent in team.items():
    result = await agent.speak()
    print(f"{name}: {result.content}")
```

## 查看游戏状态

```python
# 获取引擎状态
state = engine.get_state()
# {
#   "phase": "night",
#   "day": 2,
#   "is_game_over": False,
#   "players": [...],
#   "living_count": 4,
#   "wolf_count": 2,
#   "good_count": 2
# }

# 获取玩家信息
player_info = engine.get_player_info("Alice")
# {
#   "name": "Alice",
#   "role": "werewolf",
#   "is_alive": True,
#   "teammates": ["Bob"]  # 狼人专属
# }

# 获取 Agent 状态
agent_state = agent.get_state()
# {
#   "name": "Alice",
#   "role": "werewolf",
#   "behavior_mode": "normal",
#   "strategies": [...],
#   "role_data": {...},
#   "memory": {...}
# }
```

## 日志使用

```python
from src.engine import GameLogger

logger = GameLogger("runs/logs")

# 记录游戏事件
logger.log_game_start([...])
logger.log_phase_change(Phase.NIGHT_START, Phase.WOLF_KILL, 1)
logger.log_speech("Alice", "我觉得...", 1)
logger.log_vote("Bob", "Charlie", 1)
logger.log_death(DeathRecord(player="Charlie", cause=CauseOfDeath.VOTE, ...))
logger.log_game_over("good", "所有狼人被放逐")

# 保存日志
log_path = logger.save()
# -> "runs/logs/game_abc123.json"

# 加载日志
with open(log_path, "r") as f:
    data = json.load(f)
```

## 扩展游戏

### 添加新角色

```python
from src.agents import RoleType

# 在 GeneralAgent 中添加新角色支持
class HunterAgent(BaseAgent):
    async def night_action(self):
        # 猎人夜晚无行动
        return ActionResult(action=ActionType.WAIT)

    async def speak(self):
        # 猎人发言
        ...

    async def vote(self):
        # 猎人投票
        ...
```

### 自定义胜利条件

```python
# 修改 ObjectiveChecker
class CustomObjectiveChecker(ObjectiveChecker):
    @staticmethod
    def check_win_condition(state: GameState) -> WinCondition:
        # 自定义胜利条件
        if custom_condition:
            return WinCondition.CUSTOM_WIN
        return super().check_win_condition(state)
```

## 调试技巧

### 查看 Agent 决策

```python
# 启用调试输出
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看记忆
print(agent.memory.suspicions)
print(agent.memory.speech_history)
print(agent.memory.vote_history)
```

### 查看游戏流程

```python
# 设置回调
engine = GameEngine(
    ...,
    callback=lambda event_type, data: print(f"[{event_type}] {data}")
)
```

### 复盘游戏

```python
# 加载日志
with open("runs/logs/game_abc123.json") as f:
    log = json.load(f)

# 分析
for event in log["events"]:
    if event["type"] == "speech":
        print(f"{event['player']}: {event['content']}")
```

## 常见问题

### Q: 如何设置 LLM？

**方式一：配置文件（推荐）**
```toml
# config.toml
[llm]
provider = "claude"
api_key = "sk-ant-..."
model = "claude-sonnet-4-20250514"
```

**方式二：代码中直接创建**
```python
from src.llm import ClaudeLLM, OpenAILLM

# Claude
llm = ClaudeLLM(
    model="claude-sonnet-4-20250514",
    api_key="sk-ant-...",
)

# OpenAI
llm = OpenAILLM(
    model="gpt-4o",
    api_key="sk-...",
)

# 传递给 Agent
agents = create_all_agents(player_names, role_mapping, llm_client=llm)
```

**方式三：环境变量**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# 或
export OPENAI_API_KEY="sk-..."
```

### Q: 如何调整游戏速度？
```python
# 在 UI 中使用自动运行
# 或设置异步延迟
await asyncio.sleep(0.5)
```

### Q: 如何保存游戏记录？
```python
log_path = engine.save_log()
```