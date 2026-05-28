# AI Werewolf Agent

基于 Python 的多智能体狼人杀博弈系统，6个 AI Agent 自主进行完整的狼人杀游戏。

## 快速开始

### CLI 模式
```bash
pip install -e .
python main.py           # 智能模拟模式
python main.py --smart   # 智能模拟模式
python main.py --simple  # 简单模拟模式
```

### Web 界面
```bash
streamlit run ui/app.py
# 浏览器打开 http://localhost:8501
```

## 配置

编辑项目根目录的 `config.toml` 文件：

```toml
[llm]
# LLM 提供商: mock | claude | openai
provider = "mock"

# Claude API 配置（provider 设为 claude 时生效）
api_key = "sk-ant-..."
model = "claude-sonnet-4-20250514"
max_tokens = 512
temperature = 0.7

# Claude Code 已安装时，直接使用 claude provider：
# 1. 配置 ANTHROPIC_API_KEY 环境变量
# 2. 或将 api_key 写入 config.toml
# 3. 设置 provider = "claude"

[game]
mode = "smart"    # smart | simple
max_days = 15
```

### LLM 提供商

| 提供商 | 配置文件 provider | 需要安装 | 说明 |
|--------|-----------------|---------|------|
| Mock (模拟) | `mock` | 无 | 内置规则引擎，无需 API |
| Claude | `claude` | `anthropic` | 调用 Claude API |
| OpenAI | `openai` | `openai` | 调用 OpenAI API |

安装 Claude 依赖：
```bash
pip install anthropic>=0.40.0
```

## 项目结构

```
AI_Werewolf_Agent/
├── main.py              # CLI 入口
├── config.toml          # 配置文件（LLM 提供商、API Key 等）
├── ui/app.py            # Streamlit Web 界面
├── src/
│   ├── agents/          # AI Agent 模块
│   │   ├── base_agent.py      # 基类 + 记忆系统 + 信息隔离
│   │   ├── general_agent.py   # 通用智能体 + 自进化策略库
│   │   ├── werewolf.py         # 狼人（伪装发言、误导投票）
│   │   ├── seer.py             # 预言家（查验 + 跳身份决策）
│   │   ├── witch.py            # 女巫（救/毒决策 + 隐藏身份）
│   │   └── villager.py         # 村民（发言分析 + 逻辑推理）
│   ├── engine/          # 对局引擎
│   │   └── game_engine.py      # 状态机 + 夜晚/白天流程 + 胜负判定
│   ├── llm/             # LLM 接口
│   │   ├── base.py             # LLM 基类
│   │   ├── claude_llm.py       # Claude API 实现
│   │   ├── openai_llm.py       # OpenAI API 实现
│   │   └── config.py           # 配置加载器
│   ├── roles/           # 角色定义
│   │   └── role_def.py         # 角色配置
│   └── prompts/         # 提示词模板
│       └── templates.py
├── tests/               # 测试
│   ├── test_agents.py
│   └── test_engine.py
└── runs/logs/           # 对局日志
```

## 完整发言系统

每个白天，所有存活的 Agent 按顺序轮流发言：

1. **发言顺序**：游戏引擎依次调用每个存活 Agent 的 `speak()` 方法
2. **信息共享**：发言过程中，已发言玩家的内容会通过 `GameContext.set_public_data()` 传递给后续说话者
3. **发言分析**：每个 Agent 在 `speak()` 时会参考前面玩家的发言内容，生成回应
4. **可信度判断**：Agent 内部通过 `suspicions` 字典追踪每个玩家的可信度
5. **投票依据**：`vote()` 方法基于所有发言的分析结果做出决策，不再随机投票

### 各角色发言策略

| 角色 | 发言策略 |
|------|---------|
| 狼人 | 伪装好人发言，参考前面玩家的内容，把怀疑引向好人的方向，不攻击队友 |
| 预言家 | 根据查验结果决定是否跳身份，引导好人投票方向 |
| 女巫 | 隐藏身份，分析发言，结合夜间信息进行推理 |
| 村民 | 分析每个玩家的发言逻辑和可信度，找出可疑的人 |

## 自进化系统

`general_agent.py` 中的 `StrategyLibrary` 模块实现了策略的自进化：
- 记录成功/失败的策略
- 根据对局结果调整策略权重
- 在后续对局中优先采用高胜率策略

## 游戏配置

**6人局角色**：2狼人 + 1预言家 + 1女巫 + 2村民

**胜利条件**：
- 好人胜利：狼人全灭
- 狼人胜利：狼人数量 >= 好人数量

## Agent 架构

### 信息隔离

每个 Agent 只能访问自己的私有信息：

| 角色 | 私有信息 |
|------|---------|
| 狼人 | 队友身份 |
| 预言家 | 查验记录 |
| 女巫 | 用药状态 |
| 村民 | 无 |

### 决策流程

```
夜晚流程: 狼人刀人 → 预言家验人 → 女巫用药 → 结算死亡
白天流程: 宣布死亡 → 顺序发言(含分析和回应) → 投票(基于发言分析) → 处决
```

## 运行日志

对局结束后日志保存在 `runs/logs/game_*.json`，包含完整游戏记录。
