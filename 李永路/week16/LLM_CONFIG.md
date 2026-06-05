# LLM Agent 配置指南

## 概述

本项目支持将基于规则的 Agent 升级为基于大语言模型（LLM）的 Agent，使每个玩家能够使用 AI 进行更智能的推理、发言和决策。

## 支持的 LLM 提供商

### 1. Mock LLM（测试用）
- **无需 API Key**
- 用于快速测试和开发
- 响应简单但稳定

```bash
python main.py --use-llm --llm-provider mock
```

### 2. OpenAI GPT
- **需要 API Key**
- 模型：gpt-3.5-turbo, gpt-4 等
- 获取 API Key: https://platform.openai.com/

**安装依赖：**
```bash
pip install openai
```

**使用方法：**
```bash
# 方式1：命令行参数
python main.py --use-llm --llm-provider openai --api-key your-api-key-here

# 方式2：环境变量
export OPENAI_API_KEY="your-api-key-here"
python main.py --use-llm --llm-provider openai
```

**Python 代码：**
```python
from werewolf_agents.llm_client import OpenAIClient
from werewolf_agents.engine import WerewolfGameEngine

llm_client = OpenAIClient(api_key="your-api-key", model="gpt-3.5-turbo")
engine = WerewolfGameEngine(
    player_configs,
    use_llm=True,
    llm_client=llm_client
)
```

### 3. 通义千问（Qwen）
- **需要 API Key**
- 模型：qwen-plus, qwen-max 等
- 获取 API Key: https://dashscope.console.aliyun.com/

**安装依赖：**
```bash
pip install dashscope
```

**使用方法：**
```bash
# 方式1：命令行参数
python main.py --use-llm --llm-provider qwen --api-key your-api-key-here

# 方式2：环境变量
export DASHSCOPE_API_KEY="your-api-key-here"
python main.py --use-llm --llm-provider qwen
```

**Python 代码：**
```python
from werewolf_agents.llm_client import QwenClient
from werewolf_agents.engine import WerewolfGameEngine

llm_client = QwenClient(api_key="your-api-key", model="qwen-plus")
engine = WerewolfGameEngine(
    player_configs,
    use_llm=True,
    llm_client=llm_client
)
```

### 4. DeepSeek
- **需要 API Key**
- 模型：deepseek-chat, deepseek-coder 等
- 获取 API Key: https://platform.deepseek.com/

**安装依赖：**
```bash
pip install openai
```

**使用方法：**
```bash
# 方式1：命令行参数
python main.py --use-llm --llm-provider deepseek --api-key your-api-key-here

# 方式2：环境变量
export DEEPSEEK_API_KEY="your-api-key-here"
python main.py --use-llm --llm-provider deepseek
```

**Python 代码：**
```python
from werewolf_agents.llm_client import DeepSeekClient
from werewolf_agents.engine import WerewolfGameEngine

llm_client = DeepSeekClient(api_key="your-api-key", model="deepseek-chat")
engine = WerewolfGameEngine(
    player_configs,
    use_llm=True,
    llm_client=llm_client
)
```

## 各角色 LLM 提示词设计

### 狼人（Werewolf）
- **目标**：伪装成好人，夜间击杀神职
- **策略**：混淆视听，引导投票
- **输出格式**：玩家 ID 数字

### 村民（Villager）
- **目标**：通过推理找出狼人
- **策略**：分析发言逻辑，寻找矛盾
- **输出格式**：自然语言发言 + 投票 ID

### 预言家（Seer）
- **目标**：查验身份，建立信任
- **策略**：尽早跳身份报查验
- **输出格式**：查验结果 + 发言内容

### 女巫（Witch）
- **目标**：合理使用解药和毒药
- **策略**：首晚救人，谨慎用毒
- **输出格式**：救:ID, 毒:ID

### 猎人（Hunter）
- **目标**：威慑狼人，死亡带走敌人
- **策略**：适当强势发言
- **输出格式**：带走目标 ID

### 守卫（Guard）
- **目标**：保护关键玩家
- **策略**：避免连续守护同一人
- **输出格式**：守护目标 ID

## 性能优化建议

### 1. 温度设置（Temperature）
- **夜间行动**：0.5-0.7（更确定性）
- **白天发言**：0.7-0.9（更有创造性）
- **投票决策**：0.3-0.5（更理性）

### 2. Token 限制
- **夜间行动**：max_tokens=100
- **白天发言**：max_tokens=200
- **投票决策**：max_tokens=50

### 3. 成本控制
- 使用 Mock LLM 进行开发和测试
- 选择性价比高的模型（如 gpt-3.5-turbo）
- 限制最大回合数

## 常见问题

### Q1: LLM 响应格式错误怎么办？
A: 系统会自动解析响应，如果格式错误会返回 None 或使用默认行为。

### Q2: 如何调试 LLM Agent 的决策过程？
A: 查看日志文件中的 "LLM决策" 和 "LLM发言" 记录。

### Q3: 可以混合使用规则 Agent 和 LLM Agent 吗？
A: 当前版本所有玩家统一使用同类型 Agent。如需混合，需修改 engine.py。

### Q4: LLM Agent 比规则 Agent 强在哪里？
A: LLM Agent 能够：
- 理解复杂的游戏局势
- 生成自然的发言内容
- 进行逻辑推理和策略规划
- 适应不同的游戏风格

## 进阶方向

### ① 自演化 Agent
让 Agent 能够从历史对局中学习，不断优化自己的策略。

### ② 多轮对话
在白天讨论阶段引入多轮对话机制，让 Agent 之间进行辩论。

### ③ 情感模拟
为 Agent 添加情感状态，使其发言更具个性化和真实感。

### ④ 团队协作
狼人团队在夜间进行真正的协商讨论，而不是简单的投票。

## 示例运行

```bash
# 快速测试（Mock LLM）
python main.py --config quick_4 --use-llm --llm-provider mock --max-rounds 3

# 使用 OpenAI（需要 API Key）
python main.py --config simple_6 --use-llm --llm-provider openai --api-key sk-xxx

# 运行示例脚本
python examples_llm.py
```

## 技术支持

如有问题，请查看：
- 日志文件：`logs/werewolf_*.log`
- 游戏报告：`logs/report_*.json`
- 观测数据：`observations.json`
