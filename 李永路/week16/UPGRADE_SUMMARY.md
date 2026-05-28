# AI 狼人杀系统升级说明 - LLM Agent 版本

## 🎯 升级概述

已成功将基于规则的狼人杀 Agent 升级为基于大语言模型（LLM）的智能 Agent，使每个玩家能够使用 AI 进行更智能的推理、发言和决策。

## ✨ 核心改进

### 1. **LLM 客户端封装** (`llm_client.py`)
- ✅ 支持多种 LLM 提供商（OpenAI、Qwen、DeepSeek、Mock）
- ✅ 统一的接口设计，易于扩展
- ✅ 错误处理和降级机制

### 2. **智能 Agent 实现** (`llm_agents.py`)
- ✅ 为每个角色设计专属提示词模板
- ✅ 上下文感知的决策逻辑
- ✅ 结构化输出解析

### 3. **引擎增强** (`engine.py`)
- ✅ 支持规则 Agent 和 LLM Agent 切换
- ✅ 兼容原有游戏流程
- ✅ 特殊角色（女巫）的双动作处理

### 4. **配置管理** (`main.py`)
- ✅ 命令行参数支持 LLM 选项
- ✅ API Key 灵活配置（参数或环境变量）
- ✅ 友好的错误提示

## 📁 新增文件

```
Week16/
├── werewolf_agents/
│   ├── llm_client.py          # LLM 客户端封装 ⭐ 新增
│   └── llm_agents.py          # LLM Agent 实现 ⭐ 新增
├── examples_llm.py            # LLM 使用示例 ⭐ 新增
├── LLM_CONFIG.md              # LLM 配置指南 ⭐ 新增
├── UPGRADE_SUMMARY.md         # 本文件 ⭐ 新增
├── main.py                    # 已更新（支持 LLM 参数）
├── requirements.txt           # 已更新（添加可选依赖）
└── README.md                  # 保持不变
```

## 🚀 快速开始

### 方式1：命令行运行

```bash
# 使用 Mock LLM（无需 API Key，推荐首次测试）
python main.py --config quick_4 --use-llm --llm-provider mock

# 使用 OpenAI GPT
python main.py --config simple_6 --use-llm --llm-provider openai --api-key sk-xxx

# 使用通义千问
python main.py --config simple_6 --use-llm --llm-provider qwen --api-key your-key

# 使用 DeepSeek
python main.py --config simple_6 --use-llm --llm-provider deepseek --api-key your-key
```

### 方式2：Python 代码

```python
from werewolf_agents.engine import WerewolfGameEngine
from werewolf_agents.config import get_config
from werewolf_agents.llm_client import OpenAIClient

# 创建 LLM 客户端
llm_client = OpenAIClient(api_key="your-api-key", model="gpt-3.5-turbo")

# 创建游戏引擎（启用 LLM）
player_configs = get_config("simple_6")
engine = WerewolfGameEngine(
    player_configs,
    use_llm=True,
    llm_client=llm_client
)

# 运行游戏
engine.run_game(max_rounds=10)
```

### 方式3：运行示例脚本

```bash
python examples_llm.py
```

## 🎭 角色提示词设计

### 狼人（Werewolf）
```
系统提示：你是狼人阵营的一员，目标是消灭所有好人。
策略：夜间优先击杀神职，白天伪装成好人。
输出：玩家 ID 数字
```

### 预言家（Seer）
```
系统提示：你是预言家，每晚可以查验一名玩家的身份。
策略：尽早跳身份报查验，建立信任。
输出：查验目标 ID + 发言内容
```

### 女巫（Witch）
```
系统提示：你是女巫，拥有一瓶解药和一瓶毒药。
策略：首晚通常救人，毒药谨慎使用。
输出：救:ID, 毒:ID
```

### 村民（Villager）
```
系统提示：你是普通村民，需要通过推理找出狼人。
策略：仔细倾听发言，寻找矛盾点。
输出：自然语言发言 + 投票 ID
```

## 🔧 技术架构

### LLM 客户端层次结构

```
BaseLLMClient (抽象基类)
├── OpenAIClient        # OpenAI GPT
├── QwenClient          # 通义千问
├── DeepSeekClient      # DeepSeek
└── MockLLMClient       # 模拟客户端（测试用）
```

### Agent 工作流程

```
1. 构建上下文
   ├─ 当前游戏状态
   ├─ 存活玩家列表
   ├─ 已知信息
   └─ 最近发言记录

2. 生成提示词
   ├─ 系统提示词（角色专属）
   └─ 用户提示词（任务描述）

3. 调用 LLM
   ├─ 发送请求
   └─ 接收响应

4. 解析响应
   ├─ 提取关键信息
   └─ 验证格式
```

## 📊 性能对比

| 特性 | 规则 Agent | LLM Agent |
|------|-----------|-----------|
| 决策速度 | ⚡ 快 | 🐢 慢（依赖 API） |
| 发言质量 | 😐 固定模板 | 😊 自然流畅 |
| 推理能力 | 😐 简单逻辑 | 🧠 复杂推理 |
| 适应性 | 😐 固定策略 | 🎯 动态调整 |
| 成本 | 💰 免费 | 💰 API 费用 |
| 可调试性 | ✅ 容易 | ⚠️ 较难 |

## 🎓 学习要点

### 1. 提示词工程
- 角色定位清晰
- 任务描述明确
- 输出格式规范
- 策略建议具体

### 2. 上下文管理
- 信息隔离（每个 Agent 只能看到允许的信息）
- 历史记忆（保留最近的发言记录）
- 状态同步（实时更新游戏状态）

### 3. 错误处理
- API 调用失败降级
- 响应格式异常处理
- 超时和重试机制

## 🔮 进阶方向

### ① 自演化 Agent
```python
# 从历史对局中学习
agent.learn_from_game(game_report)
agent.update_strategy()
```

### ② 多轮对话
```python
# 白天讨论阶段的多轮交互
for round in discussion_rounds:
    speech = agent.day_speech(context)
    context.add_speech(speech)
```

### ③ 情感模拟
```python
# 为 Agent 添加情感状态
agent.emotion_state = {
    "trust": 0.7,
    "suspicion": 0.3,
    "confidence": 0.8
}
```

### ④ 团队协作
```python
# 狼人团队夜间协商
werewolf_team.discuss_kill_target()
target = werewolf_team.vote_kill()
```

## 📝 使用建议

### 开发阶段
1. 使用 Mock LLM 进行快速迭代
2. 测试各种边界情况
3. 验证提示词效果

### 部署阶段
1. 选择性价比高的模型（如 gpt-3.5-turbo）
2. 设置合理的 temperature 和 max_tokens
3. 监控 API 调用成本

### 优化阶段
1. 分析日志，优化提示词
2. A/B 测试不同策略
3. 收集对局数据进行复盘

## 🐛 常见问题

### Q1: LLM 响应不符合预期？
**A:** 检查提示词是否清晰，输出格式要求是否明确。

### Q2: API 调用失败？
**A:** 检查 API Key 是否正确，网络连接是否正常。

### Q3: 响应速度慢？
**A:** 考虑使用更快的模型，或增加 timeout 设置。

### Q4: 成本太高？
**A:** 使用 Mock LLM 开发，生产环境选择性价比高的模型。

## 📚 相关文档

- [README.md](README.md) - 项目总体说明
- [LLM_CONFIG.md](LLM_CONFIG.md) - LLM 详细配置指南
- [examples_llm.py](examples_llm.py) - 代码示例

## 🎉 总结

本次升级成功将狼人杀系统从基于规则的 Agent 进化为基于 LLM 的智能 Agent，实现了：

✅ **多模型支持** - OpenAI、Qwen、DeepSeek、Mock  
✅ **角色专属提示词** - 6 种角色的定制化策略  
✅ **灵活配置** - 命令行参数和环境变量  
✅ **完整文档** - 使用指南和示例代码  
✅ **向后兼容** - 保留原有规则 Agent  

系统现在能够展示真正的多智能体协作与博弈能力，为后续的自演化、评测复盘等进阶功能奠定了坚实基础。
