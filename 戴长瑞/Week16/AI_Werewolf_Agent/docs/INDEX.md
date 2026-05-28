# 文档目录

## 快速开始

- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南

## 模块文档

- [AGENTS.md](AGENTS.md) - Agent 模块文档
- [ENGINE.md](ENGINE.md) - 对局引擎文档
- [ROLES.md](ROLES.md) - 角色系统文档

## 使用指南

- [USAGE.md](USAGE.md) - 详细使用指南
- [API.md](API.md) - API 参考文档

## 文件结构

```
docs/
├── INDEX.md         # 本文档
├── QUICKSTART.md    # 快速开始
├── AGENTS.md        # Agent 模块
├── ENGINE.md        # 对局引擎
├── ROLES.md         # 角色系统
├── USAGE.md         # 使用指南
└── API.md           # API 参考
```

## 阅读顺序建议

1. 先阅读 [QUICKSTART.md](QUICKSTART.md) 快速了解如何运行
2. 阅读 [AGENTS.md](AGENTS.md) 了解 Agent 架构
3. 阅读 [ENGINE.md](ENGINE.md) 了解对局流程
4. 阅读 [ROLES.md](ROLES.md) 了解角色系统
5. 阅读 [USAGE.md](USAGE.md) 学习如何使用
6. 查阅 [API.md](API.md) 了解具体 API

## 关键概念

### 信息隔离
每个 Agent 只能访问自己的私有信息，通过 GameContext 实现。

### 状态机
游戏通过状态机管理阶段转换：WAITING → NIGHT → DAY → VOTE → ...

### 记忆系统
Agent 通过 Memory 类记录发言、投票、夜晚行动等历史。

### 策略模式
GeneralAgent 支持动态策略，可以通过 add_strategy 添加自定义策略。

## 代码示例

参见各模块文档中的代码示例。