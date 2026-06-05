# 快速开始指南

## 环境准备

```bash
# Python 3.10+
python --version

# 安装依赖
cd AI_Werewolf_Agent
pip install -e .
```

## 运行游戏

### CLI 模式（终端）

```bash
# 默认智能模式
python main.py

# 简单模式（更快）
python main.py --simple

# 智能模式（更智能的决策）
python main.py --smart
```

### Web 界面模式

```bash
# 安装 streamlit
pip install streamlit

# 启动 Web 界面
streamlit run ui/app.py

# 浏览器打开 http://localhost:8501
```

## 游戏流程

```
============================================================
     AI Werewolf - 全自动狼人杀多智能体对战系统

  6人局: 2狼人 + 1预言家 + 1女巫 + 2村民
============================================================

角色分配:
  Alice: werewolf
  Bob: seer
  ...

第 1 天
  [夜晚] 狼人选择杀害、预言家查验、女巫用药
  [白天] 发言、投票

...

游戏结束!
狼人胜利 / 好人胜利
```

## Web 界面操作

1. 点击「开始新游戏」初始化
2. 点击「继续下一步」单步执行
3. 点击「自动运行」快速完成
4. 实时观看发言、投票、夜晚行动

## 输出文件

- 对局日志：`runs/logs/game_*.json`
- 包含完整游戏记录，可复盘分析