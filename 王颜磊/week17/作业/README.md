# LLM Smart Cache — 基于 RedisVL 的智能缓存系统

## 项目简介

用 Redis 官方 **RedisVL** 库重新实现课程中四个 LLM 缓存模块，展示生产级方案。

## 四个模块

| 模块 | 文件 | 对应课程原始文件 | 升级点 |
|------|------|-----------------|--------|
| Embedding 缓存 | `embeddings_cache.py` | `EmbeddingsCache.py` | MD5+Redis String → RedisVL 向量索引 |
| 语义缓存 | `semantic_cache.py` | `SemanticCache.py` | FAISS 本地 → Redis 服务端向量索引 |
| 对话历史 | `semantic_message_history.py` | `SemanticMessageHistory.py` | Levenshtein → 向量语义搜索 |
| 语义路由 | `semantic_router.py` | `SemanticRouter.py` | 空接口 → 完整向量路由 |

## 环境要求

- Python >= 3.9
- Docker Desktop（运行 Redis Stack）
- 首次运行会自动下载 `sentence-transformers/all-MiniLM-L6-v2` 模型（约 80MB）

## 快速开始

### 1. 启动 Redis Stack

双击 `start_redis.bat`，或在终端执行：
```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

验证：
```bash
docker exec redis-stack redis-cli ping
# 应返回 PONG
```

### 2. 安装 Python 依赖

```bash
cd "E:\BaiduNetdiskDownload\八斗学院\第17周：大模型部署与项目部署\作业"
pip install -r requirements.txt
```

### 3. 运行演示

```bash
python main.py
```

单独测试某个模块：
```bash
python embeddings_cache.py
python semantic_cache.py
python semantic_message_history.py
python semantic_router.py
```

## 演示流程

```
用户提问
  │
  ├─ [1] SemanticRouter   → 识别意图，路由到对应 handler
  ├─ [2] SemanticCache    → 检查语义缓存，命中则直接返回
  ├─ [3] EmbeddingsCache  → 若未命中，缓存 Embedding 避免重复计算
  └─ [4] MessageHistory   → 检索相关对话历史作为 LLM 上下文
```

## 管理工具

Redis Insight 可视化界面：http://localhost:8001
