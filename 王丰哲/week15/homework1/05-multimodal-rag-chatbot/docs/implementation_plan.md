# 05 Multimodal RAG Chatbot Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to continue this plan task-by-task.

**Goal:** 完成多模态 RAG Chatbot 的需求、架构、测试逻辑和初步接口实现。

**Architecture:** 上游解析工具输出 Markdown/JSON，服务层负责 chunk、检索、回答编排，FastAPI 层只做请求响应适配。当前实现使用内存存储和确定性词袋检索，确保离线可测试。

**Tech Stack:** Python 3.12, FastAPI, Pydantic, pytest, Markdown chunking.

---

## 文件结构

- `src/rag_chatbot/chunking.py`: Markdown 切分与图片引用提取。
- `src/rag_chatbot/embedding.py`: 离线词袋向量与相似度。
- `src/rag_chatbot/store.py`: 内存 chunk 索引。
- `src/rag_chatbot/core.py`: ingest/search/chat 编排。
- `src/rag_chatbot/api.py`: FastAPI 接口。
- `tests/test_chunking.py`: chunking 行为测试。
- `tests/test_chatbot.py`: ingest/search/chat 行为测试。
- `docs/requirements.md`: 作业需求。
- `docs/architecture.md`: 架构说明。
- `docs/testing.md`: 测试逻辑。

## 已完成步骤

- [x] Step 1: 写 failing tests。

```bash
PYTHONPATH=src ../.venv/bin/pytest -q
```

初始预期：`ModuleNotFoundError: No module named 'rag_chatbot'`。

- [x] Step 2: 补 `chunking.py`、`models.py`、`embedding.py`、`store.py`、`core.py` 最小实现。

- [x] Step 3: 运行服务层测试。

```bash
PYTHONPATH=src ../.venv/bin/pytest -q
```

结果：`4 passed`。

- [x] Step 4: 补 `api.py`，实现 `/health`、`/documents`、`/search`、`/chat`。

- [x] Step 5: 验证 API app 可导入。

```bash
PYTHONPATH=src ../.venv/bin/python - <<'PY'
from rag_chatbot.api import app
print(app.title)
PY
```

结果：`05 Multimodal RAG Chatbot`。

## 后续可继续做

- [ ] 增加 FastAPI `TestClient` 的 HTTP 集成测试。
- [ ] 把 `embedding.py` 替换为真实 embedding client，并补失败重试。
- [ ] 把 `store.py` 替换为 FAISS/Milvus/pgvector。
- [ ] 接入 MinerU 输出目录，自动读取 `.md`、`middle.json` 和图片。
- [ ] 接入 OpenAI-compatible chat completion，要求回答必须引用检索片段。
