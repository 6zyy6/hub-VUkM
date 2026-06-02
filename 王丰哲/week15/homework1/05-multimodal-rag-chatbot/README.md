# 05 Multimodal RAG Chatbot

这是一份课程作业级别的初始实现：目标是把多模态文档解析结果接入 RAG Chatbot，先完成清晰需求、架构、测试逻辑和最小可运行代码。

## 快速验证

```bash
cd 05-multimodal-rag-chatbot
PYTHONPATH=src ../.venv/bin/pytest -q
```

## 接口草案

- `GET /health`: 返回服务状态和已索引文档数。
- `POST /documents`: 写入解析后的文档文本、来源和图片引用，返回 `document_id` 与 `chunk_count`。
- `POST /search`: 按问题检索相关 chunk，返回引用和分数。
- `POST /chat`: 检索上下文后生成带引用回答，返回 `answer`、`citations`、`retrieved_chunks`、`used_modalities`。

## 目录

- `docs/requirements.md`: 需求说明。
- `docs/architecture.md`: 架构设计。
- `docs/testing.md`: 测试逻辑。
- `src/rag_chatbot`: 初始代码。
- `tests`: 核心行为测试。
