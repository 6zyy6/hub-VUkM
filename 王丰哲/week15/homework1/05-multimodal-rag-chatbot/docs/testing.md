# 测试逻辑

## 测试策略

当前阶段优先覆盖服务层行为，因为它决定接口返回是否稳定。FastAPI 层保持薄适配，后续接入真实依赖后再补 HTTP 集成测试。

## 已覆盖行为

- `test_chunk_markdown_preserves_headings_and_image_references`
  - 验证 Markdown 标题会进入 chunk 元数据。
  - 验证图片引用会被提取为结构化 `images`。
  - 验证 chunk id 使用稳定前缀，便于引用。

- `test_chatbot_ingests_document_and_returns_grounded_answer_with_citations`
  - 验证文档可写入。
  - 验证聊天回答包含检索到的知识。
  - 验证返回 citations、retrieved_chunks 和 used_modalities。

- `test_chatbot_search_ranks_semantically_relevant_chunks_first`
  - 验证检索能把与问题词汇最相关的 chunk 排在前面。

- `test_chatbot_returns_safe_empty_state_answer_before_ingestion`
  - 验证知识库为空时返回安全提示，不编造答案。

## 验证命令

```bash
cd 05-multimodal-rag-chatbot
PYTHONPATH=src ../.venv/bin/pytest -q
```
