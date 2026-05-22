# 架构设计

## 推荐方案

采用“解析结果入库 + 检索增强生成”的轻量分层架构。当前实现先用内存索引和确定性词袋检索跑通接口，后续可以把每个端口替换成真实服务。

## 模块

- `chunking.py`: 将 Markdown 按标题和长度切分，提取图片引用。
- `models.py`: 定义 `Chunk` 等核心数据对象。
- `embedding.py`: 提供确定性文本向量化与相似度计算。
- `store.py`: 内存 chunk 存储与检索。
- `core.py`: 编排 ingest、search、chat。
- `api.py`: FastAPI 接口适配，不承载业务逻辑。

## 数据流

1. `POST /documents` 提交 `{title, source, content}`。
2. `chunking` 生成 chunk，并保留 heading、source、images。
3. `store` 写入 chunk，检索时按 query 相似度排序。
4. `chat` 取 top-k chunk，构造模板化回答和 citations。
5. `POST /chat` 返回答案、引用、检索片段、多模态使用情况。

## 后续替换点

- `embedding.py` 可替换为 Qwen Embedding / OpenAI Embeddings / 本地 bge。
- `store.py` 可替换为 FAISS、Milvus、pgvector。
- `core.py` 的回答生成可替换为 OpenAI-compatible chat completion。
- `chunking.py` 可直接消费 MinerU 输出的 `middle.json` 或 Markdown。
