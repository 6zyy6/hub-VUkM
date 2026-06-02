# Claude Code 逐步完成记录

## Step 1. 锁定接口边界

从原始 README 中确认已有接口边界，而不是重新发明一套新协议：

- `POST /upload/document`
- `POST /chat`

## Step 2. 收敛最小可交付范围

由于原有脚本依赖硬编码模型路径、Milvus 云实例、Kafka 和 MinerU 本地服务，直接继续堆叠会导致作业无法本地验证。因此先把初版目标收敛为：

- 先把 API 跑通
- 先把上传、解析、检索、回答链路拆开
- 先把测试补齐

## Step 3. 拆分职责

把代码拆成三个核心层次：

1. `app/store.py`：文档、chunk、任务队列存储。
2. `app/pipeline.py`：解析、切块、检索、回答、后台处理。
3. `app/main.py`：FastAPI 接口层。

## Step 4. 选择初版替身实现

为了让接口可运行、可测试，先用本地版替代线上组件：

- 解析：`txt/md` 直接读取，`pdf` 用 `pdfplumber`
- 检索：关键词重叠打分
- 回答：规则模板回答
- 存储：内存存储 + 本地上传目录

## Step 5. 给未来预留替换位

当前实现已经为后续升级保留稳定替换点：

- `LocalDocumentParser` -> MinerU / DeepSeek-OCR
- `KeywordRetriever` -> BGE + CLIP + Milvus
- `DraftAnswerGenerator` -> Qwen-VL / Claude Code 多模态生成器
- `InMemoryKnowledgeBaseStore` -> SQLite / MySQL / Milvus / Kafka

## Step 6. 补测试与文档

当前版本已经补了接口级测试和测试说明，满足“架构 + 初步代码 + 测试逻辑”的作业要求。