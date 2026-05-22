# 第十五周作业提交说明

## 作业 1：Claude Code + 05-multimodal-rag-chatbot

本次提交的重点不是把 MinerU、Milvus、Kafka、Qwen-VL 全部在线跑起来，而是先把原始需求沉淀成一份可以继续开发的工程骨架。

### 已交付内容

1. 需求文档：`docs/homework_requirements.md`
2. 测试逻辑：`docs/test_plan.md`
3. Claude Code 分步完成记录：`docs/claude_code_steps.md`
4. 初步代码：`app/main.py`、`app/store.py`、`app/pipeline.py`
5. 接口测试：`tests/test_api.py`
6. 依赖清单：`requirements.txt`

### 目前实现到什么程度

- 已实现接口：`POST /upload/document`、`POST /chat`
- 已实现能力：
  - 上传文档并保存到本地
  - 生成文档元数据
  - 入队并触发后台解析
  - 对 `txt/md/pdf` 做初版解析
  - 基于 chunk 做检索
  - 返回带来源的草稿回答

### 为什么这样实现

原始课程代码里把云端 Milvus、Kafka、模型路径、MinerU 命令行全部硬编码在脚本里，直接提交容易出现“老师本地无法运行、也无法验证”的问题。因此本次先把接口和架构固定，再把底层组件替换为生产级实现。

### 后续升级路线

1. `LocalDocumentParser` 替换成 MinerU。
2. `KeywordRetriever` 替换成 BGE + CLIP + Milvus。
3. `DraftAnswerGenerator` 替换成 Qwen-VL 或 Claude Code 多模态生成。
4. `InMemoryKnowledgeBaseStore` 替换成数据库 + 消息队列。

## 作业 2：MinerU 与 pdfplumber 对比

文字回答见：

- `../../03_MinerU_vs_pdfplumber.md`