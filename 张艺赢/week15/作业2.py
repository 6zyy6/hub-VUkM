# 作业2：MinerU 与 pdfplumber 对比笔记
## MinerU 论文与产品要点
- 论文：[MinerU: An Open-Source Solution for Precise Document Content Extraction](https://arxiv.org/abs/2409.18839)（上海 AI Lab / OpenDataLab）
- 管线：预处理 → **版面分析**（标题/段落/图/表区域）→ 分区域识别（OCR、公式、表格）→ **阅读顺序重排** → 输出 Markdown/JSON
- 开源工具页：<https://mineru.net/OpenSourceTools/Extractor>，CLI：`mineru -p <input> -o <output>`
## 本地实测（`scripts/compare_parsers.py`）
样例 PDF：`fixtures/sample.pdf`（reportlab 生成，含中英文说明）
| 维度 | pdfplumber | MinerU（本机未装 CLI） |
|------|------------|------------------------|
| 安装 | `pip install pdfplumber`，秒级 | `mineru[core]` 体积大，常需 GPU/模型缓存 |
| 原理 | 基于 PDF 字符流/坐标 `extract_text()` | 深度学习版面 + OCR + 表格/公式专用模型 |
| 页数 | 1 页 → 1 段文本，约 66 字符 | 未运行；预期输出完整 `.md` + 可选 `middle.json` |
| 表格 | `extract_tables()` 简单网格，复杂合并单元格易错 | 表格 → HTML，跨页表可合并 |
| 图片 | 不导出图片路径，多模态 RAG 需另做 | 提取图片文件 + Markdown 引用 |
| 公式/扫描件 | 无专门模块，扫描 PDF 常乱序/空文本 | 公式 → LaTeX；扫描件自动 OCR |
| 页眉页脚 | 与正文混在一起 | 规则/模型剔除，语义更连贯 |
| 适用场景 | 可复制文本的朴素 PDF、快速原型 | 教材、论文、多栏排版、图文混排知识库 |
## 结论（文字作答摘要）
**pdfplumber** 是「PDF 文本/表格抽取库」，实现快、依赖少，适合政府/合同类**单栏、文字为主**的文档，与 Week04 `rag_api._extract_pdf_content` 一致；但对**多栏、图表、公式、扫描件**容易顺序错乱或漏提，**无法直接服务多模态 RAG**（`chunk_images` 为空）。
**MinerU** 是「面向 LLM/RAG 的文档解析系统」，用 PDF-Extract-Kit 做版面理解与结构化导出，输出更接近人类阅读顺序的 Markdown，表格/公式/图片信息完整，更适合作为 **05-multimodal-rag-chatbot** 的 `parser.backend=mineru`；代价是算力、安装复杂度和解析时延更高。
**推荐组合**：开发/简单 PDF 用 pdfplumber 验证接口；生产知识库 ingest 用 MinerU；检索层仍用文本向量 + 图片路径/caption 的多模态扩展（见 `docs/ARCHITECTURE.md` P2/P3）。
  
