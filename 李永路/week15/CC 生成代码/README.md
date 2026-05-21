# Multilingual PDF Knowledge Base QA System

## Project Overview

A multimodal RAG system that processes PDF documents containing text and images, enabling intelligent question answering over the knowledge base.

## Features

- PDF Document Parsing (MinerU/DeepSeek-OCR → Markdown + Images)
- Multimodal Vector Storage (Text + Image embeddings in Milvus)
- Cross-modal Retrieval (CLIP for text↔image retrieval)
- Multimodal QA (Qwen-VL for reasoning over retrieved content)

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │───▶│   Kafka     │───▶│   Worker    │
│  (upload)   │    │   (queue)   │    │  (parsing)  │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │───▶│  Milvus     │◀───│   Chunking  │
│   (chat)    │    │  (vectors)  │    │   + Embed   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## API Endpoints

- `POST /upload/document` - Upload PDF to knowledge base
- `POST /chat` - Multimodal QA over knowledge base

## Evaluation

- Page Match (0.25 points)
- Filename Match (0.25 points)
- Answer Similarity via Jaccard (0.5 points)