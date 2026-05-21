from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from .store import ChunkRecord, DocumentRecord, InMemoryKnowledgeBaseStore

IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


class UnsupportedDocumentError(RuntimeError):
    pass


@dataclass(slots=True)
class SearchResult:
    chunk: ChunkRecord
    score: float


def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n", text) if segment.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}"
            continue
        chunks.append(current)
        current = paragraph
    if current:
        chunks.append(current)
    return chunks


def extract_images(text: str) -> list[str]:
    return IMAGE_PATTERN.findall(text)


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


class LocalDocumentParser:
    def parse(self, document: DocumentRecord) -> list[ChunkRecord]:
        suffix = document.file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return self._parse_text_like(document)
        if suffix == ".pdf":
            return self._parse_pdf(document)
        raise UnsupportedDocumentError(f"Unsupported file type: {suffix}")

    def _parse_text_like(self, document: DocumentRecord) -> list[ChunkRecord]:
        text = document.file_path.read_text(encoding="utf-8")
        return [
            ChunkRecord(
                chunk_id=uuid4().hex,
                document_id=document.document_id,
                knowledge_base_id=document.knowledge_base_id,
                text=chunk,
                source_page=None,
                images=extract_images(chunk),
                metadata={"filename": document.filename},
            )
            for chunk in chunk_text(text)
        ]

    def _parse_pdf(self, document: DocumentRecord) -> list[ChunkRecord]:
        try:
            import pdfplumber
        except ImportError as exc:
            raise UnsupportedDocumentError("pdfplumber is required for local PDF parsing") from exc

        chunks: list[ChunkRecord] = []
        with pdfplumber.open(document.file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                for chunk in chunk_text(page_text):
                    chunks.append(
                        ChunkRecord(
                            chunk_id=uuid4().hex,
                            document_id=document.document_id,
                            knowledge_base_id=document.knowledge_base_id,
                            text=chunk,
                            source_page=page_number,
                            images=[],
                            metadata={"filename": document.filename, "parser": "pdfplumber"},
                        )
                    )
        if not chunks:
            raise UnsupportedDocumentError("No extractable text found in PDF")
        return chunks


class KeywordRetriever:
    def search(self, query: str, chunks: Iterable[ChunkRecord], *, limit: int = 5) -> list[SearchResult]:
        query_tokens = tokenize(query)
        results: list[SearchResult] = []
        for chunk in chunks:
            chunk_tokens = tokenize(chunk.text)
            overlap = query_tokens & chunk_tokens
            if not overlap:
                continue
            coverage = len(overlap) / max(len(query_tokens), 1)
            density = len(overlap) / max(len(chunk_tokens), 1)
            score = round(coverage * 0.7 + density * 0.3, 4)
            results.append(SearchResult(chunk=chunk, score=score))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]


class DraftAnswerGenerator:
    def generate(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "当前知识库中没有检索到可支撑回答的内容。"

        snippets = []
        for result in results:
            prefix = f"第{result.chunk.source_page}页" if result.chunk.source_page else "文本片段"
            snippets.append(f"- {prefix}：{result.chunk.text[:160].strip()}")

        joined = "\n".join(snippets)
        return (
            f"问题：{question}\n\n"
            "基于当前初版检索链路，命中的关键信息如下：\n"
            f"{joined}\n\n"
            "当前版本为初步代码，回答由规则模板拼装，后续可替换为 Qwen-VL 或 Claude Code 驱动的多模态生成模块。"
        )


class DocumentProcessor:
    def __init__(self, store: InMemoryKnowledgeBaseStore, parser: LocalDocumentParser) -> None:
        self._store = store
        self._parser = parser

    def drain_pending_tasks(self) -> int:
        processed_count = 0
        while True:
            task = self._store.dequeue_task()
            if task is None:
                return processed_count
            self.process_document(task.document_id)
            processed_count += 1

    def process_document(self, document_id: str) -> None:
        document = self._store.get_document(document_id)
        if document is None:
            return

        self._store.update_document_status(document_id, status="processing")
        try:
            chunks = self._parser.parse(document)
        except UnsupportedDocumentError as exc:
            self._store.update_document_status(document_id, status="failed", error_message=str(exc))
            return
        except Exception as exc:
            self._store.update_document_status(document_id, status="failed", error_message=f"Unexpected parser error: {exc}")
            return

        self._store.replace_document_chunks(document_id, chunks)
        self._store.update_document_status(document_id, status="processed")