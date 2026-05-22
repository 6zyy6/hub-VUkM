from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    knowledge_base_id: str
    filename: str
    file_path: Path
    content_type: str
    status: str = "uploaded"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    error_message: str | None = None


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    knowledge_base_id: str
    text: str
    source_page: int | None
    images: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessingTask:
    task_id: str
    document_id: str
    enqueued_at: datetime = field(default_factory=utc_now)


class InMemoryKnowledgeBaseStore:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks_by_document: dict[str, list[ChunkRecord]] = {}
        self._tasks: deque[ProcessingTask] = deque()
        self._lock = Lock()

    def create_document(
        self,
        *,
        knowledge_base_id: str,
        filename: str,
        file_path: Path,
        content_type: str,
    ) -> DocumentRecord:
        record = DocumentRecord(
            document_id=uuid4().hex,
            knowledge_base_id=knowledge_base_id,
            filename=filename,
            file_path=file_path,
            content_type=content_type,
        )
        with self._lock:
            self._documents[record.document_id] = record
        return record

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)

    def update_document_status(self, document_id: str, *, status: str, error_message: str | None = None) -> None:
        with self._lock:
            record = self._documents[document_id]
            record.status = status
            record.error_message = error_message
            record.updated_at = utc_now()

    def enqueue_task(self, document_id: str) -> ProcessingTask:
        task = ProcessingTask(task_id=uuid4().hex, document_id=document_id)
        with self._lock:
            self._tasks.append(task)
        return task

    def dequeue_task(self) -> ProcessingTask | None:
        with self._lock:
            if not self._tasks:
                return None
            return self._tasks.popleft()

    def replace_document_chunks(self, document_id: str, chunks: list[ChunkRecord]) -> None:
        with self._lock:
            self._chunks_by_document[document_id] = chunks

    def list_chunks_by_kb(self, knowledge_base_id: str) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for document_id, document in self._documents.items():
            if document.knowledge_base_id != knowledge_base_id:
                continue
            chunks.extend(self._chunks_by_document.get(document_id, []))
        return chunks

    def list_documents_by_kb(self, knowledge_base_id: str) -> list[DocumentRecord]:
        return [record for record in self._documents.values() if record.knowledge_base_id == knowledge_base_id]