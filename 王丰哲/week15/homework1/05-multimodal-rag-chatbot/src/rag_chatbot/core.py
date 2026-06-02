from __future__ import annotations

import re
from typing import Any

from rag_chatbot.chunking import chunk_markdown
from rag_chatbot.embedding import tokenize
from rag_chatbot.models import Chunk
from rag_chatbot.store import InMemoryChunkStore, SearchHit


class MultimodalRAGChatbot:
    def __init__(self, store: InMemoryChunkStore | None = None) -> None:
        self.store = store or InMemoryChunkStore()

    def ingest_document(
        self,
        *,
        title: str,
        source: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chunks = chunk_markdown(title=title, source=source, content=content, metadata=metadata)
        self.store.add(chunks)
        document_id = chunks[0].document_id if chunks else "doc_empty"
        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "source": source,
            "title": title,
        }

    def search(self, question: str, top_k: int = 4) -> list[dict[str, Any]]:
        return [_hit_to_dict(hit) for hit in self.store.search(question, top_k=top_k)]

    def chat(self, question: str, top_k: int = 4) -> dict[str, Any]:
        hits = self.store.search(question, top_k=top_k)
        if not hits:
            return {
                "answer": "当前知识库为空，请先上传或索引文档。" if not self.store.chunks else "没有检索到足够相关的资料，建议换一种问法或补充文档。",
                "citations": [],
                "retrieved_chunks": [],
                "used_modalities": {"text": False, "images": False},
            }

        answer = _compose_answer(question, [hit.chunk for hit in hits])
        citations = [_citation(hit) for hit in hits]
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": [_hit_to_dict(hit) for hit in hits],
            "used_modalities": {
                "text": True,
                "images": any(hit.chunk.images for hit in hits),
            },
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "documents": self.store.document_count(),
            "chunks": len(self.store.chunks),
        }


def build_default_chatbot() -> MultimodalRAGChatbot:
    return MultimodalRAGChatbot()


def _compose_answer(question: str, chunks: list[Chunk]) -> str:
    query_tokens = set(tokenize(question))
    snippets: list[str] = []

    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            sentence_tokens = set(tokenize(sentence))
            if sentence_tokens & query_tokens:
                snippets.append(sentence)
                break
        else:
            snippets.append(chunk.text.splitlines()[0])

    compact = " ".join(snippet.strip() for snippet in snippets if snippet.strip())
    return f"基于已索引资料：{compact}"


def _sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("![")]
    joined = " ".join(lines)
    return [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", joined) if part.strip()]


def _citation(hit: SearchHit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk.chunk_id,
        "source": hit.chunk.source,
        "heading": hit.chunk.heading,
        "score": round(hit.score, 4),
    }


def _hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        **_citation(hit),
        "title": hit.chunk.title,
        "text": hit.chunk.text,
        "images": hit.chunk.images,
    }
