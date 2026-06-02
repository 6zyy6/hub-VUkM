from __future__ import annotations

from dataclasses import dataclass

from rag_chatbot.embedding import cosine_similarity, vectorize
from rag_chatbot.models import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


class InMemoryChunkStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def document_count(self) -> int:
        return len({chunk.document_id for chunk in self._chunks})

    def search(self, query: str, top_k: int = 4) -> list[SearchHit]:
        if top_k <= 0 or not self._chunks:
            return []

        query_vector = vectorize(query)
        hits = [
            SearchHit(chunk=chunk, score=cosine_similarity(query_vector, vectorize(_searchable_text(chunk))))
            for chunk in self._chunks
        ]
        ranked = sorted(hits, key=lambda hit: (hit.score, hit.chunk.heading), reverse=True)
        return [hit for hit in ranked if hit.score > 0][:top_k]


def _searchable_text(chunk: Chunk) -> str:
    return f"{chunk.title}\n{chunk.heading}\n{chunk.text}"
