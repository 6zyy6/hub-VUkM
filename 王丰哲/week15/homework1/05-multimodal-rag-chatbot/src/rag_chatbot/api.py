from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag_chatbot.core import build_default_chatbot

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent.
    FastAPI = None  # type: ignore[assignment]
    FASTAPI_IMPORT_ERROR = exc
else:
    FASTAPI_IMPORT_ERROR = None


class DocumentRequest(BaseModel):
    title: str
    source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)


def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi is required to create the API app") from FASTAPI_IMPORT_ERROR

    app = FastAPI(title="05 Multimodal RAG Chatbot", version="0.1.0")
    chatbot = build_default_chatbot()

    @app.get("/health")
    def health():
        return chatbot.health()

    @app.post("/documents")
    def ingest_document(payload: DocumentRequest):
        return chatbot.ingest_document(
            title=payload.title,
            source=payload.source,
            content=payload.content,
            metadata=payload.metadata,
        )

    @app.post("/search")
    def search(payload: SearchRequest):
        return {"results": chatbot.search(payload.question, top_k=payload.top_k)}

    @app.post("/chat")
    def chat(payload: ChatRequest):
        return chatbot.chat(payload.question, top_k=payload.top_k)

    return app


app = create_app()
