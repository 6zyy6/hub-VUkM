from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .pipeline import DocumentProcessor, DraftAnswerGenerator, KeywordRetriever, LocalDocumentParser
from .store import InMemoryKnowledgeBaseStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class UploadDocumentResponse(BaseModel):
    document_id: str
    task_id: str
    knowledge_base_id: str
    filename: str
    status: str


class ChatRequest(BaseModel):
    knowledge_base_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class ChatSource(BaseModel):
    document_id: str
    filename: str
    source_page: int | None
    score: float
    preview: str
    images: list[str]


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


def build_storage_path(base_dir: Path, knowledge_base_id: str, filename: str) -> Path:
    safe_name = filename.replace("/", "_").replace("\\", "_")
    target_dir = base_dir / knowledge_base_id
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{uuid4().hex}_{safe_name}"


def create_app(
    *,
    store: InMemoryKnowledgeBaseStore | None = None,
    upload_dir: Path | None = None,
) -> FastAPI:
    data_store = store or InMemoryKnowledgeBaseStore()
    parser = LocalDocumentParser()
    retriever = KeywordRetriever()
    answer_generator = DraftAnswerGenerator()
    processor = DocumentProcessor(data_store, parser)
    resolved_upload_dir = upload_dir or Path(__file__).resolve().parent.parent / "runtime_uploads"

    app = FastAPI(title="Multimodal RAG Chatbot", version="0.1.0")
    app.state.store = data_store
    app.state.processor = processor
    app.state.retriever = retriever
    app.state.answer_generator = answer_generator
    app.state.upload_dir = resolved_upload_dir

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/upload/document", response_model=UploadDocumentResponse)
    async def upload_document(
        background_tasks: BackgroundTasks,
        knowledge_base_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> UploadDocumentResponse:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

        destination = build_storage_path(app.state.upload_dir, knowledge_base_id, file.filename or "document")
        content = await file.read()
        destination.write_bytes(content)

        document = data_store.create_document(
            knowledge_base_id=knowledge_base_id,
            filename=file.filename or destination.name,
            file_path=destination,
            content_type=file.content_type or "application/octet-stream",
        )
        task = data_store.enqueue_task(document.document_id)
        background_tasks.add_task(processor.drain_pending_tasks)

        return UploadDocumentResponse(
            document_id=document.document_id,
            task_id=task.task_id,
            knowledge_base_id=document.knowledge_base_id,
            filename=document.filename,
            status="queued",
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        chunks = data_store.list_chunks_by_kb(request.knowledge_base_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="No processed knowledge found for this knowledge base")

        results = retriever.search(request.question, chunks, limit=request.top_k)
        answer = answer_generator.generate(request.question, results)
        sources = [
            ChatSource(
                document_id=result.chunk.document_id,
                filename=result.chunk.metadata.get("filename", "unknown"),
                source_page=result.chunk.source_page,
                score=result.score,
                preview=result.chunk.text[:200],
                images=result.chunk.images,
            )
            for result in results
        ]
        return ChatResponse(answer=answer, sources=sources)

    return app


app = create_app()