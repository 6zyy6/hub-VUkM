"""多模态 RAG Chatbot API 入口，接口与 04-government-advanced-rag 对齐。"""

import datetime
import time
import traceback
import uuid
from pathlib import Path

import uvicorn
import yaml
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from typing_extensions import Annotated

from db_api import KnowledgeDatabase, KnowledgeDocument, Session
from rag_api import MultimodalRAG
from router_schemas import (
    DocumentResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    KnowledgeRequest,
    KnowledgeResponse,
    RAGRequest,
    RAGResponse,
    RerankRequest,
    RerankResponse,
)
from vector_store import VectorStore

with open(Path(__file__).parent / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="05-multimodal-rag-chatbot", version="0.1.0")
UPLOAD_DIR = Path("upload_files")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/v1/knowledge_base")
def get_knowledge_base(knowledge_id: int, token: str = "") -> KnowledgeResponse:
    start = time.time()
    with Session() as session:
        record = (
            session.query(KnowledgeDatabase)
            .filter(KnowledgeDatabase.knowledge_id == knowledge_id)
            .first()
        )
        if record:
            return KnowledgeResponse(
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                title=str(record.title),
                category=str(record.category),
                response_code=200,
                response_msg="知识库查询成功",
                process_status="completed",
                processing_time=time.time() - start,
            )
    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=knowledge_id,
        category="",
        title="",
        response_code=404,
        response_msg="知识库不存在",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.delete("/v1/knowledge_base")
def delete_knowledge_base(knowledge_id: int, token: str = "") -> KnowledgeResponse:
    start = time.time()
    with Session() as session:
        record = (
            session.query(KnowledgeDatabase)
            .filter(KnowledgeDatabase.knowledge_id == knowledge_id)
            .first()
        )
        if not record:
            return KnowledgeResponse(
                request_id=str(uuid.uuid4()),
                knowledge_id=knowledge_id,
                category="",
                title="",
                response_code=404,
                response_msg="知识库不存在",
                process_status="completed",
                processing_time=time.time() - start,
            )
        session.delete(record)
        session.commit()
        VectorStore.delete_by_knowledge(knowledge_id)
        return KnowledgeResponse(
            request_id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            category=str(record.category),
            title=str(record.title),
            response_code=200,
            response_msg="知识库删除成功",
            process_status="completed",
            processing_time=time.time() - start,
        )


@app.post("/v1/knowledge_base")
def add_knowledge_base(req: KnowledgeRequest) -> KnowledgeResponse:
    start = time.time()
    try:
        with Session() as session:
            record = KnowledgeDatabase(
                title=req.title,
                category=req.category,
                create_dt=datetime.datetime.now(),
                update_dt=datetime.datetime.now(),
            )
            session.add(record)
            session.flush()
            kid = record.knowledge_id
            session.commit()
        return KnowledgeResponse(
            request_id=str(uuid.uuid4()),
            knowledge_id=kid,
            category=req.category,
            title=req.title,
            response_code=200,
            response_msg="知识库插入成功",
            process_status="completed",
            processing_time=time.time() - start,
        )
    except Exception:
        print(traceback.format_exc())
    return KnowledgeResponse(
        request_id=str(uuid.uuid4()),
        knowledge_id=0,
        category="",
        title="",
        response_code=504,
        response_msg="知识库插入失败",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.get("/v1/document")
def get_document(document_id: int, token: str = "") -> DocumentResponse:
    start = time.time()
    with Session() as session:
        record = (
            session.query(KnowledgeDocument)
            .filter(KnowledgeDocument.document_id == document_id)
            .first()
        )
        if record:
            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                category=str(record.category),
                title=str(record.title),
                knowledge_id=record.knowledge_id,
                file_type=str(record.file_type),
                response_code=200,
                response_msg="文档查询成功",
                process_status="completed",
                processing_time=time.time() - start,
            )
    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=document_id,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg="文档不存在",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.delete("/v1/document")
def delete_document(document_id: int, token: str = "") -> DocumentResponse:
    start = time.time()
    with Session() as session:
        record = (
            session.query(KnowledgeDocument)
            .filter(KnowledgeDocument.document_id == document_id)
            .first()
        )
        if not record:
            return DocumentResponse(
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                category="",
                title="",
                knowledge_id=0,
                file_type="",
                response_code=404,
                response_msg="文档不存在",
                process_status="completed",
                processing_time=time.time() - start,
            )
        session.delete(record)
        session.commit()
        VectorStore.delete_by_document(document_id)
        return DocumentResponse(
            request_id=str(uuid.uuid4()),
            document_id=document_id,
            category=str(record.category),
            title=str(record.title),
            knowledge_id=record.knowledge_id,
            file_type=str(record.file_type),
            response_code=200,
            response_msg="文档删除成功",
            process_status="completed",
            processing_time=time.time() - start,
        )


@app.post("/v1/document")
async def add_document(
    background_tasks: BackgroundTasks,
    knowledge_id: int = Form(...),
    title: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
) -> DocumentResponse:
    start = time.time()
    response_msg = "新增文档失败"
    try:
        with Session() as session:
            kb = (
                session.query(KnowledgeDatabase)
                .filter(KnowledgeDatabase.knowledge_id == knowledge_id)
                .first()
            )
            if kb is None:
                response_msg = "知识库不存在，请提前创建"
                raise ValueError(response_msg)

            record = KnowledgeDocument(
                title=title,
                category=category,
                knowledge_id=knowledge_id,
                file_path="",
                file_type=file.content_type or "application/pdf",
                create_dt=datetime.datetime.now(),
                update_dt=datetime.datetime.now(),
            )
            session.add(record)
            session.flush()
            document_id = record.document_id
            session.commit()

        suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
        file_path = str(UPLOAD_DIR / f"document_id_{document_id}{suffix}")
        content = await file.read()
        Path(file_path).write_bytes(content)

        with Session() as session:
            record = (
                session.query(KnowledgeDocument)
                .filter(KnowledgeDocument.document_id == document_id)
                .first()
            )
            record.file_path = file_path
            session.commit()

        background_tasks.add_task(
            MultimodalRAG().extract_content,
            knowledge_id=knowledge_id,
            document_id=document_id,
            title=title,
            file_type=file.content_type or "application/pdf",
            file_path=file_path,
        )

        return DocumentResponse(
            request_id=str(uuid.uuid4()),
            document_id=document_id,
            category=category,
            title=title,
            knowledge_id=knowledge_id,
            file_type=file.content_type or "application/pdf",
            response_code=200,
            response_msg="文档添加成功",
            process_status="completed",
            processing_time=time.time() - start,
        )
    except Exception:
        print(traceback.format_exc())

    return DocumentResponse(
        request_id=str(uuid.uuid4()),
        document_id=0,
        category="",
        title="",
        knowledge_id=0,
        file_type="",
        response_code=404,
        response_msg=response_msg,
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.post("/v1/embedding")
async def semantic_embedding(req: EmbeddingRequest) -> EmbeddingResponse:
    start = time.time()
    text = [req.text] if isinstance(req.text, str) else req.text
    vector = MultimodalRAG().get_embedding(text)
    return EmbeddingResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.post("/v1/rerank")
async def semantic_rerank(req: RerankRequest) -> RerankResponse:
    start = time.time()
    vector = MultimodalRAG().get_rank(req.text_pair)
    return RerankResponse(
        request_id=str(uuid.uuid4()),
        vector=vector.astype(float).tolist(),
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start,
    )


@app.post("/chat")
def chat(req: RAGRequest) -> RAGResponse:
    start = time.time()
    message = MultimodalRAG().chat_with_rag(req.knowledge_id, req.message)
    return RAGResponse(
        request_id=str(uuid.uuid4()),
        message=message,
        response_code=200,
        response_msg="ok",
        process_status="completed",
        processing_time=time.time() - start,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config["rag"]["port"], workers=1)
