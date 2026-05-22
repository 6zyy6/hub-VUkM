"""文档上传 API"""
import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import logging

from app.core.config import get_config
from app.core.deps import get_kafka_producer, get_milvus
from app.models.data_models import UploadResponse, DocumentStatus
from app.services.mq import KafkaProducerService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/document", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    uploader_id: Optional[str] = None
):
    """
    上传 PDF 文档

    - 保存文件到本地存储
    - 发送消息到 Kafka 进行异步处理
    - 返回 document_id 供后续查询
    """
    # 验证文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    config = get_config()
    os.makedirs(config.storage.upload_dir, exist_ok=True)

    # 生成文档 ID
    document_id = str(uuid.uuid4())
    filename = f"{document_id}_{file.filename}"
    file_path = os.path.join(config.storage.upload_dir, filename)

    # 保存文件
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)
        logger.info(f"Document uploaded: {document_id}, size: {file_size} bytes")

    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    # 发送 Kafka 消息
    try:
        kafka_producer = get_kafka_producer()
        kafka_producer.send_pdf_process_message(document_id, file_path)
        logger.info(f"PDF processing message sent to Kafka: {document_id}")
    except Exception as e:
        logger.error(f"Failed to send Kafka message: {e}")
        # 文件已保存，记录失败但不阻止上传
        status = DocumentStatus.UPLOADED  # 状态保持上传，worker 可稍后处理
    else:
        status = DocumentStatus.PROCESSING

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        status=status,
        message="Document uploaded successfully" if status == DocumentStatus.PROCESSING else "Document uploaded, processing queued"
    )


@router.get("/document/{document_id}/status")
async def get_document_status(document_id: str):
    """
    查询文档处理状态

    Returns:
        - uploaded: 已上传，等待处理
        - processing: 处理中
        - completed: 处理完成
        - failed: 处理失败
    """
    # TODO: 从数据库查询实际状态（当前用占位实现）
    # 这里应该连接 MongoDB/PostgreSQL 查询
    return {
        "document_id": document_id,
        "status": DocumentStatus.PROCESSING,
        "progress": 0.5,
        "message": "Processing in progress"
    }


@router.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """删除文档及其关联数据"""
    config = get_config()

    # 查找并删除上传的文件
    upload_dir = config.storage.upload_dir
    for filename in os.listdir(upload_dir):
        if filename.startswith(document_id):
            os.remove(os.path.join(upload_dir, filename))
            logger.info(f"Deleted file: {filename}")

    # 从 Milvus 删除向量数据
    try:
        milvus = get_milvus()
        milvus.delete_by_document_id(document_id, "mmrag_text")
        milvus.delete_by_document_id(document_id, "mmrag_image")
        logger.info(f"Deleted vectors for document: {document_id}")
    except Exception as e:
        logger.warning(f"Failed to delete vectors: {e}")

    return {"message": f"Document {document_id} deleted"}