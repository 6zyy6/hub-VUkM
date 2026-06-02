"""多模态对话 API"""
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from fastapi import UploadFile, File
import logging
import os
import shutil

from app.core.config import get_config
from app.core.deps import get_hybrid_retrieval, get_qwen_vl, get_bge_embedding, get_clip_embedding
from app.models.data_models import ChatRequest, ChatResponse, SearchResult
from app.services.retrieval import build_retrieval_context
from app.services.llm import QwenVLService
from app.services.retrieval import HybridRetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# 会话存储 (生产环境应使用 Redis)
_sessions = {}


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    images: Optional[List[UploadFile]] = File(default=None)
):
    """
    多模态对话接口

    - 接收用户问题（文本 + 可选图片）
    - 混合检索相关文档块
    - 使用 Qwen-VL 生成答案

    Args:
        request.query: 用户问题
        request.session_id: 会话 ID（可选，自动创建）
        request.image_paths: 已有图片路径（可选）
        request.top_k: 检索数量

    Returns:
        答案 + 来源信息
    """
    config = get_config()

    # 处理上传的图片
    uploaded_image_paths = []
    if images:
        os.makedirs(config.storage.image_dir, exist_ok=True)
        for img in images:
            if img.content_type.startswith("image/"):
                image_id = str(uuid.uuid4())
                image_path = os.path.join(config.storage.image_dir, f"{image_id}_{img.filename}")
                with open(image_path, "wb") as f:
                    shutil.copyfileobj(img.file, f)
                uploaded_image_paths.append(image_path)

    # 合并图片路径
    all_image_paths = request.image_paths + uploaded_image_paths

    # 获取或创建会话
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = {"history": []}

    # 检索相关文档
    retrieval_service = get_hybrid_retrieval()
    try:
        search_results = retrieval_service.retrieve(
            query=request.query,
            query_images=all_image_paths if all_image_paths else None,
            top_k=request.top_k
        )
        logger.info(f"Retrieved {len(search_results)} results for query")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        search_results = []

    # 构建上下文
    context = build_retrieval_context(search_results)

    # 调用 Qwen-VL 生成答案
    qwen_service = get_qwen_vl()
    try:
        answer = qwen_service.chat(
            query=request.query,
            context=context,
            images=all_image_paths if all_image_paths else None
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        answer = "抱歉，生成答案时出现错误。请稍后重试。"

    # 保存对话历史
    _sessions[session_id]["history"].append({
        "role": "user",
        "content": request.query,
        "images": all_image_paths
    })
    _sessions[session_id]["history"].append({
        "role": "assistant",
        "content": answer
    })

    return ChatResponse(
        answer=answer,
        sources=search_results,
        session_id=session_id,
        images=all_image_paths
    )


@router.get("/session/{session_id}/history")
async def get_chat_history(session_id: str):
    """获取会话历史"""
    if session_id not in _sessions:
        return {"history": []}
    return _sessions[session_id]


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"message": f"Session {session_id} deleted"}


@router.post("/search")
async def search_documents(query: str, top_k: int = 5):
    """仅检索（不生成答案）"""
    retrieval_service = get_hybrid_retrieval()
    try:
        results = retrieval_service.retrieve(query=query, top_k=top_k)
        return {"results": [r.dict() for r in results], "total": len(results)}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))