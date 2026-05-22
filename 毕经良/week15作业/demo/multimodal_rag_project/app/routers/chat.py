from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.config import config
from app.models.db_models import init_db, Chunk
from app.models.schemas import ChatRequest, ChatResponse
from app.services.external_services import external_services
from app.services.milvus_service import milvus_client

router = APIRouter(prefix="/chat", tags=["Chat"])
SessionLocal = init_db(config.DATABASE_URL)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=ChatResponse)
async def multimodal_chat(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. Embed query (assuming query is text, use BGE)
    query_vector = external_services.get_text_embedding(request.query)
    
    # 2. Retrieve multimodal context
    # Text Search
    text_results = milvus_client.search_text(query_vector, request.kb_id, top_k=3)
    text_chunk_ids = []
    if text_results and len(text_results) > 0:
        for hits in text_results:
            for hit in hits:
                text_chunk_ids.append(hit.entity.get("chunk_id"))
                
    # Image Search (using text-to-image semantic matching, mock uses CLIP embedding of text)
    image_query_vector = external_services.get_image_embedding(request.query) # CLIP text embedding
    image_results = milvus_client.search_image(image_query_vector, request.kb_id, top_k=2)
    image_chunk_ids = []
    if image_results and len(image_results) > 0:
        for hits in image_results:
            for hit in hits:
                image_chunk_ids.append(hit.entity.get("chunk_id"))

    # 3. Retrieve metadata & content from DB
    context_texts = []
    context_images = []
    sources = []
    
    for chunk_id in text_chunk_ids:
        chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
        if chunk:
            context_texts.append(chunk.content)
            sources.append({"type": "text", "doc_id": chunk.doc_id, "page": chunk.page_num})
            
    for chunk_id in image_chunk_ids:
        chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
        if chunk:
            context_images.append(chunk.image_path)
            sources.append({"type": "image", "doc_id": chunk.doc_id, "image_path": chunk.image_path})

    # 4. Generate answer using Qwen-VL
    answer = external_services.generate_answer_qwen_vl(request.query, context_texts, context_images)
    
    return ChatResponse(
        answer=answer,
        sources=sources
    )
