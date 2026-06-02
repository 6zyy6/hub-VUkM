"""
API endpoints for document management and multimodal QA.
"""

import os
import uuid
import shutil
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.models.schemas import (
    UploadDocumentResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse
)
from app.services import (
    DocumentParser,
    ChunkProcessor,
    EmbeddingService,
    VectorStore,
    MultimodalQA,
    SearchResult
)

router = APIRouter()

# Global service instances
embedding_service = EmbeddingService()
vector_store = VectorStore()
qa_engine = MultimodalQA()
chunk_processor = ChunkProcessor()


def get_vector_store() -> VectorStore:
    """Dependency to get vector store instance."""
    if not vector_store._connected:
        vector_store.connect()
        vector_store.create_collection()
    return vector_store


@router.post("/upload/document", response_model=UploadDocumentResponse)
async def upload_document(
    knowledge_base_id: str,
    file: UploadFile = File(...),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Upload a PDF document to a knowledge base.

    Steps:
    1. Save uploaded PDF to local storage
    2. Insert parsing task into queue (Kafka in production)
    3. Return document ID for tracking
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate document ID
    document_id = str(uuid.uuid4())[:12]

    # Save file to storage
    upload_dir = Path("storage/documents") / knowledge_base_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{document_id}.pdf"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # In production: send to Kafka for async processing
    # For now: parse synchronously (for demo purposes)
    try:
        # Parse document
        parser = DocumentParser(parser_type="mineru")
        parsed_doc = parser.parse(str(file_path), output_dir="storage/markdown")

        # Chunk and embed text
        chunks = chunk_processor.chunk_markdown(parsed_doc.markdown_content, document_id)
        if chunks:
            text_embeddings = embedding_service.embed_text([c["text"] for c in chunks])
            vector_store.insert_text_chunks(chunks, text_embeddings)

        # Embed and store images
        if parsed_doc.images:
            image_paths = [img["path"] for img in parsed_doc.images]
            image_embeddings = embedding_service.embed_image(image_paths)
            vector_store.insert_images(parsed_doc.images, image_embeddings)

        return UploadDocumentResponse(
            document_id=document_id,
            status="success",
            message=f"Document parsed and indexed successfully"
        )

    except Exception as e:
        return UploadDocumentResponse(
            document_id=document_id,
            status="error",
            message=f"Parsing failed: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Multimodal QA over a knowledge base.

    Steps:
    1. Embed user question
    2. Retrieve relevant text and images from vector store
    3. Use Qwen-VL to generate answer with retrieved content
    """
    try:
        # Get text and image query vectors
        text_embedding = embedding_service.embed_text([request.question])
        # Use same embedding for hybrid search initially
        image_embedding = embedding_service.embed_text([request.question])

        # Search vector store
        retrieved_results = vector_store.search_hybrid(
            text_query_vector=text_embedding[0],
            image_query_vector=image_embedding[0],
            top_k=request.top_k,
            document_id=None  # Search across all documents in knowledge base
        )

        # Generate answer with Qwen-VL
        qa_response = qa_engine.answer(
            question=request.question,
            retrieved_content=retrieved_results,
            knowledge_base_id=request.knowledge_base_id
        )

        # Convert sources to response format
        sources = [
            {
                "content": src["content"],
                "type": src["type"],
                "page_num": src["page_num"],
                "document_id": src["document_id"]
            }
            for src in qa_response.sources
        ]

        return ChatResponse(
            answer=qa_response.answer,
            sources=sources,
            score=qa_response.score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.delete("/document/{document_id}")
async def delete_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Delete a document from the knowledge base."""
    try:
        vector_store.delete_by_document_id(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))