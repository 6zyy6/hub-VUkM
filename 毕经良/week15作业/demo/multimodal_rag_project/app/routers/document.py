import os
import shutil
from fastapi import APIRouter, File, UploadFile, Depends, Form
from sqlalchemy.orm import Session
from app.config import config
from app.models.db_models import init_db
from app.services.db_service import create_document
from app.services.kafka_service import kafka_service
from app.models.schemas import UploadResponse, ParseMessage

router = APIRouter(prefix="/document", tags=["Document"])
SessionLocal = init_db(config.DATABASE_URL)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
    uploader: str = Form("anonymous"),
    db: Session = Depends(get_db)
):
    # 1. Save file locally (acting as OSS)
    file_path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Record metadata
    doc = create_document(db, kb_id, file.filename, uploader, file_path)
    
    # 3. Send parsing task to Kafka
    task = ParseMessage(doc_id=doc.id, file_path=file_path, kb_id=kb_id)
    kafka_service.send_parse_task(task)
    
    return UploadResponse(
        doc_id=doc.id,
        filename=file.filename,
        status="uploaded",
        message="Document uploaded and parsing task submitted."
    )
