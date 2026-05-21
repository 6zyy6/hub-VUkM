from sqlalchemy.orm import Session
from app.models.db_models import Document, Chunk
import uuid

def create_document(db: Session, kb_id: str, filename: str, uploader: str, file_path: str):
    doc = Document(
        id=str(uuid.uuid4()),
        kb_id=kb_id,
        filename=filename,
        uploader=uploader,
        file_path=file_path,
        status="uploaded"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

def update_document_status(db: Session, doc_id: str, status: str):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        doc.status = status
        db.commit()
        db.refresh(doc)
    return doc

def get_document(db: Session, doc_id: str):
    return db.query(Document).filter(Document.id == doc_id).first()

def create_chunk(db: Session, doc_id: str, chunk_type: str, content: str = None, image_path: str = None, page_num: int = None):
    chunk = Chunk(
        id=str(uuid.uuid4()),
        doc_id=doc_id,
        chunk_type=chunk_type,
        content=content,
        image_path=image_path,
        page_num=page_num
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk
