from sqlalchemy import Column, String, Integer, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    kb_id = Column(String, index=True)
    filename = Column(String)
    uploader = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)
    status = Column(String) # 'uploaded', 'parsing', 'vectorizing', 'completed', 'failed'
    file_path = Column(String)

class Chunk(Base):
    __tablename__ = 'chunks'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id = Column(String, index=True)
    chunk_type = Column(String) # 'text' or 'image'
    page_num = Column(Integer, nullable=True)
    content = Column(Text, nullable=True) # Text content or Markdown
    image_path = Column(String, nullable=True) # Path to image in OSS
    
def init_db(database_url: str):
    engine = create_engine(database_url, connect_args={"check_same_thread": False} if "sqlite" in database_url else {})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal
