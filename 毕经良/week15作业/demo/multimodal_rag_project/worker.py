import json
import uuid
import time
from app.config import config
from app.services.kafka_service import kafka_service
from app.services.external_services import external_services
from app.services.milvus_service import milvus_client
from app.models.db_models import init_db, Chunk
from app.services.db_service import update_document_status, create_chunk
from app.models.schemas import ParseMessage

def process_task(task: ParseMessage, db_session):
    print(f"Processing task for doc_id: {task.doc_id}")
    update_document_status(db_session, task.doc_id, "parsing")
    
    # 1. Parse Document (MinerU)
    parse_result = external_services.parse_document_mineru(task.file_path)
    
    update_document_status(db_session, task.doc_id, "vectorizing")
    
    # 2. Content Processing & Vectorization
    # Text Processing
    # Mock splitting markdown
    text_chunks = parse_result.markdown.split("\n\n")
    for idx, text in enumerate(text_chunks):
        if not text.strip(): continue
        # Get Embedding (BGE)
        vector = external_services.get_text_embedding(text)
        
        # Save Metadata
        chunk = create_chunk(db_session, task.doc_id, "text", content=text, page_num=1)
        
        # Save to Milvus
        milvus_client.insert_text(chunk.id, task.doc_id, task.kb_id, vector)
        
    # Image Processing
    for idx, img_path in enumerate(parse_result.images):
        # Get Embedding (CLIP)
        vector = external_services.get_image_embedding(img_path)
        
        # Save Metadata
        chunk = create_chunk(db_session, task.doc_id, "image", image_path=img_path, page_num=1)
        
        # Save to Milvus
        milvus_client.insert_image(chunk.id, task.doc_id, task.kb_id, vector)
        
    update_document_status(db_session, task.doc_id, "completed")
    print(f"Completed processing for doc_id: {task.doc_id}")

def run_worker():
    SessionLocal = init_db(config.DATABASE_URL)
    milvus_client.connect()
    consumer = kafka_service.get_consumer()
    print("Worker started, listening for messages...")
    
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
                
            task_data = json.loads(msg.value().decode('utf-8'))
            task = ParseMessage(**task_data)
            
            db_session = SessionLocal()
            try:
                process_task(task, db_session)
            except Exception as e:
                print(f"Error processing task: {e}")
                update_document_status(db_session, task.doc_id, "failed")
            finally:
                db_session.close()
                
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == "__main__":
    run_worker()
