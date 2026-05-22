import os

class Config:
    # Storage and DB
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/rag_metadata.db")
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    OSS_BASE_DIR = os.getenv("OSS_BASE_DIR", "./data/oss")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_PARSE_TOPIC = "parse_document_topic"

    # External Services (Mock Endpoints)
    MINERU_API_URL = os.getenv("MINERU_API_URL", "http://mineru-service:8000/parse")
    QWEN_VL_API_URL = os.getenv("QWEN_VL_API_URL", "http://qwen-vl-service:8000/generate")
    BGE_EMBEDDING_API_URL = os.getenv("BGE_EMBEDDING_API_URL", "http://bge-service:8000/embed")
    CLIP_EMBEDDING_API_URL = os.getenv("CLIP_EMBEDDING_API_URL", "http://clip-service:8000/embed")

    # Model Params
    TEXT_VECTOR_DIM = 768  # BGE
    IMAGE_VECTOR_DIM = 512 # CLIP

config = Config()

# Ensure directories exist
os.makedirs(config.OSS_BASE_DIR, exist_ok=True)
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
