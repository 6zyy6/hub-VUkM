"""
配置文件 - 统一管理所有配置项
支持环境变量覆盖，便于不同环境部署
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DB_PATH = BASE_DIR / "db.db"

# 确保目录存在
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# ==================== 模型配置 ====================
# BGE文本向量化模型
BGE_MODEL_PATH = os.getenv("BGE_MODEL_PATH", "/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5")

# CLIP图文向量化模型
CLIP_MODEL_PATH = os.getenv("CLIP_MODEL_PATH", "/root/autodl-tmp/models/jinaai/jina-clip-v2")
CLIP_OUTPUT_DIM = 1024  # CLIP输出维度

# Qwen大语言模型
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "sk-711c186f74494136ba26035be25a7cb8")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

# ==================== Milvus向量数据库配置 ====================
MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
)
MILVUS_TOKEN = os.getenv(
    "MILVUS_TOKEN",
    "9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_data_new")

# ==================== Kafka配置 ====================
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "rag-data")

# ==================== MinerU配置 ====================
MINERU_BASE_URL = os.getenv("MINERU_BASE_URL", "http://127.0.0.1:30000")
MINERU_BACKEND = os.getenv("MINERU_BACKEND", "vlm-http-client")
MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "600"))  # 秒

# ==================== 处理参数 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))  # 文本分块大小
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5"))  # 检索返回数量
BATCH_INSERT_SIZE = int(os.getenv("BATCH_INSERT_SIZE", "50"))  # Milvus批量插入大小

# ==================== 文件状态枚举 ====================
class FileState:
    UPLOADED = "已上传"
    PARSING = "解析中"
    COMPLETED = "已完成"
    FAILED = "失败"

# ==================== 向量维度常量 ====================
BGE_DIMENSION = 512
CLIP_DIMENSION = 1024

# ==================== HuggingFace镜像 ====================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
