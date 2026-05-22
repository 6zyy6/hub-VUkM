"""PDF 处理 Worker - 异步解析 PDF"""
import os
import json
import uuid
import logging
from typing import Dict, Any, List

from app.core.factory import ServiceFactory, initialize_factories
from app.core.config import get_config
from app.models.data_models import ChunkType
from app.services.pdf_parser import MinerUParser, ParseResult
from app.services.embedding import BGEEmbedding, CLIPEmbedding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PDFWorker:
    """PDF 处理 Worker"""

    def __init__(self):
        self.config = get_config()
        self.pdf_parser = None
        self.bge = None
        self.clip = None
        self.milvus = None

    def initialize(self):
        """初始化服务"""
        initialize_factories()

        # PDF 解析器
        self.pdf_parser = ServiceFactory.get("pdf_parser_mineru")

        # Embedding 服务
        self.bge = ServiceFactory.get("embedding_bge")
        self.clip = ServiceFactory.get("embedding_clip")

        # Milvus
        self.milvus = ServiceFactory.get("vector_db_milvus")
        self.milvus.initialize_collections()

        logger.info("PDF Worker initialized")

    def process(self, message: Dict[str, Any]):
        """
        处理 PDF 消息

        Args:
            message: Kafka 消息 {"document_id": str, "file_path": str}
        """
        document_id = message.get("document_id")
        file_path = message.get("file_path")

        logger.info(f"Processing PDF: {document_id}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

        try:
            # 1. 解析 PDF
            output_dir = os.path.join(self.config.storage.parsed_dir, document_id)
            os.makedirs(output_dir, exist_ok=True)

            parse_result: ParseResult = self.pdf_parser.parse(file_path, output_dir)
            logger.info(f"Parsed PDF: {len(parse_result.images)} images, {len(parse_result.tables)} tables")

            # 2. 分块
            chunks = self._chunk_content(parse_result, document_id, output_dir)

            # 3. 向量化并存储
            self._store_chunks(chunks)

            logger.info(f"PDF processing completed: {document_id}")

        except Exception as e:
            logger.error(f"Failed to process PDF {document_id}: {e}")
            raise

    def _chunk_content(self, parse_result: ParseResult, document_id: str, output_dir: str) -> List[Dict]:
        """内容分块"""
        chunks = []

        # 按页面分块
        pages = parse_result.markdown.split("\n\n")
        for page_idx, page_content in enumerate(pages):
            if not page_content.strip():
                continue

            # 文本块
            chunk_id = str(uuid.uuid4())
            chunk = {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_type": ChunkType.TEXT.value,
                "content": page_content.strip(),
                "image_paths": [],
                "page_number": page_idx + 1,
                "metadata": {}
            }
            chunks.append(chunk)

        # 图片块
        for img_info in parse_result.images:
            chunk_id = str(uuid.uuid4())
            chunk = {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_type": ChunkType.IMAGE.value,
                "content": f"[图片] {os.path.basename(img_info['path'])}",
                "image_paths": [img_info["path"]],
                "page_number": img_info["page"],
                "metadata": {"bbox": img_info.get("bbox")}
            }
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def _store_chunks(self, chunks: List[Dict]):
        """存储到 Milvus"""
        # 分离文本块和图片块
        text_chunks = [c for c in chunks if c["chunk_type"] == ChunkType.TEXT.value]
        image_chunks = [c for c in chunks if c["chunk_type"] == ChunkType.IMAGE.value]

        # 存储文本块 (BGE 1024维)
        if text_chunks:
            text_vectors = self.bge.encode([c["content"] for c in text_chunks])
            text_data = [
                {
                    "id": c["id"],
                    "document_id": c["document_id"],
                    "chunk_type": c["chunk_type"],
                    "content": c["content"],
                    "page_number": c["page_number"],
                    "image_paths": json.dumps(c["image_paths"]),
                    "vector": text_vectors[i].tolist()
                }
                for i, c in enumerate(text_chunks)
            ]
            self.milvus.insert("mmrag_text", text_data)
            logger.info(f"Stored {len(text_chunks)} text chunks")

        # 存储图片块 (CLIP 512维)
        if image_chunks:
            from PIL import Image

            image_paths = [c["image_paths"][0] for c in image_chunks if c["image_paths"]]
            image_vectors = self.clip.encode_image([Image.open(p) for p in image_paths])

            image_data = [
                {
                    "id": c["id"],
                    "document_id": c["document_id"],
                    "chunk_type": c["chunk_type"],
                    "content": c["content"],
                    "page_number": c["page_number"],
                    "image_paths": json.dumps(c["image_paths"]),
                    "vector": image_vectors[i].tolist()
                }
                for i, c in enumerate(image_chunks) if c["image_paths"]
            ]
            self.milvus.insert("mmrag_image", image_data)
            logger.info(f"Stored {len(image_chunks)} image chunks")


def run_worker():
    """运行 Worker"""
    from app.services.mq import KafkaConsumerService

    worker = PDFWorker()
    worker.initialize()

    kafka_config = worker.config.kafka
    consumer = KafkaConsumerService(
        bootstrap_servers=kafka_config.bootstrap_servers,
        group_id=kafka_config.consumer_group,
        topics=[kafka_config.topic_pdf]
    )

    logger.info(f"Starting PDF Worker, consuming from {kafka_config.topic_pdf}")

    def callback(topic: str, message: Dict[str, Any]):
        worker.process(message)

    consumer.start_consuming(callback)


if __name__ == "__main__":
    run_worker()