"""图片处理 Worker - 向量化图片"""
import os
import json
import logging
from typing import Dict, Any

from app.core.factory import ServiceFactory, initialize_factories
from app.core.config import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ImageWorker:
    """图片处理 Worker"""

    def __init__(self):
        self.config = get_config()
        self.clip = None
        self.milvus = None

    def initialize(self):
        """初始化服务"""
        initialize_factories()

        self.clip = ServiceFactory.get("embedding_clip")
        self.milvus = ServiceFactory.get("vector_db_milvus")
        self.milvus.initialize_collections()

        logger.info("Image Worker initialized")

    def process(self, message: Dict[str, Any]):
        """
        处理图片消息

        Args:
            message: Kafka 消息 {"document_id": str, "chunk_id": str, "image_path": str}
        """
        document_id = message.get("document_id")
        chunk_id = message.get("chunk_id")
        image_path = message.get("image_path")

        logger.info(f"Processing image: {image_path}")

        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return

        try:
            from PIL import Image

            # 向量化
            image = Image.open(image_path)
            vector = self.clip.encode_image(image)

            # 存储
            data = [{
                "id": chunk_id,
                "document_id": document_id,
                "chunk_type": "image",
                "content": f"[图片] {os.path.basename(image_path)}",
                "page_number": 0,
                "image_paths": json.dumps([image_path]),
                "vector": vector[0].tolist()
            }]
            self.milvus.insert("mmrag_image", data)

            logger.info(f"Image processed: {image_path}")

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise


def run_worker():
    """运行 Worker"""
    from app.services.mq import KafkaConsumerService

    worker = ImageWorker()
    worker.initialize()

    kafka_config = worker.config.kafka
    consumer = KafkaConsumerService(
        bootstrap_servers=kafka_config.bootstrap_servers,
        group_id=f"{kafka_config.consumer_group}-images",
        topics=[kafka_config.topic_image]
    )

    logger.info(f"Starting Image Worker, consuming from {kafka_config.topic_image}")

    def callback(topic: str, message: Dict[str, Any]):
        worker.process(message)

    consumer.start_consuming(callback)


if __name__ == "__main__":
    run_worker()