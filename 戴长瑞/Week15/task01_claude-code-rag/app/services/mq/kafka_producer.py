"""Kafka 消息队列服务"""
import json
import logging
from typing import Callable, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """支持 datetime 序列化的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class KafkaProducerService:
    """Kafka 生产者服务"""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer: Optional[KafkaProducer] = None

    def _get_producer(self) -> KafkaProducer:
        """延迟初始化生产者"""
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, cls=DateTimeEncoder).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3
            )
            logger.info(f"Kafka producer connected to {self.bootstrap_servers}")
        return self._producer

    def send_message(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        """发送消息"""
        producer = self._get_producer()
        future = producer.send(topic, value=message, key=key)
        try:
            record_metadata = future.get(timeout=10)
            logger.info(f"Message sent to {record_metadata.topic}:{record_metadata.partition}")
            return record_metadata
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            raise

    def send_pdf_process_message(self, document_id: str, file_path: str):
        """发送 PDF 处理消息"""
        message = {
            "document_id": document_id,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat()
        }
        self.send_message("pdf-processing", message, key=document_id)

    def send_image_process_message(self, document_id: str, chunk_id: str, image_path: str):
        """发送图片处理消息"""
        message = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat()
        }
        self.send_message("image-processing", message, key=chunk_id)

    def close(self):
        """关闭生产者"""
        if self._producer:
            self._producer.close()
            self._producer = None
            logger.info("Kafka producer closed")


class KafkaConsumerService:
    """Kafka 消费者服务"""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "mmrag-workers",
        topics: Optional[list] = None
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or ["pdf-processing", "image-processing"]
        self._consumer: Optional[KafkaConsumer] = None

    def _get_consumer(self) -> KafkaConsumer:
        """延迟初始化消费者"""
        if self._consumer is None:
            self._consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                auto_offset_reset="earliest",
                enable_auto_commit=True
            )
            logger.info(f"Kafka consumer connected to {self.bootstrap_servers}, group={self.group_id}")
        return self._consumer

    def consume(self, callback: Callable[[str, Dict[str, Any]], None], timeout_ms: int = 1000):
        """消费消息"""
        consumer = self._get_consumer()

        for message in consumer.poll(timeout_ms=timeout_ms):
            topic = message.topic
            key = message.key
            value = message.value

            try:
                callback(topic, value)
            except Exception as e:
                logger.error(f"Error processing message from {topic}: {e}")

    def start_consuming(self, callback: Callable[[str, Dict[str, Any]], None]):
        """持续消费"""
        consumer = self._get_consumer()
        logger.info(f"Starting to consume from topics: {self.topics}")

        try:
            for message in consumer:
                topic = message.topic
                value = message.value
                logger.debug(f"Received message from {topic}: {value.get('document_id', 'unknown')}")

                try:
                    callback(topic, value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except KeyboardInterrupt:
            logger.info("Consumer stopped")
        finally:
            self.close()

    def close(self):
        """关闭消费者"""
        if self._consumer:
            self._consumer.close()
            self._consumer = None
            logger.info("Kafka consumer closed")