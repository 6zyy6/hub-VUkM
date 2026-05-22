"""
Kafka工具函数
"""
import json
from kafka import KafkaProducer
from typing import Dict

from config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC


def send_parse_task_to_kafka(file_name: str, file_path: str, file_id: int):
    """
    发送文档解析任务到Kafka

    Args:
        file_name: 文件名
        file_path: 文件路径
        file_id: 文件ID
    """
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    message = {
        "file_name": file_name,
        "file_path": file_path,
        "id": file_id
    }

    producer.send(KAFKA_TOPIC, value=message)
    producer.flush()
    producer.close()

    print(f"已发送解析任务到Kafka: {message}")
