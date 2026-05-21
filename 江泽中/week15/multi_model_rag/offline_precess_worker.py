import os
import logging
from kafka import KafkaConsumer
from pymilvus import MilvusClient

from config import (
    KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC,
    MILVUS_URI, MILVUS_TOKEN, COLLECTION_NAME,
    BATCH_INSERT_SIZE
)
from services import DocumentService, FileService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化服务
document_service = DocumentService()
milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
)


def main():
    logger.info("Worker启动，开始消费Kafka消息...")
    for msg in consumer:
        try:
            logger.info(f"收到消息: {msg.value}")
            file_name = msg.value['file_name']
            file_path = msg.value['file_path']
            file_id = msg.value['id']

            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                FileService.update_file_state(file_id, "失败", "文件不存在")
                continue

            # 更新状态为解析中
            FileService.update_file_state(file_id, "解析中")

            # 使用文档服务解析PDF
            logger.info(f"开始解析PDF: {file_name}")
            markdown_path = document_service.parse_pdf_with_mineru(file_path)

            # 处理文档（分块+编码）
            logger.info(f"开始编码文档: {markdown_path}")
            data_list = document_service.process_document(
                markdown_path=markdown_path,
                file_id=file_id,
                file_name=file_name,
                file_path=file_path
            )

            # 批量插入Milvus
            success_count = 0
            for i in range(0, len(data_list), BATCH_INSERT_SIZE):
                batch = data_list[i:i+BATCH_INSERT_SIZE]
                milvus_client.insert(
                    collection_name=COLLECTION_NAME,
                    data=batch
                )
                success_count += len(batch)

            # 更新状态为完成
            FileService.update_file_state(file_id, "已完成")
            logger.info(f"文档处理完成: {file_name}, 插入 {success_count} 个chunk")

        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            traceback.print_exc()
            # 更新状态为失败
            if 'file_id' in locals():
                FileService.update_file_state(file_id, "失败", str(e))

if __name__ == "__main__":
    import json
    import traceback
    main()
