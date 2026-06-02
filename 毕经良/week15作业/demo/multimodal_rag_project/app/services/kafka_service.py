import json
from confluent_kafka import Producer, Consumer, KafkaException
from app.config import config
from app.models.schemas import ParseMessage

class KafkaService:
    def __init__(self):
        self.producer = Producer({'bootstrap.servers': config.KAFKA_BOOTSTRAP_SERVERS})
        self.topic = config.KAFKA_PARSE_TOPIC

    def delivery_report(self, err, msg):
        if err is not None:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def send_parse_task(self, task: ParseMessage):
        data = task.model_dump_json()
        self.producer.produce(self.topic, data.encode('utf-8'), callback=self.delivery_report)
        self.producer.poll(0)

    def get_consumer(self, group_id="parse_worker_group"):
        consumer = Consumer({
            'bootstrap.servers': config.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        consumer.subscribe([self.topic])
        return consumer

kafka_service = KafkaService()
