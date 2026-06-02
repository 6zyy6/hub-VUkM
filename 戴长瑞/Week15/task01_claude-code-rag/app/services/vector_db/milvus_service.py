"""Milvus 向量数据库服务"""
from typing import List, Optional, Tuple
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import logging

logger = logging.getLogger(__name__)


class MilvusService:
    """Milvus 向量数据库服务"""

    def __init__(self, host: str = "localhost", port: int = 19530, alias: str = "default"):
        self.host = host
        self.port = port
        self.alias = alias
        self._connected = False
        self._collections = {}

    def connect(self):
        """连接 Milvus"""
        if not self._connected:
            connections.connect(host=self.host, port=self.port, alias=self.alias)
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")

    def disconnect(self):
        """断开连接"""
        if self._connected:
            connections.disconnect(alias=self.alias)
            self._connected = False
            logger.info("Disconnected from Milvus")

    def initialize_collections(self, recreate: bool = False):
        """初始化集合"""
        self.connect()

        # 文本集合 (BGE 1024维)
        self._create_collection(
            "mmrag_text",
            dimension=1024,
            description="Text chunks collection",
            recreate=recreate
        )

        # 图片集合 (CLIP 512维)
        self._create_collection(
            "mmrag_image",
            dimension=512,
            description="Image chunks collection",
            recreate=recreate
        )

        logger.info("Milvus collections initialized")

    def _create_collection(self, name: str, dimension: int, description: str = "", recreate: bool = False):
        """创建集合"""
        if utility.has_collection(name):
            if recreate:
                utility.drop_collection(name)
                logger.info(f"Dropped collection: {name}")
            else:
                self._collections[name] = Collection(name)
                return

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="page_number", dtype=DataType.INT32),
            FieldSchema(name="image_paths", dtype=DataType.VARCHAR, max_length=2048),  # JSON 存储
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]

        schema = CollectionSchema(fields=fields, description=description)
        collection = Collection(name=name, schema=schema)

        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="vector", index_params=index_params)

        collection.load()
        self._collections[name] = collection
        logger.info(f"Created collection: {name}, dim={dimension}")

    def insert(self, collection_name: str, data: List[dict]):
        """插入数据"""
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self._collections[collection_name]
        collection.insert(data)
        collection.flush()
        logger.info(f"Inserted {len(data)} records to {collection_name}")

    def search(
        self,
        collection_name: str,
        query_vectors: np.ndarray,
        top_k: int = 5,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[List[dict]]:
        """向量检索"""
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self._collections[collection_name]

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        if output_fields is None:
            output_fields = ["id", "document_id", "chunk_type", "content", "page_number", "image_paths"]

        results = collection.search(
            data=query_vectors.tolist(),
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            hits_list = []
            for hit in hits:
                hit_dict = {field: hit.entity.get(field) for field in output_fields}
                hit_dict["score"] = hit.distance
                hits_list.append(hit_dict)
            formatted_results.append(hits_list)

        return formatted_results

    def delete_by_document_id(self, document_id: str, collection_name: str):
        """按文档ID删除"""
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self._collections[collection_name]
        expr = f'document_id == "{document_id}"'
        collection.delete(expr)
        collection.flush()
        logger.info(f"Deleted documents with document_id={document_id}")

    def query(self, collection_name: str, expr: str, output_fields: Optional[List[str]] = None) -> List[dict]:
        """标量查询"""
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self._collections[collection_name]

        if output_fields is None:
            output_fields = ["id", "document_id", "chunk_type", "content", "page_number", "image_paths"]

        results = collection.query(expr=expr, output_fields=output_fields)
        return results

    def get_collection(self, name: str) -> Optional[Collection]:
        """获取集合"""
        return self._collections.get(name)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()