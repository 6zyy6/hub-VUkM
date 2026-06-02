from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from app.config import config

class MilvusService:
    def __init__(self):
        self.uri = config.MILVUS_URI
        self.text_collection_name = "text_vectors"
        self.image_collection_name = "image_vectors"
        
    def connect(self):
        try:
            connections.connect("default", uri=self.uri)
            self._init_collections()
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")

    def _init_collections(self):
        # Text Collection
        if not utility.has_collection(self.text_collection_name):
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.TEXT_VECTOR_DIM)
            ]
            schema = CollectionSchema(fields, "Text Vector Collection")
            collection = Collection(self.text_collection_name, schema)
            
            # Create Index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            
        # Image Collection
        if not utility.has_collection(self.image_collection_name):
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.IMAGE_VECTOR_DIM)
            ]
            schema = CollectionSchema(fields, "Image Vector Collection")
            collection = Collection(self.image_collection_name, schema)
            
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="vector", index_params=index_params)

    def insert_text(self, chunk_id: str, doc_id: str, kb_id: str, vector: list):
        collection = Collection(self.text_collection_name)
        data = [
            [chunk_id],
            [doc_id],
            [kb_id],
            [vector]
        ]
        collection.insert(data)
        collection.flush()

    def insert_image(self, chunk_id: str, doc_id: str, kb_id: str, vector: list):
        collection = Collection(self.image_collection_name)
        data = [
            [chunk_id],
            [doc_id],
            [kb_id],
            [vector]
        ]
        collection.insert(data)
        collection.flush()

    def search_text(self, query_vector: list, kb_id: str, top_k: int = 5):
        collection = Collection(self.text_collection_name)
        collection.load()
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=f"kb_id == '{kb_id}'",
            output_fields=["chunk_id", "doc_id"]
        )
        return results

    def search_image(self, query_vector: list, kb_id: str, top_k: int = 5):
        collection = Collection(self.image_collection_name)
        collection.load()
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=f"kb_id == '{kb_id}'",
            output_fields=["chunk_id", "doc_id"]
        )
        return results

milvus_client = MilvusService()
