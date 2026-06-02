"""
Vector storage service using Milvus for multimodal embeddings.
Stores and retrieves text chunks and image vectors.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

from app.core.config import settings


@dataclass
class SearchResult:
    """Search result from vector database."""
    content: str  # Text chunk or image path
    content_type: str  # "text" or "image"
    score: float
    document_id: str
    metadata: Dict[str, Any]


class VectorStore:
    """Handles vector storage and retrieval in Milvus."""

    def __init__(self):
        self.connections = connections
        self.collection: Optional[Collection] = None
        self._connected = False

    def connect(self):
        """Connect to Milvus server."""
        if self._connected:
            return

        self.connections.connect(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            alias="default"
        )
        self._connected = True

    def disconnect(self):
        """Disconnect from Milvus server."""
        if self._connected:
            self.connections.disconnect(alias="default")
            self._connected = False

    def create_collection(self, if_not_exists: bool = True):
        """
        Create collection for storing text and image embeddings.

        Schema:
        - text_chunk_id: Primary key
        - document_id: Reference to document
        - content_type: "text" or "image"
        - content: Text content or image path
        - vector: Embedding vector (1024 dim for text, 512 dim for image)
        - page_num: Page number in original PDF
        """
        if not self._connected:
            self.connect()

        collection_name = settings.MILVUS_COLLECTION

        if utility.has_collection(collection_name):
            if if_not_exists:
                self.collection = Collection(collection_name)
                return
            else:
                utility.drop_collection(collection_name)

        # Schema definition
        fields = [
            FieldSchema(name="text_chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),  # Unified dim
            FieldSchema(name="page_num", dtype=DataType.INT32),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
        ]

        schema = CollectionSchema(fields=fields, description="PDF Knowledge Base Multimodal Vectors")
        self.collection = Collection(name=collection_name, schema=schema)

        # Create indexes
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }

        self.collection.create_index(field_name="vector", index_params=index_params)
        self.collection.create_index(field_name="document_id")

    def insert_text_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Insert text chunks with their embeddings.

        Args:
            chunks: List of chunk dictionaries with text, document_id, page_num, chunk_index
            embeddings: Corresponding embedding vectors
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection first.")

        entities = [
            [chunk["document_id"] for chunk in chunks],
            ["text"] * len(chunks),
            [chunk["text"][:65535] for chunk in chunks],  # Truncate if needed
            embeddings,
            [chunk.get("page_num", 0) for chunk in chunks],
            [chunk["chunk_index"] for chunk in chunks],
        ]

        self.collection.insert(entities)
        self.collection.flush()

    def insert_images(self, images: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Insert image metadata with their embeddings.

        Args:
            images: List of image dictionaries with path, document_id, page_num, description
            embeddings: Corresponding embedding vectors
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized. Call create_collection first.")

        entities = [
            [img["document_id"] for img in images],
            ["image"] * len(images),
            [img.get("description", img["path"])[:65535] for img in images],
            embeddings,
            [img.get("page_num", 0) for img in images],
            [-1] * len(images),  # chunk_index = -1 for images
        ]

        self.collection.insert(entities)
        self.collection.flush()

    def search_text(self, query_vector: List[float], top_k: int = 5,
                    document_id: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar text chunks.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            document_id: Optional filter by document

        Returns:
            List of SearchResult objects
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized.")

        # Build search params
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # Filter by content_type = "text"
        expr = 'content_type == "text"'
        if document_id:
            expr = f'{expr} and document_id == "{document_id}"'

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["document_id", "content_type", "content", "page_num", "chunk_index"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    content=hit.entity.get("content", ""),
                    content_type="text",
                    score=hit.distance,
                    document_id=hit.entity.get("document_id", ""),
                    metadata={
                        "page_num": hit.entity.get("page_num", 0),
                        "chunk_index": hit.entity.get("chunk_index", 0)
                    }
                ))

        return search_results

    def search_image(self, query_vector: List[float], top_k: int = 5,
                     document_id: Optional[str] = None) -> List[SearchResult]:
        """
        Search for similar images.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            document_id: Optional filter by document

        Returns:
            List of SearchResult objects
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized.")

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # Filter by content_type = "image"
        expr = 'content_type == "image"'
        if document_id:
            expr = f'{expr} and document_id == "{document_id}"'

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["document_id", "content_type", "content", "page_num"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    content=hit.entity.get("content", ""),
                    content_type="image",
                    score=1.0 / (1.0 + hit.distance),  # Convert distance to similarity
                    document_id=hit.entity.get("document_id", ""),
                    metadata={"page_num": hit.entity.get("page_num", 0)}
                ))

        return search_results

    def search_hybrid(self, text_query_vector: List[float], image_query_vector: List[float],
                      top_k: int = 5, document_id: Optional[str] = None) -> List[SearchResult]:
        """
        Search both text and images, merge results.

        Args:
            text_query_vector: Text query embedding
            image_query_vector: Image query embedding
            top_k: Number of results to return for each type
            document_id: Optional filter by document

        Returns:
            Merged and ranked list of SearchResults
        """
        text_results = self.search_text(text_query_vector, top_k, document_id)
        image_results = self.search_image(image_query_vector, top_k, document_id)

        # Merge and sort by score
        all_results = text_results + image_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:top_k]

    def delete_by_document_id(self, document_id: str):
        """
        Delete all vectors associated with a document.

        Args:
            document_id: Document ID to delete
        """
        if not self.collection:
            raise RuntimeError("Collection not initialized.")

        expr = f'document_id == "{document_id}"'
        self.collection.delete(expr)
        self.collection.flush()