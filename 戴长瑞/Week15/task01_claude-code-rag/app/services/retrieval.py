"""混合检索服务 - CLIP + BGE 联合检索"""
from typing import List, Optional
import numpy as np
import logging

from app.models.data_models import SearchResult, ChunkType
from app.core.factory import get_service

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """混合检索服务 - 结合 CLIP 图搜图和 BGE 文本检索"""

    def __init__(
        self,
        bge_service=None,
        clip_service=None,
        milvus_service=None,
        text_weight: float = 0.6,
        image_weight: float = 0.4
    ):
        # 延迟加载服务
        self._bge = bge_service
        self._clip = clip_service
        self._milvus = milvus_service
        self.text_weight = text_weight
        self.image_weight = image_weight

    @property
    def bge(self):
        if self._bge is None:
            self._bge = get_service("embedding_bge")
        return self._bge

    @property
    def clip(self):
        if self._clip is None:
            self._clip = get_service("embedding_clip")
        return self._clip

    @property
    def milvus(self):
        if self._milvus is None:
            self._milvus = get_service("vector_db_milvus")
        return self._milvus

    def retrieve(
        self,
        query: str,
        query_images: Optional[List[str]] = None,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        混合检索入口

        Args:
            query: 文本查询
            query_images: 可选的查询图片（用于图搜图场景）
            top_k: 返回数量
            document_ids: 可选的文档ID过滤

        Returns:
            SearchResult 列表，按相关性排序
        """
        results = []

        # 1. 文本检索 (BGE)
        text_results = self._retrieve_text(query, top_k * 2, document_ids)
        results.extend(text_results)

        # 2. 图片检索 (CLIP) - 如果有查询图片或用户想找相似图片
        if query_images:
            image_results = self._retrieve_image(query_images, top_k * 2, document_ids)
            results.extend(image_results)

        # 3. RRF 融合排序
        fused_results = self._fusion_rrf(results, top_k)

        return fused_results[:top_k]

    def _retrieve_text(self, query: str, top_k: int, document_ids: Optional[List[str]]) -> List[SearchResult]:
        """BGE 文本检索"""
        logger.info(f"Text retrieval: {query[:50]}...")

        # 生成查询向量
        query_vector = self.bge.encode(query)
        query_vector = np.array(query_vector).reshape(1, -1)

        # 构建过滤表达式
        expr = None
        if document_ids:
            doc_filter = " || ".join([f'document_id == "{doc_id}"' for doc_id in document_ids])
            expr = f"({doc_filter})"

        # 搜索文本集合
        results = self.milvus.search("mmrag_text", query_vector, top_k=top_k, expr=expr)

        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    chunk_id=hit["id"],
                    document_id=hit["document_id"],
                    content=hit["content"],
                    image_paths=self._parse_image_paths(hit.get("image_paths", "")),
                    score=1.0 - hit["score"],  # L2 距离转相似度
                    chunk_type=ChunkType(hit.get("chunk_type", "text"))
                ))

        return search_results

    def _retrieve_image(self, image_paths: List[str], top_k: int, document_ids: Optional[List[str]]) -> List[SearchResult]:
        """CLIP 图片检索"""
        from PIL import Image

        logger.info(f"Image retrieval: {len(image_paths)} images")

        # 加载图片
        images = [Image.open(path) for path in image_paths]

        # 生成图片向量
        image_vectors = self.clip.encode_image(images)
        query_vector = np.mean(image_vectors, axis=0, keepdims=True)

        # 构建过滤表达式
        expr = None
        if document_ids:
            doc_filter = " || ".join([f'document_id == "{doc_id}"' for doc_id in document_ids])
            expr = f"({doc_filter})"

        # 搜索图片集合
        results = self.milvus.search("mmrag_image", query_vector, top_k=top_k, expr=expr)

        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    chunk_id=hit["id"],
                    document_id=hit["document_id"],
                    content=hit["content"],
                    image_paths=self._parse_image_paths(hit.get("image_paths", "")),
                    score=1.0 - hit["score"],
                    chunk_type=ChunkType.IMAGE
                ))

        return search_results

    def _fusion_rrf(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """RRF (Reciprocal Rank Fusion) 融合排序"""
        if not results:
            return []

        # 按 chunk_id 分组，取最高分
        chunk_scores = {}
        for result in results:
            if result.chunk_id not in chunk_scores or result.score > chunk_scores[result.chunk_id].score:
                chunk_scores[result.chunk_id] = result

        # 重新排序（按 score 降序）
        sorted_results = sorted(chunk_scores.values(), key=lambda x: x.score, reverse=True)

        return sorted_results[:top_k]

    def _parse_image_paths(self, image_paths_str: str) -> List[str]:
        """解析图片路径字符串（JSON格式）"""
        if not image_paths_str:
            return []
        try:
            import json
            return json.loads(image_paths_str)
        except:
            return []


def build_retrieval_context(results: List[SearchResult]) -> str:
    """构建检索上下文供 LLM 使用"""
    if not results:
        return "No relevant information found."

    context_parts = []
    for i, result in enumerate(results, 1):
        part = f"[来源 {i}]\n"
        part += f"文档: {result.document_id}\n"
        part += f"内容: {result.content}\n"
        if result.image_paths:
            part += f"图片: {', '.join(result.image_paths)}\n"
        context_parts.append(part)

    return "\n---\n".join(context_parts)