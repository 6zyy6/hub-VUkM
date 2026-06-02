"""
Embedding service for text and image vectors.
Supports BGE for text and CLIP for image embeddings.
"""

import os
from typing import List, Dict, Any, Optional
import httpx

from app.core.config import settings


class EmbeddingService:
    """Handles text and image embedding generation."""

    def __init__(self):
        self.text_model = None
        self.image_model = None
        self._initialized = False

    def initialize(self):
        """Initialize embedding models."""
        if self._initialized:
            return

        # Text embedding with BGE via API
        self._init_text_embedding()
        # Image embedding with CLIP
        self._init_image_embedding()

        self._initialized = True

    def _init_text_embedding(self):
        """Initialize text embedding model."""
        # For BGE, we can use HuggingFace Inference API or local deployment
        # Using Jina API as mentioned in the project docs
        pass

    def _init_image_embedding(self):
        """Initialize image embedding model."""
        # CLIP model initialization
        pass

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Generate text embeddings.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            self.initialize()

        # Use Jina API for text embedding
        try:
            response = httpx.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "jina-embeddings-v4",
                    "input": texts
                },
                timeout=30
            )
            if response.status_code == 200:
                return [item["embedding"] for item in response.json()["data"]]
        except Exception as e:
            print(f"Text embedding failed: {e}")

        # Fallback: return zero vectors
        return [[0.0] * 1024 for _ in texts]

    def embed_image(self, image_paths: List[str]) -> List[List[float]]:
        """
        Generate image embeddings using CLIP.

        Args:
            image_paths: List of image file paths

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            self.initialize()

        # Use Jina CLIP API for image embedding
        embeddings = []
        for img_path in image_paths:
            try:
                with open(img_path, "rb") as f:
                    files = {"file": f}
                    data = {"model": "jina-clip-v2"}
                    response = httpx.post(
                        "https://api.jina.ai/v1/embeddings",
                        headers={"Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}"},
                        files=files,
                        data=data,
                        timeout=30
                    )
                if response.status_code == 200:
                    embeddings.append(response.json()["data"][0]["embedding"])
                else:
                    embeddings.append([0.0] * 512)
            except Exception as e:
                print(f"Image embedding failed for {img_path}: {e}")
                embeddings.append([0.0] * 512)

        return embeddings

    def embed_image_from_url(self, image_urls: List[str]) -> List[List[float]]:
        """
        Generate embeddings for images from URLs.

        Args:
            image_urls: List of image URLs

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for url in image_urls:
            try:
                response = httpx.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "jina-clip-v2",
                        "input": [{"image_url": url}]
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["data"][0]["embedding"])
                else:
                    embeddings.append([0.0] * 512)
            except Exception as e:
                print(f"Image embedding failed for {url}: {e}")
                embeddings.append([0.0] * 512)

        return embeddings