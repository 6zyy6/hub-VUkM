"""Embedding 服务 - CLIP / BGE"""
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """Embedding 服务基类"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码文本"""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass


class CLIPEmbedding(BaseEmbeddingService):
    """CLIP Embedding - 用于图搜图和多模态检索"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import CLIPProcessor, CLIPModel
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, inputs: Union[str, List[str], Image.Image, List[Image.Image]], **kwargs) -> np.ndarray:
        """编码输入（文本或图像）"""
        is_image = isinstance(inputs, (Image.Image, list)) and isinstance(
            inputs[0] if isinstance(inputs, list) else inputs, Image.Image
        ) or (isinstance(inputs, Image.Image))

        if is_image:
            return self.encode_image(inputs)
        return self.encode_text(inputs)

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        return outputs.cpu().numpy()

    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """编码图像"""
        if isinstance(images, Image.Image):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        return outputs.cpu().numpy()

    def dimension(self) -> int:
        return 512  # CLIP ViT-B/32


class BGEEmbedding(BaseEmbeddingService):
    """BGE Embedding - 用于文本检索"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        from transformers import AutoTokenizer, AutoModel
        logger.info(f"Loading BGE model: {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码文本"""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 使用 [CLS] 向量
        embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()

        # L2 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def dimension(self) -> int:
        return 1024  # BGE-M3