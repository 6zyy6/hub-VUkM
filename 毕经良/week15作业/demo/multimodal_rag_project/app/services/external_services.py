import requests
import json
import random
from typing import List
from app.config import config
from app.models.schemas import ParseResult

class ExternalServices:
    @staticmethod
    def parse_document_mineru(file_path: str) -> ParseResult:
        """
        Mock calling MinerU / DeepSeek-OCR for document parsing.
        """
        try:
            # response = requests.post(config.MINERU_API_URL, files={"file": open(file_path, "rb")})
            # return ParseResult(**response.json())
            print(f"Mock calling MinerU for {file_path}")
            return ParseResult(
                markdown=f"# Parsed Content for {file_path}\n\nThis is a mock text chunk extracted by MinerU.",
                images=[f"{file_path}_mock_image_1.png", f"{file_path}_mock_image_2.png"]
            )
        except Exception as e:
            print(f"MinerU API Error: {e}")
            raise e

    @staticmethod
    def get_text_embedding(text: str) -> List[float]:
        """
        Mock calling BGE Text Embedding
        """
        try:
            # response = requests.post(config.BGE_EMBEDDING_API_URL, json={"text": text})
            # return response.json()["embedding"]
            print(f"Mock BGE embedding for text length {len(text)}")
            return [random.random() for _ in range(config.TEXT_VECTOR_DIM)]
        except Exception as e:
            print(f"BGE API Error: {e}")
            return [0.0] * config.TEXT_VECTOR_DIM

    @staticmethod
    def get_image_embedding(image_path: str) -> List[float]:
        """
        Mock calling CLIP Image Embedding
        """
        try:
            # response = requests.post(config.CLIP_EMBEDDING_API_URL, json={"image_path": image_path})
            # return response.json()["embedding"]
            print(f"Mock CLIP embedding for image {image_path}")
            return [random.random() for _ in range(config.IMAGE_VECTOR_DIM)]
        except Exception as e:
            print(f"CLIP API Error: {e}")
            return [0.0] * config.IMAGE_VECTOR_DIM

    @staticmethod
    def generate_answer_qwen_vl(query: str, context_texts: List[str], context_images: List[str]) -> str:
        """
        Mock calling Qwen-VL Multimodal LLM
        """
        try:
            # payload = {"query": query, "texts": context_texts, "images": context_images}
            # response = requests.post(config.QWEN_VL_API_URL, json=payload)
            # return response.json()["answer"]
            print(f"Mock Qwen-VL called with query: {query}")
            return f"Based on {len(context_texts)} texts and {len(context_images)} images, this is a generated multimodal answer from Qwen-VL."
        except Exception as e:
            print(f"Qwen-VL API Error: {e}")
            return "Failed to generate answer."

external_services = ExternalServices()
