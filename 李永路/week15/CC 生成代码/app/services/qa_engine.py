"""
Multimodal QA service using Qwen-VL for answering questions
based on retrieved text and image content.
"""

import os
import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.core.config import settings
from app.services.vector_store import SearchResult


@dataclass
class QAResponse:
    """Response from QA system."""
    answer: str
    sources: List[Dict[str, Any]]  # [{content, type, page_num, document_id}]
    score: float


class MultimodalQA:
    """Handles multimodal question answering with Qwen-VL."""

    def __init__(self):
        self.api_key = settings.QWEN_API_KEY or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = settings.QWEN_API_BASE

    def answer(self, question: str, retrieved_content: List[SearchResult],
               knowledge_base_id: Optional[str] = None) -> QAResponse:
        """
        Generate answer based on retrieved content.

        Args:
            question: User's question
            retrieved_content: List of SearchResults from vector search
            knowledge_base_id: Optional knowledge base identifier

        Returns:
            QAResponse with answer and sources
        """
        # Organize content for Qwen-VL
        text_contents = []
        image_contents = []
        sources = []

        for result in retrieved_content:
            if result.content_type == "text":
                text_contents.append(result.content)
                sources.append({
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "type": "text",
                    "page_num": result.metadata.get("page_num", 0),
                    "document_id": result.document_id
                })
            elif result.content_type == "image":
                image_contents.append(result.content)
                sources.append({
                    "content": result.content,
                    "type": "image",
                    "page_num": result.metadata.get("page_num", 0),
                    "document_id": result.document_id
                })

        # Build prompt for Qwen-VL
        prompt = self._build_prompt(question, text_contents, image_contents)

        # Call Qwen-VL API
        answer = self._call_qwen_vl(prompt, image_contents)

        return QAResponse(
            answer=answer,
            sources=sources,
            score=1.0  # Placeholder - could be computed from retrieval scores
        )

    def _build_prompt(self, question: str, text_contents: List[str],
                      image_contents: List[str]) -> str:
        """Build prompt for Qwen-VL with retrieved content."""
        prompt_parts = [
            f"你是一个专业的知识库问答助手。请根据以下检索到的信息回答用户的问题。\n\n",
            f"用户问题: {question}\n\n",
            "检索到的文本内容:\n"
        ]

        for i, text in enumerate(text_contents[:5]):  # Limit to 5 text chunks
            prompt_parts.append(f"[文本{i+1}]\n{text[:500]}...\n\n")

        if image_contents:
            prompt_parts.append(f"\n检索到的图片数量: {len(image_contents)}\n")
            prompt_parts.append("请结合以上所有信息，包括图片内容，进行分析和回答。\n")

        prompt_parts.append("\n请给出准确、简洁的回答，并指出信息来源。")

        return "".join(prompt_parts)

    def _call_qwen_vl(self, prompt: str, image_paths: List[str]) -> str:
        """
        Call Qwen-VL API for multimodal reasoning.

        Args:
            prompt: Text prompt
            image_paths: List of image file paths to include

        Returns:
            Generated answer
        """
        if not self.api_key:
            return "API key not configured. Please set QWEN_API_KEY or DASHSCOPE_API_KEY."

        try:
            # Build messages for Qwen-VL
            messages = [{"role": "user", "content": []}]

            # Add images
            for img_path in image_paths[:10]:  # Limit to 10 images
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    # In practice, would need to upload to a URL or use base64
                    # For now, add as image_url reference
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{img_path}"}
                    })

            # Add text prompt
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })

            # Call API
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen-vl-max",  # or qwen-vl-ocr for single-turn
                    "messages": messages,
                    "max_tokens": 2048
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling Qwen-VL: {str(e)}"

    def answer_with_images(self, question: str, image_urls: List[str],
                           context_text: Optional[str] = None) -> str:
        """
        Simple image QA without retrieval - for testing.

        Args:
            question: User question
            image_urls: List of image URLs
            context_text: Optional additional context

        Returns:
            Generated answer
        """
        if not self.api_key:
            return "API key not configured."

        try:
            messages = [{"role": "user", "content": []}]

            # Add images
            for url in image_urls:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })

            # Add question
            text_prompt = question
            if context_text:
                text_prompt = f"Context: {context_text}\n\nQuestion: {question}"

            messages[0]["content"].append({
                "type": "text",
                "text": text_prompt
            })

            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "qwen-vl-max",
                    "messages": messages,
                    "max_tokens": 2048
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API error: {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"