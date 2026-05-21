"""
检索和问答服务 - 处理向量检索和答案生成
"""
import os
import re
from typing import List, Dict
from pymilvus import MilvusClient
import openai

from config import (
    MILVUS_URI, MILVUS_TOKEN, COLLECTION_NAME, SEARCH_LIMIT,
    QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
)


class RetrievalService:
    """检索服务类"""

    def __init__(self):
        self.milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

    def search_relevant_texts(
            self,
            query_embedding: List[float],
            limit: int = None
    ) -> List[Dict]:
        """
        在Milvus中检索相关文本

        Args:
            query_embedding: 查询向量
            limit: 返回结果数量

        Returns:
            List[Dict]: 检索结果列表
        """
        if limit is None:
            limit = SEARCH_LIMIT

        results = self.milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=limit,
            anns_field="text_vector",
            output_fields=["text", "db_id", "file_name", "file_path"]
        )

        # 格式化结果
        formatted_results = []
        if results and len(results) > 0:
            for item in results[0]:
                formatted_results.append({
                    "text": item["entity"]["text"],
                    "db_id": item["entity"]["db_id"],
                    "file_name": item["entity"]["file_name"],
                    "file_path": item["entity"]["file_path"],
                    "score": item["distance"]
                })

        return formatted_results

    def format_context_for_qwen(self, results: List[Dict]) -> str:
        """
        将检索结果格式化为Qwen可用的上下文

        Args:
            results: 检索结果列表

        Returns:
            str: 格式化后的上下文字符串
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            # 修正图片路径
            file_dir = os.path.basename(result["file_path"]).split(".")[0]
            corrected_text = result["text"].replace(
                "images/",
                f"./processed/{file_dir}/vlm/images/"
            )

            # 添加来源标注
            source_info = f"\n--- 来源: {result['file_name']} ---\n"
            context_parts.append(source_info + corrected_text)

        return "\n\n".join(context_parts)


class ChatService:
    """问答服务类"""

    def __init__(self):
        self.qwen_client = openai.OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
        )
        self.retrieval_service = RetrievalService()

    def generate_answer(
            self,
            question: str,
            context: str,
            model: str = None
    ) -> str:
        """
        基于上下文生成答案

        Args:
            question: 用户问题
            context: 检索到的相关上下文
            model: 使用的模型名称

        Returns:
            str: 生成的答案
        """
        if model is None:
            model = QWEN_MODEL

        prompt = self._build_prompt(question, context)

        completion = self.qwen_client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': '你是一个专业的多模态AI助手，能够基于提供的资料回答问题。'},
                {'role': 'user', 'content': prompt}
            ],
        )

        return completion.choices[0].message.content

    def _build_prompt(self, question: str, context: str) -> str:
        """
        构建提示词

        Args:
            question: 用户问题
            context: 相关上下文

        Returns:
            str: 完整的提示词
        """
        prompt_template = """基于以下资料回答问题：{question}

相关资料：
{context}

回答要求：
1. 回答要客观、有逻辑，严格基于提供的资料
2. 如果资料中包含图片链接（![](xxx)），请单独一行输出，保留图片的原始链接
3. 将图片放在相关内容的合适位置
4. 在答案末尾注明信息来源（文件名）

请开始回答：
"""
        return prompt_template.format(question=question, context=context)

    @staticmethod
    def render_markdown_with_images(markdown_text: str) -> List[Dict]:
        """
        解析Markdown文本，分离文本和图片

        Args:
            markdown_text: Markdown文本

        Returns:
            List[Dict]: 包含文本和图片的结构化列表
        """
        pattern = re.compile(r'!\[.*?\]\((.*?)\)')
        parts = []
        last_pos = 0

        for match in pattern.finditer(markdown_text):
            # 添加前面的文本
            if match.start() > last_pos:
                text_content = markdown_text[last_pos:match.start()]
                if text_content.strip():
                    parts.append({'type': 'text', 'content': text_content})

            # 添加图片
            img_url = match.group(1)
            parts.append({'type': 'image', 'url': img_url})

            last_pos = match.end()

        # 添加剩余文本
        if last_pos < len(markdown_text):
            remaining_text = markdown_text[last_pos:]
            if remaining_text.strip():
                parts.append({'type': 'text', 'content': remaining_text})

        return parts
