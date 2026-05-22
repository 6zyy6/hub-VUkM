"""Qwen-VL 多模态 LLM 服务"""
from typing import List, Optional, Union, Dict
from PIL import Image
import logging
import torch

logger = logging.getLogger(__name__)


class QwenVLService:
    """Qwen-VL 多模态大语言模型服务"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._processor = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading Qwen-VL model: {self.model_name} on {self.device}")

        # 加载模型
        if self.load_in_8bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto"
            )

        # 加载处理器
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info("Qwen-VL model loaded successfully")

    def chat(
        self,
        query: str,
        context: Optional[str] = None,
        images: Optional[List[Union[str, Image.Image]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        多模态对话

        Args:
            query: 用户问题
            context: 检索到的上下文
            images: 图片列表（路径或 PIL Image）
            system_prompt: 系统提示
            max_tokens: 最大生成长度
            temperature: 采样温度

        Returns:
            生成的回复文本
        """
        self._load_model()

        # 构建消息
        messages = []

        # 系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            default_system = """你是一个多模态知识库助手。基于给定的上下文信息回答用户问题。
            如果上下文中有图片，请结合图片内容回答。
            如果上下文中没有相关信息，请如实说明。"""
            messages.append({"role": "system", "content": default_system})

        # 构建用户消息
        user_content = []

        # 添加上下文
        if context:
            user_content.append({
                "type": "text",
                "text": f"上下文信息:\n{context}\n\n用户问题: {query}"
            })
        else:
            user_content.append({
                "type": "text",
                "text": query
            })

        # 添加图片
        if images:
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img)
                user_content.append({
                    "type": "image",
                    "image": img
                })

        messages.append({"role": "user", "content": user_content})

        # 处理输入
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self._processor(
            text=[text],
            images=[images[0] if images and isinstance(images[0], Image.Image) else None] if images else [None],
            return_tensors="pt"
        ).to(self.device)

        # 生成
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        # 解码
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response

    def answer_with_sources(
        self,
        query: str,
        search_results: List[Dict],
        images: Optional[List[Union[str, Image.Image]]] = None
    ) -> tuple[str, List[Dict]]:
        """
        基于检索结果生成带来源的答案

        Args:
            query: 用户问题
            search_results: 检索结果列表
            images: 可选的额外图片

        Returns:
            (答案文本, 来源列表)
        """
        # 构建上下文
        context_parts = []
        for i, result in enumerate(search_results, 1):
            part = f"[来源 {i}]\n文档: {result.get('document_id', 'unknown')}\n内容: {result.get('content', '')}"
            if result.get('image_paths'):
                part += f"\n图片: {', '.join(result.get('image_paths', []))}"
            context_parts.append(part)

        context = "\n---\n".join(context_parts)

        # 添加来源到回答的提示
        system_prompt = """你是一个多模态知识库助手。基于给定的上下文信息回答用户问题。
        回答时适当引用来源编号，如"根据来源2..."。
        如果上下文中没有相关信息，请如实说明。"""

        answer = self.chat(query, context, images, system_prompt)

        # 构建来源列表
        sources = [
            {
                "document_id": r.get("document_id"),
                "content": r.get("content", "")[:200] + "...",
                "image_paths": r.get("image_paths", [])
            }
            for r in search_results
        ]

        return answer, sources