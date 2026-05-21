"""
文档解析服务 - 处理PDF解析、文本分块、向量编码等
"""
import os
import glob
import subprocess
import traceback
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    BGE_MODEL_PATH, CLIP_MODEL_PATH, CHUNK_SIZE,
    MINERU_BASE_URL, MINERU_BACKEND, MINERU_TIMEOUT,
    PROCESSED_DIR, BGE_DIMENSION, CLIP_DIMENSION
)


class ModelManager:
    """模型管理器 - 单例模式，避免重复加载"""
    _instance = None
    _bge_model = None
    _clip_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_bge_model(self):
        """获取BGE模型（懒加载）"""
        if self._bge_model is None:
            print(f"正在加载BGE模型: {BGE_MODEL_PATH}")
            self._bge_model = SentenceTransformer(BGE_MODEL_PATH)
            print("BGE模型加载完成")
        return self._bge_model

    def get_clip_model(self):
        """获取CLIP模型（懒加载）"""
        if self._clip_model is None:
            print(f"正在加载CLIP模型: {CLIP_MODEL_PATH}")
            self._clip_model = SentenceTransformer(
                CLIP_MODEL_PATH,
                trust_remote_code=True,
                truncate_dim=CLIP_DIMENSION
            )
            print("CLIP模型加载完成")
        return self._clip_model


class DocumentService:
    """文档解析服务类"""

    def __init__(self):
        self.model_manager = ModelManager()

    def split_text2chunks(self, lines: List[str], chunk_size: int = None) -> List[str]:
        """
        将文本分割成多个块，每个块的长度不超过chunk_size个字符

        Args:
            lines: 文本行列表
            chunk_size: 每块最大字符数

        Returns:
            List[str]: 分块后的文本列表
        """
        if chunk_size is None:
            chunk_size = CHUNK_SIZE

        chunks = []
        for line in lines:
            line = line.strip()

            # 过滤空行
            if not line:
                continue

            # 过滤参考文献标记
            if line == "# References":
                continue

            # 过滤引用编号（如 [1], [2] 等）
            if len(line) > 2 and line[0] == "[" and line[1].isdigit():
                continue

            # 合并到现有块或创建新块
            if len(chunks) == 0:
                chunks.append(line)
            else:
                if len(chunks[-1]) + len(line) + 1 <= chunk_size:
                    chunks[-1] += "\n" + line
                else:
                    chunks.append(line)

        return chunks

    def encode_text_and_image(
            self,
            text: str,
            markdown_path: str
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        将文本和图片编码成向量

        Args:
            text: Markdown文本内容
            markdown_path: Markdown文件路径

        Returns:
            (text_bge_embedding, text_clip_embedding, image_clip_embedding)
        """
        bge_model = self.model_manager.get_bge_model()
        clip_model = self.model_manager.get_clip_model()

        # 分离纯文本和带图片的行
        text_lines = [line for line in text.split("\n") if not line.startswith("![")]
        text_with_no_image = "\n".join(text_lines)
        text_with_image = [line for line in text.split("\n") if line.startswith("![")]

        # BGE文本编码
        try:
            text_bge_embedding = bge_model.encode(
                text_with_no_image,
                normalize_embeddings=True
            )
            text_bge_embedding = text_bge_embedding.tolist()
        except Exception as e:
            print(f"BGE编码失败: {e}")
            traceback.print_exc()
            text_bge_embedding = [0.0] * BGE_DIMENSION

        # CLIP文本编码
        try:
            text_clip_embedding = clip_model.encode(
                text_with_no_image,
                normalize_embeddings=True
            )
            text_clip_embedding = text_clip_embedding.tolist()
        except Exception as e:
            print(f"CLIP文本编码失败: {e}")
            traceback.print_exc()
            text_clip_embedding = [0.0] * CLIP_DIMENSION

        # CLIP图片编码
        image_clip_embedding = [0.0] * CLIP_DIMENSION
        if len(text_with_image) > 0:
            try:
                # 提取图片路径
                image_rel_path = text_with_image[0].split("](")[1].split(")")[0]
                image_real_path = os.path.join(
                    os.path.dirname(markdown_path),
                    image_rel_path.split("/")[-1]
                )

                if os.path.exists(image_real_path):
                    print(f"正在编码图片: {image_real_path}")
                    image_clip_embedding = clip_model.encode(
                        image_real_path,
                        normalize_embeddings=True
                    )
                    image_clip_embedding = image_clip_embedding.tolist()
                else:
                    print(f"图片文件不存在: {image_real_path}")
            except Exception as e:
                print(f"CLIP图片编码失败: {e}")
                traceback.print_exc()

        return text_bge_embedding, text_clip_embedding, image_clip_embedding

    def parse_pdf_with_mineru(self, pdf_path: str, output_dir: str = None) -> str:
        """
        使用MinerU解析PDF文件

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录

        Returns:
            str: Markdown文件路径
        """
        if output_dir is None:
            output_dir = str(PROCESSED_DIR)

        try:
            cmd = (
                f"mineru -p {pdf_path} "
                f"-o {output_dir} "
                f"-b {MINERU_BACKEND} "
                f"-u {MINERU_BASE_URL}"
            )
            print(f"执行命令: {cmd}")
            subprocess.check_output(
                cmd,
                shell=True,
                timeout=MINERU_TIMEOUT
            )

            # 查找生成的Markdown文件
            pdf_basename = os.path.basename(pdf_path).split(".")[0]
            markdown_pattern = os.path.join(
                output_dir,
                pdf_basename,
                "**",
                "*.md"
            )
            markdown_files = glob.glob(markdown_pattern, recursive=True)

            if markdown_files:
                return markdown_files[0]
            else:
                raise FileNotFoundError(f"未找到生成的Markdown文件: {markdown_pattern}")

        except subprocess.TimeoutExpired:
            raise Exception(f"MinerU解析超时（{MINERU_TIMEOUT}秒）")
        except Exception as e:
            print(f"MinerU解析失败: {e}")
            traceback.print_exc()
            raise

    def process_document(
            self,
            markdown_path: str,
            file_id: int,
            file_name: str,
            file_path: str
    ) -> List[dict]:
        """
        处理完整文档：读取、分块、编码

        Args:
            markdown_path: Markdown文件路径
            file_id: 文件ID
            file_name: 文件名
            file_path: 原始文件路径

        Returns:
            List[dict]: 准备插入Milvus的数据列表
        """
        # 读取Markdown文件
        with open(markdown_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 文本分块
        chunks = self.split_text2chunks(lines)

        # 编码每个chunk
        data_list = []
        for chunk in chunks:
            try:
                text_bge_emb, text_clip_emb, image_clip_emb = self.encode_text_and_image(
                    chunk,
                    markdown_path
                )

                data_item = {
                    "text_vector": text_bge_emb,
                    "clip_text_vector": text_clip_emb,
                    "clip_image_vector": image_clip_emb,
                    "text": chunk,
                    "db_id": file_id,
                    "file_name": file_name,
                    "file_path": file_path
                }
                data_list.append(data_item)
            except Exception as e:
                print(f"处理chunk失败: {e}")
                traceback.print_exc()
                continue

        return data_list
