"""PDF 解析服务 - MinerU / DeepSeek-OCR"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """解析结果"""
    markdown: str
    images: List[Dict]  # [{"path": str, "page": int, "bbox": tuple}]
    tables: List[Dict]  # [{"content": str, "page": int}]
    metadata: Dict


class BasePDFParser(ABC):
    """PDF 解析器基类"""

    @abstractmethod
    def parse(self, pdf_path: str, output_dir: str) -> ParseResult:
        """解析 PDF"""
        pass

    @abstractmethod
    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict]:
        """提取图片"""
        pass


class MinerUParser(BasePDFParser):
    """MinerU PDF 解析器"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._client = None

    def _init_client(self):
        """延迟初始化 MinerU 客户端"""
        if self._client is None:
            try:
                from markitdown import MarkItDown
                self._client = MarkItDown()
                logger.info("MinerU client initialized")
            except ImportError:
                logger.warning("markitdown not installed, using fallback")
                self._client = None

    def parse(self, pdf_path: str, output_dir: str) -> ParseResult:
        self._init_client()

        if self._client:
            result = self._client.convert(pdf_path)
            markdown = result.text_content
        else:
            markdown = self._fallback_parse(pdf_path)

        images = self.extract_images(pdf_path, output_dir)
        tables = self._extract_tables(markdown)

        return ParseResult(
            markdown=markdown,
            images=images,
            tables=tables,
            metadata={"parser": "mineru", "pages": self._count_pages(pdf_path)}
        )

    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict]:
        """使用 PyMuPDF 提取图片"""
        import fitz  # PyMuPDF

        os.makedirs(output_dir, exist_ok=True)
        images = []

        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_name = f"doc_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                image_path = os.path.join(output_dir, image_name)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "path": image_path,
                    "page": page_num + 1,
                    "bbox": img[1:5]  # bbox
                })

        doc.close()
        return images

    def _fallback_parse(self, pdf_path: str) -> str:
        """降级解析：提取纯文本"""
        import fitz
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)

    def _extract_tables(self, markdown: str) -> List[Dict]:
        """提取表格（简单正则匹配）"""
        import re
        tables = []
        table_pattern = r'\|.*\|.*\n\|[-:\s|]+\|.*'
        matches = re.finditer(table_pattern, markdown)
        for match in matches:
            tables.append({"content": match.group(), "page": 0})
        return tables

    def _count_pages(self, pdf_path: str) -> int:
        import fitz
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count


class DeepSeekOCRParser(BasePDFParser):
    """DeepSeek-OCR PDF 解析器"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

    def parse(self, pdf_path: str, output_dir: str) -> ParseResult:
        """使用 DeepSeek-OCR API 解析"""
        # TODO: 实现 DeepSeek-OCR 集成
        logger.info(f"DeepSeek-OCR parsing: {pdf_path}")
        raise NotImplementedError("DeepSeek-OCR parser not implemented")

    def extract_images(self, pdf_path: str, output_dir: str) -> List[Dict]:
        raise NotImplementedError("DeepSeek-OCR parser not implemented")