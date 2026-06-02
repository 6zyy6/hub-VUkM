"""工具函数"""
import hashlib
import os
from typing import List, Optional
import json


def generate_id(content: str) -> str:
    """生成内容 ID"""
    return hashlib.md5(content.encode()).hexdigest()[:16]


def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    import re
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename[:200]


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def read_file(file_path: str) -> Optional[str]:
    """读取文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None


def write_file(file_path: str, content: str) -> bool:
    """写入文本文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except:
        return False


def list_files(dir_path: str, extension: Optional[str] = None) -> List[str]:
    """列出目录下的文件"""
    files = []
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            full_path = os.path.join(dir_path, f)
            if os.path.isfile(full_path):
                if extension is None or f.endswith(extension):
                    files.append(full_path)
    return files


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """分块列表"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]