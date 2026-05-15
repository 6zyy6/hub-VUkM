import argparse
import os
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np


class MarkdownChunk:
    def __init__(self, text: str, header: str, source_path: Path):
        self.text = text
        self.header = header
        self.source_path = Path(source_path)


class RetrievalResult:
    def __init__(self, chunk: MarkdownChunk, score: float):
        self.chunk = chunk
        self.score = score


EmbeddingFunction = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def split_markdown_by_headers_simple(
    markdown_text: str,
    path: Path,
    max_length: Optional[int] = 1024,
) -> List[MarkdownChunk]:
    """Split Markdown text by headings, then split oversized sections."""
    header_pattern = re.compile(r"^#+\s+.+$")
    chunks: List[MarkdownChunk] = []
    current_chunk: List[str] = []
    current_header = "Document"

    for line in markdown_text.splitlines():
        if header_pattern.match(line.strip()):
            if current_chunk:
                text = "\n".join(current_chunk).strip()
                if text:
                    chunks.append(MarkdownChunk(text, current_header, path))
            current_chunk = [line]
            current_header = line.strip()
        else:
            current_chunk.append(line)

    if current_chunk:
        text = "\n".join(current_chunk).strip()
        if text:
            chunks.append(MarkdownChunk(text, current_header, path))

    if not max_length:
        return chunks

    final_chunks: List[MarkdownChunk] = []
    for chunk in chunks:
        if len(chunk.text) <= max_length:
            final_chunks.append(chunk)
            continue

        for start in range(0, len(chunk.text), max_length):
            part_number = start // max_length + 1
            final_chunks.append(
                MarkdownChunk(
                    chunk.text[start : start + max_length],
                    f"{chunk.header} (Part {part_number})",
                    chunk.source_path,
                )
            )

    return final_chunks


def load_markdown_chunks(
    documents_dir: Path,
    max_length: Optional[int] = 1024,
) -> List[MarkdownChunk]:
    root = Path(documents_dir)
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在：{root}")

    markdown_paths = sorted(root.rglob("*.md"))
    if not markdown_paths:
        raise ValueError(f"知识库目录中没有 Markdown 文件：{root}")

    chunks: List[MarkdownChunk] = []
    for path in markdown_paths:
        content = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(split_markdown_by_headers_simple(content, path, max_length=max_length))

    return chunks


def _is_external_or_absolute_path(path_text: str) -> bool:
    normalized = path_text.strip().lower()
    return (
        normalized.startswith("http://")
        or normalized.startswith("https://")
        or normalized.startswith("data:")
        or normalized.startswith("#")
        or Path(path_text).is_absolute()
        or re.match(r"^[a-zA-Z]:[\\/]", path_text) is not None
    )


def rewrite_relative_image_paths(chunk: MarkdownChunk) -> str:
    """Make relative Markdown image paths resolvable from the source file."""

    def replace_image(match: re.Match) -> str:
        alt_text = match.group(1)
        image_path = match.group(2).strip()
        if _is_external_or_absolute_path(image_path):
            return match.group(0)

        rewritten = (chunk.source_path.parent / image_path).as_posix()
        return f"![{alt_text}]({rewritten})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, chunk.text)


def get_embeddings(
    input_texts: Iterable[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 30,
) -> List[List[float]]:
    texts = [input_texts] if isinstance(input_texts, str) else list(input_texts)
    if not texts:
        return []

    import requests

    embedding_model = model or os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B/")
    embedding_base_url = (base_url or os.getenv("EMBEDDING_BASE_URL", "http://localhost:8081")).rstrip("/")
    url = f"{embedding_base_url}/v1/embeddings"

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"model": embedding_model, "input": texts},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"无法连接本地 embedding 服务：{url}") from exc

    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(f"embedding 服务返回格式不正确：{payload}")

    embeddings: List[List[float]] = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list):
            raise RuntimeError(f"embedding 服务返回缺少向量：{payload}")
        embeddings.append(embedding)

    return embeddings


def _normalize_vectors(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def retrieve_relevant_chunks(
    question: str,
    chunks: Sequence[MarkdownChunk],
    embedding_fn: EmbeddingFunction = get_embeddings,
    top_k: int = 5,
) -> List[RetrievalResult]:
    if not chunks:
        raise ValueError("没有可检索的文档分块")

    document_vectors = _normalize_vectors(embedding_fn([chunk.text for chunk in chunks]))
    question_vector = _normalize_vectors(embedding_fn([question]))
    scores = document_vectors @ question_vector[0]

    ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), item[0]))
    return [
        RetrievalResult(chunks[index], float(score))
        for index, score in ranked[: max(1, top_k)]
    ]


def build_prompt(question: str, results: Sequence[RetrievalResult]) -> str:
    context_parts = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        context_parts.append(
            "\n".join(
                [
                    f"[资料{index}]",
                    f"来源：{chunk.source_path}",
                    f"标题：{chunk.header}",
                    f"相似度：{result.score:.4f}",
                    rewrite_relative_image_paths(chunk),
                ]
            )
        )

    related_text = "\n\n".join(context_parts)
    return (
        f"已有资料：\n{related_text}\n\n"
        f"用户提问：{question}\n\n"
        "请只基于已有资料回答问题。如果资料不足，请说明无法从资料中确定。"
        "如果资料中包含图片链接，请在回答中保留相关图片并进行清晰的图文排版。"
    )


def build_llm():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请先在环境变量中配置 DashScope API Key")

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("DASHSCOPE_MODEL", "qwen-flash"),
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=api_key,
    )


def answer_question(
    question: str,
    documents_dir: Path = Path("./documents"),
    chunks: Optional[Sequence[MarkdownChunk]] = None,
    embedding_fn: EmbeddingFunction = get_embeddings,
    llm=None,
    top_k: int = 5,
) -> str:
    knowledge_chunks = list(chunks) if chunks is not None else load_markdown_chunks(documents_dir)
    results = retrieve_relevant_chunks(question, knowledge_chunks, embedding_fn=embedding_fn, top_k=top_k)
    prompt = build_prompt(question, results)
    model = llm or build_llm()

    response = model.invoke([{"role": "user", "content": prompt}])
    content = getattr(response, "content", None)
    return content if isinstance(content, str) else str(response)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="基于本地 Markdown 知识库进行检索增强问答")
    parser.add_argument("question", nargs="?", default="什么是VLLM的Memory layout？")
    parser.add_argument("--documents", default="./documents", help="本地知识库目录，默认 ./documents")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回的文档片段数量")
    args = parser.parse_args(argv)

    answer = answer_question(
        args.question,
        documents_dir=Path(args.documents),
        top_k=args.top_k,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
