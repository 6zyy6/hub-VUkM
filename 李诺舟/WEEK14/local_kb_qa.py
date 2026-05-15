from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DEFAULT_KB_DIR = Path("D:/1_Project/knowledge_base")
SUPPORTED_SUFFIXES = {".md", ".txt", ".py"}


def iter_source_files(kb_dir: Path) -> Iterable[Path]:
    for path in kb_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_documents(kb_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in iter_source_files(kb_dir):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(path)},
            )
        )
    return documents


def build_vectorstore(documents: list[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return FAISS.from_documents(chunks, embeddings)


def format_context(documents: list[Document]) -> str:
    parts: list[str] = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[片段{index}] 来源: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, kb_dir: Path, top_k: int = 4) -> dict[str, object]:
    documents = load_documents(kb_dir)
    if not documents:
        raise ValueError(f"知识库目录中没有可用文件: {kb_dir}")

    vectorstore = build_vectorstore(documents)
    retrieved_docs = vectorstore.similarity_search(question, k=top_k)
    context = format_context(retrieved_docs)

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("请先设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    llm = ChatOpenAI(
        model=os.getenv("LC_LLM_MODEL", "qwen-flash"),
        base_url=os.getenv("LC_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=api_key,
        temperature=0.2,
    )

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "你是本地知识库问答助手。"
                    "只能依据提供的检索片段回答。"
                    "如果上下文不足，就明确说不知道，并指出缺少的信息。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题: {question}\n\n"
                    f"检索上下文:\n{context}\n\n"
                    "请先给出简洁答案，再列出你主要依据的来源文件。"
                )
            ),
        ]
    )

    return {
        "answer": response.content,
        "sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 LangChain 的本地知识库问答")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR, help="本地知识库目录")
    parser.add_argument("--question", required=True, help="用户问题")
    parser.add_argument("--top-k", type=int, default=4, help="检索返回片段数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = answer_question(args.question, args.kb_dir, args.top_k)
    print("回答:\n")
    print(result["answer"])
    print("\n来源:")
    for source in result["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()
