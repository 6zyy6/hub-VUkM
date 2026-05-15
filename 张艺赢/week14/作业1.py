"""
本地知识库问答系统 - 基于LangChain框架
作业1: 文档检索 + LLM回答流程
"""
import os
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Iterator
from pathlib import Path
from html.parser import HTMLParser
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

TOKEN_PATTERN = re.compile(r"[a-z0-9_+-]+|[&]|[\u4e00-\u9fff]+", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
TAG_PATTERN = re.compile(r"<[^>]+>")
BLOCK_PATTERN = re.compile(
    r"<(h[1-3]|p|li)[^>]*>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
IGNORED_BLOCK_PATTERN = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", unescape(text)).strip()


def strip_tags(html_fragment: str) -> str:
    cleaned = IGNORED_BLOCK_PATTERN.sub(" ", html_fragment)
    cleaned = TAG_PATTERN.sub(" ", cleaned)
    return normalize_text(cleaned)


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text).lower()
    tokens: List[str] = []
    for match in TOKEN_PATTERN.finditer(normalized):
        chunk = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
                continue
            for size in (2, 3):
                if len(chunk) >= size:
                    tokens.extend(chunk[i:i + size] for i in range(len(chunk) - size + 1))
        else:
            tokens.append(chunk)
    return tokens


class VisibleTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._in_title = False
        self._text_parts: List[str] = []
        self._title_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._text_parts.append(data)

    @property
    def visible_text(self) -> str:
        return normalize_text(" ".join(self._text_parts))

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self._title_parts))


def extract_visible_text_and_title(html: str) -> tuple[str, str]:
    parser = VisibleTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.visible_text, parser.title


def build_snippet(text: str, query: str, radius: int = 100) -> str:
    normalized_query = normalize_text(query).lower()
    if not text:
        return ""
    if not normalized_query:
        return text[: radius * 2] + ("..." if len(text) > radius * 2 else "")

    lower_text = text.lower()
    position = lower_text.find(normalized_query)
    if position >= 0:
        start = max(0, position - radius)
        end = min(len(text), position + len(normalized_query) + radius)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet
    return text[: radius * 2] + ("..." if len(text) > radius * 2 else "")


@dataclass
class KnowledgeDocument:
    id: str
    title: str
    content: str
    tokens: List[str]
    term_freq: Counter


class SimpleVectorStore:
    def __init__(self, embedding_model: Optional[Embeddings] = None):
        self._documents: Dict[str, KnowledgeDocument] = {}
        self._doc_freq: defaultdict[str, int] = defaultdict(int)
        self._total_terms = 0
        self._embedding_model = embedding_model
        self._vectors: Dict[str, List[float]] = {}

    def add_document(self, doc_id: str, title: str, content: str) -> None:
        text = normalize_text(content)
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("Document has no indexable text")

        previous = self._documents.get(doc_id)
        if previous:
            self._total_terms -= previous.token_count
            for token in previous.term_freq:
                self._doc_freq[token] -= 1
                if self._doc_freq[token] <= 0:
                    del self._doc_freq[token]

        doc = KnowledgeDocument(
            id=doc_id,
            title=title or doc_id,
            content=text,
            tokens=tokens,
            term_freq=Counter(tokens),
        )
        self._documents[doc_id] = doc
        self._total_terms += len(tokens)
        for token in doc.term_freq:
            self._doc_freq[token] += 1

        if self._embedding_model and doc.content:
            embedding = self._embedding_model.embed_query(doc.content)
            self._vectors[doc_id] = embedding

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        doc_count = len(self._documents)
        if doc_count == 0:
            return []

        average_length = self._total_terms / doc_count if self._total_terms else 0.0
        results = []

        for doc in self._documents.values():
            score = self._bm25_score(doc, query_tokens, doc_count, average_length)
            if score > 0:
                results.append({
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content,
                    "snippet": build_snippet(doc.content, query),
                    "score": round(score, 6)
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _bm25_score(
        self,
        document: KnowledgeDocument,
        query_tokens: List[str],
        doc_count: int,
        average_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        score = 0.0
        denominator_base = k1 * (1 - b + b * len(document.tokens) / average_length) if average_length else k1

        for token in query_tokens:
            frequency = document.term_freq.get(token, 0)
            if frequency == 0:
                continue
            doc_frequency = self._doc_freq.get(token, 0)
            idf = math.log(1 + (doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            score += idf * ((frequency * (k1 + 1)) / (frequency + denominator_base))
        return score

    def search_hybrid(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        bm25_results = self.search_bm25(query, top_k * 2)
        if not bm25_results:
            return []

        if not self._embedding_model or not self._vectors:
            return bm25_results[:top_k]

        query_embedding = self._embedding_model.embed_query(query)

        for result in bm25_results:
            doc_vector = self._vectors.get(result["id"])
            if doc_vector:
                similarity = self._cosine_similarity(query_embedding, doc_vector)
                result["semantic_score"] = similarity
                result["score"] = 0.5 * result["score"] + 0.5 * similarity

        bm25_results.sort(key=lambda x: x["score"], reverse=True)
        return bm25_results[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def as_retriever(self, search_type: str = "similarity", top_k: int = 5):
        return SimpleVectorStoreRetriever(vectorstore=self, search_type=search_type, top_k=top_k)

    def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        return self._documents.get(doc_id)


class SimpleVectorStoreRetriever:
    def __init__(self, vectorstore: SimpleVectorStore, search_type: str = "similarity", top_k: int = 5):
        self._vectorstore = vectorstore
        self._search_type = search_type
        self._top_k = top_k

    def invoke(self, query: str) -> List[Document]:
        if self._search_type == "mmr":
            results = self._vectorstore.search_hybrid(query, self._top_k)
        else:
            results = self._vectorstore.search_bm25(query, self._top_k)

        return [
            Document(
                page_content=result["snippet"],
                metadata={"id": result["id"], "title": result["title"], "full_content": result["content"]}
            )
            for result in results
        ]


class DashScopeEmbeddings(BaseModel):
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    model_name: str = "text-embedding-v4"
    dimensions: int = 1024

    def embed_query(self, text: str) -> List[float]:
        if not self.api_key:
            return self._fallback_embedding(text)

        endpoint = f"{self.base_url}/services/embeddings/text-embedding/text-embedding"
        payload = {
            "model": self.model_name,
            "input": {"texts": [text]},
            "parameters": {"dimension": self.dimensions, "output_type": "dense"},
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                embeddings = result.get("output", {}).get("embeddings", [])
                if embeddings:
                    return embeddings[0].get("embedding", [])
        except (HTTPError, URLError):
            pass
        return self._fallback_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def _fallback_embedding(self, text: str) -> List[float]:
        tokens = tokenize(text)
        vector = [0.0] * self.dimensions
        for i, token in enumerate(tokens[:self.dimensions]):
            vector[i] = hash(token) % 1000 / 1000.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class DashScopeLLM:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen3.6-plus",
    ):
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "").strip()
        self._base_url = base_url
        self._model_name = model_name

    def invoke(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY")

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        request = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DashScope chat failed: HTTP {exc.code}. {details}")
        except URLError as exc:
            raise RuntimeError(f"DashScope chat failed: {exc.reason}")


class RAGChain:
    def __init__(
        self,
        vectorstore: SimpleVectorStore,
        llm: Optional[DashScopeLLM] = None,
        system_prompt: str = ""
    ):
        self._vectorstore = vectorstore
        self._llm = llm
        self._system_prompt = system_prompt or (
            "你是一个专业的知识库问答助手。\n"
            "根据提供的参考文档片段回答用户问题。\n"
            "如果参考文档中没有相关信息，请明确告知用户。\n"
            "请用简洁、有条理的方式组织答案。"
        )

    def _build_context(self, docs: List[Document]) -> str:
        if not docs:
            return "没有找到相关参考文档。"
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "未知文档")
            content = doc.page_content
            context_parts.append(f"【文档{i}】{title}\n{content}")
        return "\n\n".join(context_parts)

    def _build_messages(self, query: str, docs: List[Document]) -> List[Dict[str, str]]:
        context = self._build_context(docs)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"参考文档：\n{context}\n\n问题：{query}"}
        ]
        return messages

    def invoke(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        docs = self._vectorstore.as_retriever(top_k=top_k).invoke(query)
        if not self._llm:
            return {
                "query": query,
                "answer": "LLM未配置，无法生成回答。请配置DASHSCOPE_API_KEY。",
                "source_documents": [
                    {"title": doc.metadata.get("title"), "content": doc.page_content}
                    for doc in docs
                ]
            }

        try:
            messages = self._build_messages(query, docs)
            response = self._llm.invoke(messages)
            choice = response.get("choices", [{}])[0]
            answer = choice.get("message", {}).get("content", "")
            return {
                "query": query,
                "answer": answer.strip() if answer else "抱歉，无法生成回答。",
                "source_documents": [
                    {"title": doc.metadata.get("title"), "content": doc.page_content}
                    for doc in docs
                ]
            }
        except Exception as e:
            return {
                "query": query,
                "answer": f"生成回答时出错: {str(e)}",
                "source_documents": [
                    {"title": doc.metadata.get("title"), "content": doc.page_content}
                    for doc in docs
                ]
            }


def create_knowledge_base_qa_system(
    documents_path: str = "./knowledge_base",
    embedding_api_key: str = "",
    llm_api_key: str = "",
) -> RAGChain:
    """
    创建本地知识库问答系统
    """
    embedding_model = None
    if embedding_api_key:
        embedding_model = DashScopeEmbeddings(api_key=embedding_api_key)
    elif os.getenv("DASHSCOPE_API_KEY"):
        embedding_model = DashScopeEmbeddings(api_key=os.getenv("DASHSCOPE_API_KEY", ""))

    vectorstore = SimpleVectorStore(embedding_model=embedding_model)

    docs_path = Path(documents_path)
    if docs_path.exists() and docs_path.is_dir():
        for html_file in docs_path.glob("*.html"):
            try:
                html_content = html_file.read_text(encoding="utf-8")
                text, title = extract_visible_text_and_title(html_content)
                doc_id = html_file.stem
                vectorstore.add_document(doc_id, title or doc_id, text)
            except Exception:
                continue

    llm = None
    if llm_api_key:
        llm = DashScopeLLM(api_key=llm_api_key)
    elif os.getenv("DASHSCOPE_API_KEY"):
        llm = DashScopeLLM(api_key=os.getenv("DASHSCOPE_API_KEY", ""))

    return RAGChain(vectorstore=vectorstore, llm=llm)


if __name__ == "__main__":
    qa_system = create_knowledge_base_qa_system()
    result = qa_system.invoke("LangChain是什么？")
    print(f"问题: {result['query']}")
    print(f"回答: {result['answer']}")
    print(f"参考文档数量: {len(result['source_documents'])}")
