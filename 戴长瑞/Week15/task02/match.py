# requirements.txt
# pip install pdfplumber sentence-transformers chromadb mineru-python

import pdfplumber
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


# ---------- 1. 使用 pdfplumber 提取并分块 ----------
def pdfplumber_chunking(pdf_path, chunk_size=512):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # 简单按句号+换行分块
    sentences = re.split(r'(?<=[。!?])\s+', full_text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += sent
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks


# ---------- 2. 使用 MinerU 解析并分块 ----------
def mineru_chunking(pdf_path):
    # 需要先启动 MinerU 服务: docker run -d -p 8080:8080 mineru/server:latest
    import requests
    import base64

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    resp = requests.post(
        "http://localhost:8080/extract",
        json={"file_b64": pdf_b64, "return_format": "markdown"}
    )
    data = resp.json()
    markdown = data["markdown"]

    # 按标题分块 (## 开头)
    lines = markdown.split("\n")
    chunks = []
    current = []
    for line in lines:
        if line.startswith("##"):
            if current:
                chunks.append("\n".join(current))
                current = []
        current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------- 3. 构建向量库并检索测试 ----------
def build_retriever(chunks, model_name="all-MiniLM-L6-v2"):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = chromadb.Client()
    collection = client.create_collection("docs", embedding_function=ef)
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[str(i)])
    return collection


def search(collection, query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results['documents'][0]


# ---------- 4. 主流程对比 ----------
if __name__ == "__main__":
    pdf_file = "2509-MinerU2.5.pdf"

    # pdfplumber 管道
    chunks_pdf = pdfplumber_chunking(pdf_file)
    col_pdf = build_retriever(chunks_pdf)

    # MinerU 管道（需要先启动服务）
    # chunks_mineru = mineru_chunking(pdf_file)
    # col_mineru = build_retriever(chunks_mineru)

    # 测试查询
    queries = [
        "表格中第三季度的销售额是多少？",
        "公式 E=mc^2 出现在哪一页？",
        "作者对双栏排版的结论是什么？"
    ]

    print("=== pdfplumber 检索结果 ===")
    for q in queries:
        res = search(col_pdf, q)
        print(f"Q: {q}\nTop1: {res[:200]}...\n")

    # 取消注释即可对比 MinerU
    # print("=== MinerU 检索结果 ===")
    # for q in queries:
    #     res = search(col_mineru, q)
    #     print(f"Q: {q}\nTop1: {res[:200]}...\n")
