import pdfplumber
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


# ---------- pdfplumber 分块 ----------
def pdfplumber_chunking(pdf_path, chunk_size=512):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    # 分块
    sentences = re.split(r'(?<=[。!?])\s+', full_text)
    chunks = []
    cur = ""
    for sent in sentences:
        if len(cur) + len(sent) < chunk_size:
            cur += sent
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return chunks


# ---------- 构建 chromadb 检索器 ----------
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


# ---------- 主程序 ----------
if __name__ == "__main__":
    pdf_file = "2509-MinerU2.5.pdf"

    print("正在使用 pdfplumber 提取并分块...")
    chunks = pdfplumber_chunking(pdf_file)
    print(f"共生成 {len(chunks)} 个文本块")

    print("正在构建向量索引...")
    collection = build_retriever(chunks)

    queries = [
        "文档的主要结论是什么？",
        "提到了哪些技术指标？"
    ]

    print("\n=== 检索结果 ===")
    for q in queries:
        res = search(collection, q)
        print(f"查询: {q}")
        print(f"最相关片段: {res[:300]}...\n")