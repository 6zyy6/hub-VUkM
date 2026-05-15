"""
本地知识库问答系统
基于 LangChain + ChromaDB 实现文档检索 + LLM 回答流程

核心流程：
1. 文档加载（支持 txt/pdf/md）
2. 文本分割（RecursiveCharacterTextSplitter）
3. 向量化存储（ChromaDB + sentence-transformers）
4. 相似度检索（top-k）
5. LLM 基于检索上下文生成回答

参考：八斗学院第14周 LangChain 教程
"""

import os

# ============================================================
# 1. 模型配置（参考课程示例，使用通义千问）
# ============================================================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

LLM_MODEL = "qwen-flash"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = input("请输入 API Key：")

llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.3,
)

# 使用 OpenAI 兼容的 Embedding 接口（通义千问）
# 注意：通义千问 embedding-v2 模型更稳定，v3 对 dimensions 参数有限制
embeddings = OpenAIEmbeddings(
    model="text-embedding-v2",
    base_url=BASE_URL,
    api_key=API_KEY,
    # 不设置 dimensions，使用模型默认维度
    check_embedding_ctx_length=False,
)

# ============================================================
# 2. 文档加载
# ============================================================
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)


def load_documents(file_path: str):
    """
    根据文件后缀自动选择加载器
    支持：.txt, .pdf, .md
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .txt / .pdf / .md")

    documents = loader.load()
    print(f"[INFO] 加载文档: {file_path}，共 {len(documents)} 个分片")
    return documents


# ============================================================
# 3. 文本分割
# ============================================================
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    使用递归字符分割器将文档切分成小块
    chunk_size: 每个文本块的最大字符数
    chunk_overlap: 相邻文本块之间的重叠字符数（保证上下文连贯）
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[INFO] 文本分割完成: {len(documents)} 片 → {len(chunks)} 块")
    return chunks


# ============================================================
# 4. 向量存储（ChromaDB）
# ============================================================
from langchain_community.vectorstores import Chroma


def build_vector_store(chunks, persist_directory="./chroma_db"):
    """
    将文本块向量化并存储到 ChromaDB
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print(f"[INFO] 向量库构建完成，存储路径: {persist_directory}")
    return vectorstore


def load_vector_store(persist_directory="./chroma_db"):
    """
    从已有目录加载向量库
    """
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    print(f"[INFO] 从已有向量库加载: {persist_directory}")
    return vectorstore


# ============================================================
# 5. 检索 + LLM 回答
# ============================================================
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def create_qa_chain(vectorstore, top_k=3):
    """
    创建 RAG 问答链：检索 → 组装 prompt → LLM 生成
    """

    # 检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # Prompt 模板
    template = """你是一个专业的知识库问答助手。请根据以下检索到的参考内容来回答用户的问题。
如果参考内容中没有相关信息，请如实告知"知识库中未找到相关信息"，不要编造答案。

参考内容：
{context}

用户问题：{question}

请用简洁清晰的中文回答："""

    prompt = ChatPromptTemplate.from_template(template)

    # 文档格式化函数：将检索到的 Document 列表拼接成字符串
    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[来源 {i+1}] {doc.page_content}" for i, doc in enumerate(docs)
        )

    # RAG 链组装
    qa_chain = (
        {
            "context": retriever | format_docs,  # 检索 → 格式化
            "question": RunnablePassthrough(),    # 用户问题直接传递
        }
        | prompt        # 组装 prompt
        | llm           # LLM 生成
        | StrOutputParser()  # 提取纯文本
    )

    return qa_chain


# ============================================================
# 6. 交互式问答
# ============================================================
def interactive_qa(qa_chain):
    """
    启动交互式问答循环
    """
    print("\n" + "=" * 50)
    print("  本地知识库问答系统（输入 'quit' 退出）")
    print("=" * 50 + "\n")

    while True:
        question = input("请输入问题 >>> ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if not question:
            continue

        print("\n[思考中...]\n")
        answer = qa_chain.invoke(question)
        print(f"回答: {answer}\n")
        print("-" * 50 + "\n")


# ============================================================
# 7. 主流程（交互式）
# ============================================================
def main():
    print("=" * 55)
    print("       本地知识库问答系统")
    print("       支持 .txt / .pdf / .md 格式")
    print("=" * 55)

    # Step 1: 让用户输入文档绝对路径
    while True:
        print()
        file_path = input("请输入知识库文档的绝对路径: ").strip().strip('"').strip("'")

        if not file_path:
            print("[ERROR] 路径不能为空，请重新输入。")
            continue

        if not os.path.exists(file_path):
            print(f"[ERROR] 文件不存在: {file_path}，请重新输入。")
            continue

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".txt", ".pdf", ".md"):
            print(f"[ERROR] 不支持的格式: {ext}，仅支持 .txt / .pdf / .md，请重新输入。")
            continue

        break

    print(f"\n[OK] 已确认文件: {file_path}")

    # Step 2: 文本分割参数（可选自定义）
    chunk_size_input = input(f"请输入文本块大小 (直接回车默认500): ").strip()
    chunk_size = int(chunk_size_input) if chunk_size_input else 500

    chunk_overlap_input = input(f"请输入文本块重叠大小 (直接回车默认100): ").strip()
    chunk_overlap = int(chunk_overlap_input) if chunk_overlap_input else 100

    top_k_input = input(f"请输入检索返回文档数 (直接回车默认3): ").strip()
    top_k = int(top_k_input) if top_k_input else 3

    # Step 3: 加载文档
    print(f"\n--- 正在处理文档 ---")
    documents = load_documents(file_path)

    # Step 4: 文本分割
    chunks = split_documents(documents, chunk_size, chunk_overlap)

    # Step 5: 构建向量库
    vectorstore = build_vector_store(chunks)

    # Step 6: 创建问答链
    qa_chain = create_qa_chain(vectorstore, top_k)

    # Step 7: 进入交互式问答
    interactive_qa(qa_chain)


if __name__ == "__main__":
    main()
