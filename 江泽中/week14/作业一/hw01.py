# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 1. 加载环境变量（API密钥）
load_dotenv(override=True)

# 2. 文档加载与分割
loader = DirectoryLoader(
    "./knowledge_base/",
    glob="**/*.md",
    show_progress=True
)
documents = loader.load()
print(f"加载 {len(documents)} 个文档")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""]
)
split_docs = text_splitter.split_documents(documents)
print(f"分割为 {len(split_docs)} 个文本块")

# 3. 向量存储
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vector_store = Chroma.from_documents(
    split_docs, embeddings, persist_directory="./chroma_db"
)

# 4. 初始化通义千问大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.3,
    max_tokens=1000
)

# 5. 构建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# 6. 问答
question = "用户提出的问题"
response = qa_chain.invoke({"query": question})
print("答案:", response["result"])