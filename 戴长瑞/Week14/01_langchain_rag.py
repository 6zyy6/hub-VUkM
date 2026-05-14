import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ========== 1. 使用本地 jina embedding 模型 ==========
embedding_model = HuggingFaceEmbeddings(
    model_name=r"C:\Users\Admin\models\jinaai\jina-embeddings-v2-base-zh",
    model_kwargs={'device': 'cpu'}      #  GPU => 'cuda'
)

# ========== 2. 示例文档（知识库）==========
documents = [
    "LangChain 是一个用于构建 LLM 应用的开源框架。",
    "FAISS 是 Facebook 开源的向量相似度检索库，适合本地部署。",
    "RAG（检索增强生成）结合了检索系统和生成模型，能提升回答准确性。",
    "本地知识库问答通常采用 Embedding + 向量库 + LLM 的架构。"
]

vectorstore = FAISS.from_texts(documents, embedding_model)

# ========== 3. 初始化本地 LLM（Ollama）==========
# 本地: ollama serve qwen2.5:7b
llm = Ollama(model="qwen2.5:7b", temperature=0.3)

# ========== 4. 创建检索问答链 ==========
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# ========== 5. 执行问答 ==========
if __name__ == "__main__":
    question = "什么是 RAG？"
    answer = qa_chain.invoke({"query": question})
    print(f"问题: {question}\n回答: {answer['result']}")