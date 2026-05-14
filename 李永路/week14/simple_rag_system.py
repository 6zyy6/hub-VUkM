"""
基于 LangChain 的本地知识库问答系统
功能：文档加载 + 文本分割 + 向量存储 + 检索 + LLM 回答
参考：Week06/04-government-advanced-rag 项目
"""

import os
import yaml
from typing import List, Dict
from pathlib import Path

# LangChain 核心组件
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class LocalRAGSystem:
    """本地知识库问答系统"""
    
    def __init__(self, config_path: str = "rag_config.yaml"):
        """
        初始化 RAG 系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化嵌入模型
        self.embeddings = self._init_embeddings()
        
        # 初始化向量存储
        self.vectorstore = None
        
        # 初始化 LLM
        self.llm = self._init_llm()
        
        # 初始化检索器
        self.retriever = None
        
        # 初始化问答链
        self.qa_chain = None
        
        print("✓ RAG 系统初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                "embedding": {
                    "model_name": "BAAI/bge-small-zh-v1.5",
                    "model_kwargs": {"device": "cpu"},
                    "encode_kwargs": {"normalize_embeddings": True}
                },
                "llm": {
                    "model": "qwen-flash",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": "sk-3d13848166aa4a5c902ad99e6c141e73",
                    "temperature": 0.7,
                    "max_tokens": 1024
                },
                "text_splitter": {
                    "chunk_size": 500,
                    "chunk_overlap": 50
                },
                "vectorstore": {
                    "persist_directory": "./faiss_index"
                },
                "retrieval": {
                    "search_kwargs": {"k": 5}
                }
            }
    
    def _init_embeddings(self):
        """初始化嵌入模型"""
        emb_config = self.config["embedding"]
        embeddings = HuggingFaceEmbeddings(
            model_name=emb_config["model_name"],
            model_kwargs=emb_config.get("model_kwargs", {}),
            encode_kwargs=emb_config.get("encode_kwargs", {})
        )
        print(f"✓ 嵌入模型加载完成: {emb_config['model_name']}")
        return embeddings
    
    def _init_llm(self):
        """初始化大语言模型"""
        llm_config = self.config["llm"]
        llm = ChatOpenAI(
            model=llm_config["model"],
            base_url=llm_config["base_url"],
            api_key=llm_config["api_key"],
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 1024)
        )
        print(f"✓ LLM 初始化完成: {llm_config['model']}")
        return llm
    
    def load_documents(self, data_path: str, file_type: str = "auto") -> List:
        """
        加载文档
        
        Args:
            data_path: 文档路径（文件或目录）
            file_type: 文件类型 ('pdf', 'txt', 'docx', 'auto')
            
        Returns:
            文档列表
        """
        path = Path(data_path)
        
        if path.is_file():
            # 单个文件
            if file_type == "auto":
                file_type = path.suffix.lower().lstrip('.')
            
            if file_type == "pdf":
                loader = PyPDFLoader(str(path))
            elif file_type == "txt":
                loader = TextLoader(str(path), encoding='utf-8')
            elif file_type in ["docx", "doc"]:
                loader = UnstructuredWordDocumentLoader(str(path))
            else:
                raise ValueError(f"不支持的文件类型: {file_type}")
            
            documents = loader.load()
            print(f"✓ 加载文件: {path.name}, 共 {len(documents)} 个文档块")
            
        elif path.is_dir():
            # 目录
            if file_type == "auto":
                # 自动检测目录中的文件类型
                loaders = []
                for ext in ['*.pdf', '*.txt', '*.docx']:
                    try:
                        pattern = f"**/{ext}"
                        files = list(path.glob(pattern))
                        if files:
                            if ext == "*.pdf":
                                loader = DirectoryLoader(str(path), glob=pattern, loader_cls=PyPDFLoader)
                            elif ext == "*.txt":
                                loader = DirectoryLoader(str(path), glob=pattern, loader_cls=TextLoader, 
                                                       loader_kwargs={'encoding': 'utf-8'})
                            loaders.append(loader)
                    except:
                        continue
                
                if not loaders:
                    raise ValueError(f"目录中未找到支持的文档文件: {data_path}")
                
                # 合并所有加载器的文档
                documents = []
                for loader in loaders:
                    documents.extend(loader.load())
            else:
                # 指定类型
                if file_type == "pdf":
                    loader = DirectoryLoader(str(path), glob="**/*.pdf", loader_cls=PyPDFLoader)
                elif file_type == "txt":
                    loader = DirectoryLoader(str(path), glob="**/*.txt", loader_cls=TextLoader,
                                           loader_kwargs={'encoding': 'utf-8'})
                else:
                    raise ValueError(f"不支持的目录文件类型: {file_type}")
                
                documents = loader.load()
            
            print(f"✓ 加载目录: {data_path}, 共 {len(documents)} 个文档块")
        else:
            raise FileNotFoundError(f"路径不存在: {data_path}")
        
        return documents
    
    def split_documents(self, documents: List) -> List:
        """
        分割文档
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档块列表
        """
        splitter_config = self.config["text_splitter"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=splitter_config["chunk_size"],
            chunk_overlap=splitter_config["chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✓ 文档分割完成: {len(documents)} -> {len(chunks)} 个文本块")
        print(f"  - Chunk大小: {splitter_config['chunk_size']}, 重叠: {splitter_config['chunk_overlap']}")
        
        return chunks
    
    def build_vectorstore(self, chunks: List, persist: bool = True):
        """
        构建向量存储
        
        Args:
            chunks: 文本块列表
            persist: 是否持久化到磁盘
        """
        vs_config = self.config["vectorstore"]
        persist_dir = vs_config["persist_directory"]
        
        # 创建向量存储
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print(f"✓ 向量存储构建完成: {len(chunks)} 个向量")
        
        # 持久化
        if persist:
            os.makedirs(persist_dir, exist_ok=True)
            self.vectorstore.save_local(persist_dir)
            print(f"✓ 向量存储已保存到: {persist_dir}")
    
    def load_vectorstore(self, persist_dir: str = None):
        """
        从磁盘加载向量存储
        
        Args:
            persist_dir: 持久化目录路径
        """
        if persist_dir is None:
            persist_dir = self.config["vectorstore"]["persist_directory"]
        
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"向量存储目录不存在: {persist_dir}")
        
        self.vectorstore = FAISS.load_local(
            folder_path=persist_dir,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print(f"✓ 向量存储加载完成: {persist_dir}")
    
    def init_retriever(self, search_kwargs: Dict = None):
        """
        初始化检索器
        
        Args:
            search_kwargs: 检索参数
        """
        if self.vectorstore is None:
            raise ValueError("请先构建或加载向量存储")
        
        if search_kwargs is None:
            search_kwargs = self.config["retrieval"]["search_kwargs"]
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs=search_kwargs
        )
        
        print(f"✓ 检索器初始化完成: top_k={search_kwargs.get('k', 5)}")
    
    def build_qa_chain(self):
        """构建问答链"""
        if self.retriever is None:
            self.init_retriever()
        
        # 定义提示词模板
        template = """你是一个专业的知识问答助手。请根据提供的参考资料回答问题。
如果资料中没有相关信息，请如实告知用户"根据现有资料无法回答该问题"。

参考资料：
{context}

问题：{question}

请逐步分析并给出准确的回答："""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 构建检索增强生成链
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        self.qa_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✓ 问答链构建完成")
    
    def query(self, question: str) -> str:
        """
        查询问答
        
        Args:
            question: 用户问题
            
        Returns:
            AI 回答
        """
        if self.qa_chain is None:
            self.build_qa_chain()
        
        print(f"\n🔍 正在检索相关知识...")
        answer = self.qa_chain.invoke(question)
        
        return answer
    
    def query_with_sources(self, question: str) -> Dict:
        """
        查询问答并返回来源
        
        Args:
            question: 用户问题
            
        Returns:
            包含回答和来源的字典
        """
        if self.qa_chain is None:
            self.build_qa_chain()
        
        # 获取相关文档
        relevant_docs = self.retriever.invoke(question)
        
        # 生成回答
        answer = self.qa_chain.invoke(question)
        
        # 提取来源信息
        sources = []
        for i, doc in enumerate(relevant_docs, 1):
            source_info = {
                "index": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "source_count": len(sources)
        }
    
    def process_pipeline(self, data_path: str, file_type: str = "auto", build: bool = True):
        """
        完整的处理流程：加载 -> 分割 -> 构建向量库
        
        Args:
            data_path: 数据路径
            file_type: 文件类型
            build: 是否重新构建（False 则从磁盘加载）
        """
        if build:
            # 步骤1: 加载文档
            print("\n" + "="*60)
            print("步骤 1: 加载文档")
            print("="*60)
            documents = self.load_documents(data_path, file_type)
            
            # 步骤2: 分割文档
            print("\n" + "="*60)
            print("步骤 2: 分割文档")
            print("="*60)
            chunks = self.split_documents(documents)
            
            # 步骤3: 构建向量存储
            print("\n" + "="*60)
            print("步骤 3: 构建向量存储")
            print("="*60)
            self.build_vectorstore(chunks, persist=True)
        else:
            # 从磁盘加载
            print("\n" + "="*60)
            print("从磁盘加载向量存储")
            print("="*60)
            self.load_vectorstore()
        
        # 步骤4: 初始化检索器和问答链
        print("\n" + "="*60)
        print("步骤 4: 初始化检索系统")
        print("="*60)
        self.init_retriever()
        self.build_qa_chain()
        
        print("\n" + "="*60)
        print("✅ RAG 系统就绪！可以开始提问了")
        print("="*60)


def main():
    """主函数 - 演示使用"""
    
    # 初始化 RAG 系统
    rag = LocalRAGSystem(config_path="rag_config.yaml")
    
    # 示例1: 处理单个 PDF 文件
    rag.process_pipeline("documents/汽车知识手册.pdf", file_type="pdf", build=True)
    
    # 示例2: 处理文本文件目录
    # rag.process_pipeline("documents/texts", file_type="txt", build=True)
    
    # 示例3: 从已有向量库加载
    try:
        rag.process_pipeline("./faiss_index", build=False)
    except FileNotFoundError:
        print("⚠️  未找到已有的向量存储，请先构建向量库")
        return
    
    # 交互式问答
    print("\n" + "="*60)
    print("开始问答（输入 'quit' 或 'exit' 退出）")
    print("="*60)
    
    while True:
        question = input("\n❓ 请输入您的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("再见！👋")
            break
        
        if not question:
            continue
        
        try:
            # 方式1: 简单问答
            # answer = rag.query(question)
            # print(f"\n💡 回答:\n{answer}")
            
            # 方式2: 带来源的问答
            result = rag.query_with_sources(question)
            print(f"\n💡 回答:\n{result['answer']}")
            
            print(f"\n📚 参考来源 ({result['source_count']} 条):")
            for source in result['sources']:
                print(f"\n[{source['index']}] {source['content']}")
                if source['metadata']:
                    print(f"    元数据: {source['metadata']}")
        
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
