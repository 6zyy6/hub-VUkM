import streamlit as st
import openai
import re
import os
from sentence_transformers import SentenceTransformer

from services import RetrievalService, ChatService
from config import BGE_MODEL_PATH, QWEN_MODEL

# 初始化服务
retrieval_service = RetrievalService()
chat_service = ChatService()

# 加载BGE模型
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
bge_model = SentenceTransformer(BGE_MODEL_PATH)


def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手。"}
    ]


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手。"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.sidebar:
    st.button('清空当前聊天', on_click=clear_chat_history, use_container_width=True)


def render_markdown_with_images(markdown_text):
    pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    last_pos = 0

    for match in pattern.finditer(markdown_text):
        if match.start() > last_pos:
            st.markdown(markdown_text[last_pos:match.start()], unsafe_allow_html=True)
        img_url = match.group(1)
        st.image(img_url)
        last_pos = match.end()

    if last_pos < len(markdown_text):
        st.markdown(markdown_text[last_pos:], unsafe_allow_html=True)


prompt = st.chat_input("请输入您的问题...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在检索相关资料..."):
            # 使用BGE编码
            prompt_embedding = bge_model.encode(prompt, normalize_embeddings=True)

            # 使用检索服务
            results = retrieval_service.search_relevant_texts(
                query_embedding=list(prompt_embedding),
                limit=5
            )

            # 格式化上下文
            context = retrieval_service.format_context_for_qwen(results)
            source_files = set([r['file_name'] for r in results])

        with st.spinner("正在生成答案..."):
            # 使用问答服务
            answer = chat_service.generate_answer(
                question=prompt,
                context=context
            )

            if source_files:
                answer += f"\n\n---\n**信息来源**: {', '.join(source_files)}"

            render_markdown_with_images(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
