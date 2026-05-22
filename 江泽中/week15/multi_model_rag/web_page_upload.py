import streamlit as st
from services import FileService
from utils.kafka_utils import send_parse_task_to_kafka
import socket


def check_kafka_connection():
    """检查Kafka是否可用"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 9092))
        sock.close()
        return result == 0
    except:
        return False


def query_files():
    """查询所有文件并展示"""
    files = FileService.get_all_files()

    if files:
        for file in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{file.filename}**")
                st.caption(f"状态: {file.filestate} | 路径: {file.filepath}")

            with col2:
                if st.button(f"🗑️ 删除", key=f"delete_{file.id}"):
                    FileService.delete_file(file.id)

                    # 同时删除Milvus中的数据
                    from pymilvus import MilvusClient
                    from config import MILVUS_URI, MILVUS_TOKEN
                    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
                    client.delete(collection_name="rag_data_new", filter=f"db_id == {file.id}")

                    st.rerun()
    else:
        st.info("暂无文件，请上传文档开始使用。")


def delete_file_with_milvus(file_id):
    """删除文件并清理Milvus数据"""
    # 删除文件和数据库记录
    FileService.delete_file(file_id)

    # 删除Milvus向量数据
    from pymilvus import MilvusClient
    from config import MILVUS_URI, MILVUS_TOKEN
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    client.delete(collection_name="rag_data_new", filter=f"db_id == {file_id}")


st.markdown("### 📁 文件管理")
query_files()

st.markdown("---")
st.markdown("### ⬆️ 文件上传")

uploaded_file = st.file_uploader(
    "选择要上传的文件",
    type=["pdf", "docx", "txt"],
    help="支持PDF、DOCX、TXT格式文档"
)

if uploaded_file is not None:
    file_name = uploaded_file.name

    # 验证文件类型
    if not FileService.validate_file_type(file_name):
        st.error("不支持的文件类型！")
        st.stop()

    with st.spinner(f"正在上传 {file_name}..."):
        # 使用服务层保存文件
        save_path = FileService.save_uploaded_file(uploaded_file)

        # 创建数据库记录
        file_record = FileService.create_file_record(file_name, save_path)

        kafka_available = check_kafka_connection()

        if kafka_available:
            try:
                # 发送Kafka消息
                send_parse_task_to_kafka(
                    file_name=file_name,
                    file_path=save_path,
                    file_id=file_record.id
                )

                st.success(f"✅ 文件上传成功！\n\n文件名: {file_name}\n状态: 已发送到解析队列")
            except Exception as e:
                FileService.update_file_state(file_record.id, "失败", str(e))
                st.warning(f"⚠️ 文件已保存，但解析任务发送失败: {str(e)}")
        else:
            st.warning(f"⚠️ 文件已保存，但Kafka服务未启动，无法自动解析。")
