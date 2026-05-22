"""
文件管理服务 - 处理文件上传、删除、查询等操作
"""
import os
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session

from config import UPLOAD_DIR, FileState
from orm_model import File, get_session


class FileService:
    """文件管理服务类"""

    @staticmethod
    def save_uploaded_file(uploaded_file) -> str:
        """
        保存上传的文件到本地

        Args:
            uploaded_file: Streamlit上传的文件对象

        Returns:
            str: 保存路径
        """
        import uuid
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1]
        unique_name = f"{uuid.uuid4()}{file_extension}"
        save_path = UPLOAD_DIR / unique_name

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        return str(save_path)

    @staticmethod
    def create_file_record(filename: str, filepath: str) -> File:
        """
        在数据库中创建文件记录

        Args:
            filename: 原始文件名
            filepath: 文件保存路径

        Returns:
            File: 创建的文件记录对象
        """
        session = get_session()
        try:
            file_record = File(
                filename=filename,
                filepath=filepath,
                filestate=FileState.UPLOADED,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            session.add(file_record)
            session.commit()
            session.refresh(file_record)
            return file_record
        finally:
            session.close()

    @staticmethod
    def get_all_files() -> List[File]:
        """获取所有文件记录"""
        session = get_session()
        try:
            return session.query(File).order_by(File.created_at.desc()).all()
        finally:
            session.close()

    @staticmethod
    def get_file_by_id(file_id: int) -> Optional[File]:
        """根据ID获取文件记录"""
        session = get_session()
        try:
            return session.query(File).filter(File.id == file_id).first()
        finally:
            session.close()

    @staticmethod
    def update_file_state(file_id: int, state: str, error_message: str = None):
        """
        更新文件状态

        Args:
            file_id: 文件ID
            state: 新状态
            error_message: 错误信息（可选）
        """
        session = get_session()
        try:
            file_record = session.query(File).filter(File.id == file_id).first()
            if file_record:
                file_record.filestate = state
                file_record.updated_at = datetime.now()
                if error_message:
                    file_record.error_message = error_message
                session.commit()
        finally:
            session.close()

    @staticmethod
    def delete_file(file_id: int) -> bool:
        """
        删除文件记录和本地文件

        Args:
            file_id: 文件ID

        Returns:
            bool: 是否删除成功
        """
        session = get_session()
        try:
            file_record = session.query(File).filter(File.id == file_id).first()
            if file_record:
                # 删除本地文件
                if os.path.exists(file_record.filepath):
                    try:
                        os.remove(file_record.filepath)
                    except Exception as e:
                        print(f"删除本地文件失败: {e}")

                # 删除数据库记录
                session.delete(file_record)
                session.commit()
                return True
            return False
        finally:
            session.close()

    @staticmethod
    def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
        """
        验证文件类型

        Args:
            filename: 文件名
            allowed_types: 允许的文件扩展名列表

        Returns:
            bool: 是否为允许的文件类型
        """
        if allowed_types is None:
            allowed_types = ['.pdf', '.docx', '.txt']

        ext = os.path.splitext(filename)[1].lower()
        return ext in allowed_types
