"""
文件处理工具函数

功能：
- 临时文件管理
- 文件格式验证
- 文件上传处理
"""

import os
import tempfile
from typing import Optional, Tuple
from fastapi import UploadFile
from .logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class TempFileManager:
    """
    临时文件管理器
    
    功能：
    - 创建临时文件
    - 自动清理临时文件
    - 支持上下文管理器
    """
    
    def __init__(self, suffix: str = "", prefix: str = "tmp_"):
        """
        初始化临时文件管理器
        
        Args:
            suffix: 文件后缀（如 ".wav"）
            prefix: 文件前缀（默认 "tmp_"）
        """
        self.suffix = suffix
        self.prefix = prefix
        self.temp_path: Optional[str] = None
    
    def __enter__(self):
        """进入上下文管理器，创建临时文件"""
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=self.suffix,
            prefix=self.prefix
        )
        self.temp_path = tmp_file.name
        tmp_file.close()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，清理临时文件"""
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
            except Exception as e:
                logger.warning(f"[TempFileManager] 清理临时文件失败: {e}")
    
    def get_path(self) -> str:
        """获取临时文件路径"""
        if self.temp_path is None:
            raise RuntimeError("临时文件尚未创建")
        return self.temp_path
    
    async def save_upload_file(self, upload_file: UploadFile) -> str:
        """
        保存上传的文件到临时文件
        
        Args:
            upload_file: FastAPI UploadFile 对象
            
        Returns:
            临时文件路径
        """
        if self.temp_path is None:
            raise RuntimeError("临时文件尚未创建")
        
        contents = await upload_file.read()
        with open(self.temp_path, "wb") as f:
            f.write(contents)
        
        return self.temp_path


def validate_file_format(filename: str, allowed_extensions: Tuple[str, ...]) -> bool:
    """
    检查文件名是否以允许的扩展名之一结尾
    
    Args:
        filename: 文件名
        allowed_extensions: 允许的扩展名元组（如 ('.wav', '.WAV')）
        
    Returns:
        若文件名以任一允许的扩展名结尾返回 True，否则返回 False
    """
    if not filename:
        return False
    return filename.lower().endswith(allowed_extensions)


def parse_speaker_list(speakers_str: Optional[str]) -> Optional[list]:
    """
    解析说话人列表字符串
    
    Args:
        speakers_str: 逗号分隔的说话人名称字符串（如 "doctor,patient"）
        
    Returns:
        说话人名称列表，如果输入为空则返回 None
    """
    if not speakers_str:
        return None
    
    speakers = [s.strip() for s in speakers_str.split(",") if s.strip()]
    return speakers if speakers else None

