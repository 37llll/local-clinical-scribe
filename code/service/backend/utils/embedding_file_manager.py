"""
Embedding 文件管理器

功能：
- 管理说话人 embedding 文件的存储和读取
- 文件系统操作（检查、保存、删除、查询）
- 目录管理
"""

import os
import numpy as np
from typing import List, Optional
from pathlib import Path
from .logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class EmbeddingFileManager:
    """
    Embedding 文件管理器
    
    功能：
    - 管理 embedding 文件的存储路径
    - 提供文件操作接口（保存、加载、删除、查询）
    - 确保目录存在
    """
    
    def __init__(self, embedding_dir: str):
        """
        初始化 Embedding 文件管理器
        
        Args:
            embedding_dir: embedding 存储目录路径
        """
        self.embedding_dir = Path(embedding_dir)
        # 确保目录存在
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
    
    def get_embedding_path(self, speaker_name: str) -> Path:
        """
        获取指定说话人的 embedding 文件路径
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            embedding 文件路径
        """
        return self.embedding_dir / f"{speaker_name}.npy"
    
    def exists(self, speaker_name: str) -> bool:
        """
        检查说话人 embedding 文件是否存在
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            如果文件存在返回 True，否则返回 False
        """
        return self.get_embedding_path(speaker_name).exists()
    
    def save_embedding(self, speaker_name: str, embedding: np.ndarray) -> str:
        """
        保存 embedding 到文件
        
        Args:
            speaker_name: 说话人名称
            embedding: embedding 向量（numpy数组）
            
        Returns:
            保存的文件路径（字符串）
        """
        embedding_path = self.get_embedding_path(speaker_name)
        np.save(embedding_path, embedding)
        return str(embedding_path)
    
    def load_embedding(self, speaker_name: str) -> Optional[np.ndarray]:
        """
        加载指定说话人的 embedding
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            embedding 向量（numpy数组），如果文件不存在返回 None
        """
        embedding_path = self.get_embedding_path(speaker_name)
        
        if not embedding_path.exists():
            return None
        
        try:
            embedding = np.load(embedding_path)
            # 确保是一维数组（处理可能的二维数组情况）
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            return embedding
        except Exception as e:
            logger.error(f"[EmbeddingFileManager] 加载失败: {e}")
            return None
    
    def delete_embedding(self, speaker_name: str) -> bool:
        """
        删除指定说话人的 embedding 文件
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            如果删除成功返回 True，否则返回 False
        """
        embedding_path = self.get_embedding_path(speaker_name)
        
        if not embedding_path.exists():
            return False
        
        try:
            embedding_path.unlink()
            return True
        except Exception as e:
            logger.error(f"[EmbeddingFileManager] 删除失败: {e}")
            return False
    
    def list_speakers(self) -> List[str]:
        """
        列出所有已注册的说话人名称
        
        Returns:
            说话人名称列表（已排序）
        """
        npy_files = list(self.embedding_dir.glob("*.npy"))
        speakers = [f.stem for f in npy_files]
        speakers.sort()
        return speakers
    
    def get_embedding_dir(self) -> str:
        """
        获取 embedding 存储目录路径
        
        Returns:
            目录路径（字符串）
        """
        return str(self.embedding_dir)

