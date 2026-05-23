"""
说话人注册管理服务

功能：
1. 说话人声纹注册 - 从音频中提取embedding并保存
2. 说话人列表查询 - 获取所有已注册说话人
3. 说话人删除 - 删除指定说话人的声纹文件
4. 说话人加载 - 加载已注册说话人的embedding
5. 说话人验证 - 验证音频是否为指定说话人

说明：
- 每个说话人的embedding保存为单独的.npy文件
- 文件命名格式：{speaker_name}.npy
- 存储路径：配置文件中的EMBEDDING_DIR
- 推荐注册音频时长：≥15秒，采样率：16kHz
"""

import warnings
from typing import Dict, Optional, Any
import numpy as np

warnings.filterwarnings('ignore')

from ..utils.embedding_file_manager import EmbeddingFileManager
from ..utils.audio_file_handler import AudioFileHandler
from ..utils.audio_constants import AudioConstants


class SpeakerEnrollmentService:
    """
    说话人注册管理服务类
    
    提供说话人声纹的注册、查询、删除、加载、验证功能
    """
    
    def __init__(self, model_manager, embedding_dir: Optional[str] = None):
        """
        初始化说话人注册管理器
        
        Args:
            model_manager: ModelManager实例
            embedding_dir: embedding存储目录（如果不指定，从配置文件读取）
        """
        self.model_manager = model_manager
        
        # 设置embedding存储目录
        if embedding_dir is None:
            from config import EMBEDDING_DIR
            embedding_dir = EMBEDDING_DIR
        
        # 使用 EmbeddingFileManager 管理文件操作
        self.file_manager = EmbeddingFileManager(embedding_dir)
        
        from ..utils.logger_manager import LoggerManager
        self.logger = LoggerManager.get_backend_logger()
        self.logger.info(f"[SpeakerEnrollmentService] 初始化完成")
        self.logger.info(f"[SpeakerEnrollmentService] Embedding存储目录: {self.file_manager.get_embedding_dir()}")
    
    def enroll(
        self,
        audio_path: str,
        speaker_name: str,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        注册说话人声纹
        
        从音频文件中提取说话人特征向量（embedding），并保存为.npy文件
        
        Args:
            audio_path: 音频文件路径（支持WAV等格式）
            speaker_name: 说话人名称（作为标识符）
            overwrite: 是否覆盖已存在的说话人（默认False）
        
        Returns:
            结果字典：
            {
                "success": True/False,
                "message": "注册成功/失败信息",
                "speaker_name": "说话人名称",
                "embedding_path": "embedding文件保存路径",
                "audio_duration": 音频时长（秒）,
                "embedding_dim": embedding维度
            }
        
        Raises:
            FileNotFoundError: 如果音频文件不存在
            ValueError: 如果说话人名称不合法或已存在（且overwrite=False）
        """
        # 检查说话人名称是否合法
        if not speaker_name or not speaker_name.strip():
            raise ValueError("说话人名称不能为空")
        
        speaker_name = speaker_name.strip()
        
        # 检查是否已存在
        if self.file_manager.exists(speaker_name) and not overwrite:
            raise ValueError(f"说话人 '{speaker_name}' 已存在，如需覆盖请设置overwrite=True")
        
        try:
            # 获取embedding提取处理器
            from ..processors.speaker_processor import SpeakerEmbeddingProcessor
            embedding_processor = SpeakerEmbeddingProcessor(self.model_manager)
            
            # 使用 AudioFileHandler 加载音频
            self.logger.info(f"[注册] 加载音频: {audio_path}")
            audio, sample_rate = AudioFileHandler.load_audio(audio_path)
            
            # 检查采样率
            if not AudioFileHandler.validate_sample_rate(sample_rate):
                self.logger.warning(f"[警告] 音频采样率为{sample_rate}Hz，建议使用16kHz音频")
            
            # 计算并验证音频时长
            duration = AudioFileHandler.calculate_duration(audio, sample_rate)
            self.logger.info(f"[注册] 音频时长: {duration:.2f}秒")
            
            is_valid, warning_msg = AudioFileHandler.validate_duration(duration)
            if not is_valid:
                raise ValueError(warning_msg)
            if warning_msg:
                self.logger.warning(f"[警告] {warning_msg}")
            
            # 提取embedding
            self.logger.info(f"[注册] 提取说话人特征...")
            embedding = embedding_processor.extract_embedding(audio)
            
            # 使用 EmbeddingFileManager 保存embedding
            embedding_path = self.file_manager.save_embedding(speaker_name, embedding)
            self.logger.info(f"[注册] 保存embedding到: {embedding_path}")
            
            result = {
                "success": True,
                "message": f"说话人 '{speaker_name}' 注册成功",
                "speaker_name": speaker_name,
                "embedding_path": embedding_path,
                "audio_duration": float(duration),
                "embedding_dim": int(embedding.shape[0])
            }
            
            self.logger.info(f"[注册] ✅ 成功！speaker={speaker_name}, dim={embedding.shape[0]}")
            return result
            
        except Exception as e:
            error_msg = f"注册失败: {str(e)}"
            self.logger.error(f"[注册] ❌ {error_msg}")
            return {
                "success": False,
                "message": error_msg,
                "speaker_name": speaker_name,
                "embedding_path": None,
                "audio_duration": None,
                "embedding_dim": None
            }
    
    def list_speakers(self) -> Dict[str, Any]:
        """
        获取所有已注册说话人列表
        
        Returns:
            结果字典：
            {
                "success": True,
                "speakers": ["speaker1", "speaker2", ...],
                "count": 说话人数量,
                "embedding_dir": embedding存储目录
            }
        """
        try:
            # 使用 EmbeddingFileManager 查询说话人列表
            speakers = self.file_manager.list_speakers()
            
            result = {
                "success": True,
                "speakers": speakers,
                "count": len(speakers),
                "embedding_dir": self.file_manager.get_embedding_dir()
            }
            
            self.logger.info(f"[查询] 共有 {len(speakers)} 个已注册说话人")
            return result
            
        except Exception as e:
            error_msg = f"查询失败: {str(e)}"
            self.logger.error(f"[查询] ❌ {error_msg}")
            return {
                "success": False,
                "speakers": [],
                "count": 0,
                "embedding_dir": self.file_manager.get_embedding_dir(),
                "error": error_msg
            }
    
    def delete_speaker(self, speaker_name: str) -> Dict[str, Any]:
        """
        删除指定说话人
        
        Args:
            speaker_name: 说话人名称
        
        Returns:
            结果字典：
            {
                "success": True/False,
                "message": "删除成功/失败信息",
                "speaker_name": "说话人名称"
            }
        """
        try:
            # 使用 EmbeddingFileManager 删除文件
            if not self.file_manager.delete_embedding(speaker_name):
                raise FileNotFoundError(f"说话人 '{speaker_name}' 不存在")
            
            result = {
                "success": True,
                "message": f"说话人 '{speaker_name}' 已删除",
                "speaker_name": speaker_name
            }
            
            self.logger.info(f"[删除] ✅ 已删除说话人: {speaker_name}")
            return result
            
        except Exception as e:
            error_msg = f"删除失败: {str(e)}"
            self.logger.error(f"[删除] ❌ {error_msg}")
            return {
                "success": False,
                "message": error_msg,
                "speaker_name": speaker_name
            }
    
    def load_speaker(self, speaker_name: str) -> Optional[np.ndarray]:
        """
        加载指定说话人的embedding
        
        Args:
            speaker_name: 说话人名称
        
        Returns:
            embedding: numpy数组，如果不存在则返回None
        """
        try:
            # 使用 EmbeddingFileManager 加载embedding
            embedding = self.file_manager.load_embedding(speaker_name)
            
            if embedding is None:
                self.logger.warning(f"[加载] 说话人 '{speaker_name}' 不存在")
            else:
                self.logger.info(f"[加载] 成功加载 '{speaker_name}', 维度: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"[加载] 失败: {e}")
            return None
    
    def load_all_speakers(self) -> Dict[str, np.ndarray]:
        """
        加载所有已注册说话人的embedding
        
        Returns:
            字典：{speaker_name: embedding}
        """
        speakers_info = self.list_speakers()
        
        if not speakers_info["success"]:
            self.logger.error(f"[加载全部] 失败: {speakers_info.get('error', '未知错误')}")
            return {}
        
        embeddings = {}
        for speaker_name in speakers_info["speakers"]:
            emb = self.load_speaker(speaker_name)
            if emb is not None:
                embeddings[speaker_name] = emb
        
        self.logger.info(f"[加载全部] 成功加载 {len(embeddings)} 个说话人")
        return embeddings
    
    def verify_speaker(
        self,
        audio_path: str,
        speaker_name: str,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        验证音频是否为指定说话人
        
        Args:
            audio_path: 音频文件路径
            speaker_name: 要验证的说话人名称
            threshold: 相似度阈值（默认使用recognition的阈值）
        
        Returns:
            结果字典：
            {
                "success": True/False,
                "speaker_name": "说话人名称",
                "is_match": True/False (是否匹配),
                "similarity": 相似度分数,
                "threshold": 使用的阈值
            }
        """
        try:
            # 获取embedding提取处理器
            from ..processors.speaker_processor import SpeakerEmbeddingProcessor
            embedding_processor = SpeakerEmbeddingProcessor(self.model_manager)
            
            # 加载已注册的embedding
            registered_emb = self.load_speaker(speaker_name)
            if registered_emb is None:
                raise ValueError(f"说话人 '{speaker_name}' 未注册")
            
            # 使用 AudioFileHandler 加载测试音频
            audio, sample_rate = AudioFileHandler.load_audio(audio_path)
            
            # 提取测试音频的embedding
            test_emb = embedding_processor.extract_embedding(audio)
            
            # 计算相似度
            similarity = embedding_processor.compute_similarity(test_emb, registered_emb)
            
            # 判断是否匹配
            if threshold is None:
                # 从clustering processor获取默认阈值
                from ..processors.speaker_processor import SpeakerClusteringProcessor
                clustering_processor = SpeakerClusteringProcessor(self.model_manager)
                threshold = clustering_processor.similarity_threshold
            
            is_match = similarity >= threshold
            
            result = {
                "success": True,
                "speaker_name": speaker_name,
                "is_match": is_match,
                "similarity": float(similarity),
                "threshold": float(threshold)
            }
            
            match_str = "✅ 匹配" if is_match else "❌ 不匹配"
            self.logger.info(f"[验证] {match_str} speaker={speaker_name}, similarity={similarity:.3f}, threshold={threshold}")
            
            return result
            
        except Exception as e:
            error_msg = f"验证失败: {str(e)}"
            self.logger.error(f"[验证] ❌ {error_msg}")
            return {
                "success": False,
                "speaker_name": speaker_name,
                "is_match": False,
                "similarity": 0.0,
                "threshold": threshold or 0.5,
                "error": error_msg
            }


if __name__ == "__main__":
    # 简单测试
    from ..utils.logger_manager import LoggerManager
    logger = LoggerManager.init_backend_logger()
    logger.info("说话人注册管理服务测试")
    logger.info("=" * 60)
    
    from ..models.model_manager import ModelManager
    
    manager = ModelManager()
    enrollment = SpeakerEnrollmentService(manager)
    
    # 测试查询列表
    result = enrollment.list_speakers()
    logger.info(f"\n当前已注册说话人: {result}")

