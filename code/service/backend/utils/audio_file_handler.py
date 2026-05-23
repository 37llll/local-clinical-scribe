"""
音频文件处理工具

功能：
- 加载音频文件
- 音频格式转换（立体声转单声道、采样率检查）
- 音频时长验证
"""

import os
import numpy as np
import soundfile as sf
from typing import Tuple, Optional
from .audio_constants import AudioConstants


class AudioFileHandler:
    """
    音频文件处理器
    
    功能：
    - 加载音频文件
    - 音频格式转换和验证
    - 音频时长检查
    """
    
    @staticmethod
    def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            (audio, sample_rate) 元组
            - audio: 音频数据（numpy数组，float32，单声道）
            - sample_rate: 采样率
            
        Raises:
            FileNotFoundError: 如果音频文件不存在
            ValueError: 如果音频文件格式不正确
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        audio, sample_rate = sf.read(audio_path)
        
        # 如果是立体声，转换为单声道
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # 转换为float32
        audio = audio.astype(np.float32)
        
        return audio, sample_rate
    
    @staticmethod
    def validate_sample_rate(sample_rate: int, target_rate: int = None) -> bool:
        """
        验证采样率
        
        Args:
            sample_rate: 实际采样率
            target_rate: 目标采样率（默认使用 AudioConstants.SAMPLE_RATE）
            
        Returns:
            如果采样率匹配返回 True，否则返回 False
        """
        if target_rate is None:
            target_rate = AudioConstants.SAMPLE_RATE
        
        return sample_rate == target_rate
    
    @staticmethod
    def calculate_duration(audio: np.ndarray, sample_rate: int) -> float:
        """
        计算音频时长
        
        Args:
            audio: 音频数据
            sample_rate: 采样率
            
        Returns:
            音频时长（秒）
        """
        return len(audio) / sample_rate
    
    @staticmethod
    def validate_duration(duration: float, min_duration: float = 1.0) -> Tuple[bool, Optional[str]]:
        """
        验证音频时长
        
        Args:
            duration: 音频时长（秒）
            min_duration: 最小时长（秒）
            
        Returns:
            (is_valid, warning_message) 元组
            - is_valid: 如果时长有效返回 True
            - warning_message: 警告消息（如果有）
        """
        if duration < min_duration:
            return False, f"音频时长过短({duration:.2f}秒)，建议至少{min_duration}秒"
        
        if duration < 15.0:
            return True, f"音频时长较短({duration:.2f}秒)，建议至少15秒以获得更好的声纹特征"
        
        return True, None

