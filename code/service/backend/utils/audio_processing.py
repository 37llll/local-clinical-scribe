"""
音频处理工具函数

功能：
- 音频重采样（支持任意采样率转换）
- 其他音频处理工具函数
"""

import numpy as np
import librosa
from .audio_constants import AudioConstants


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = None) -> np.ndarray:
    """
    将任意采样率音频重采样为目标采样率
    
    Args:
        audio: 原始音频数据（numpy数组）
        orig_sr: 原始采样率
        target_sr: 目标采样率（默认使用 AudioConstants.SAMPLE_RATE，即16kHz）
        
    Returns:
        重采样后的音频数据（numpy数组）
        
    Raises:
        ValueError: 如果音频数据为空或采样率无效
        
    Note:
        如果原始采样率等于目标采样率，直接返回原音频，不进行重采样
    """
    if target_sr is None:
        target_sr = AudioConstants.SAMPLE_RATE
    
    if len(audio) == 0:
        raise ValueError("音频数据为空")
    
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"无效的采样率: orig_sr={orig_sr}, target_sr={target_sr}")
    
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

