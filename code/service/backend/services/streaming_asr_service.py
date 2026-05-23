"""
流式ASR服务层

职责：提供流式ASR的业务接口，管理会话生命周期
- 管理每个WebSocket连接的缓存状态
- 提供统一的对外接口
- 处理业务逻辑和错误处理
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..models.model_manager import ModelManager
from ..processors.streaming_processor import StreamingASRProcessor
from ..utils.streaming_utils import CacheInitializer


class StreamingASRService:
    """
    流式ASR服务类
    
    职责：
    - 管理流式ASR处理器的生命周期
    - 为每个WebSocket连接维护独立的缓存状态
    - 提供统一的业务接口
    """
    
    def __init__(self, model_manager: ModelManager):
        """
        初始化流式ASR服务
        
        Args:
            model_manager: 模型管理器实例
        """
        self.model_manager = model_manager
        self.processor = StreamingASRProcessor(model_manager)
    
    def create_session_cache(self) -> Dict[str, Any]:
        """
        创建新的会话缓存
        
        每个WebSocket连接应该调用此方法创建独立的缓存状态
        
        Returns:
            初始化后的缓存字典
        """
        cache: Dict[str, Any] = {}
        CacheInitializer.ensure_state(cache)
        return cache
    
    def reset_session_cache(self, cache: Dict[str, Any]):
        """
        重置会话缓存（清空所有累积结果）
        
        Args:
            cache: 要重置的缓存字典
        """
        CacheInitializer.ensure_state(cache)
        # 清空累积结果
        cache["streaming_asr_results"] = []
        cache["offline_asr_segments"] = []
        cache["segments"] = []
        cache["audio_buffer"] = []
        cache["samples_processed"] = 0
        cache["chunk_counter"] = 0
    
    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        cache: Dict[str, Any],
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        处理单个音频chunk（完整流程：VAD + 流式ASR + 离线ASR + 对齐）
        
        Args:
            audio_chunk: 输入音频chunk (推荐200ms, 3200采样点 @ 16kHz)
            cache: 会话缓存（由 create_session_cache 创建）
            enable_speaker_diarization: 是否启用说话人识别（默认True）
            speaker_mode: 说话人识别模式（默认"cluster"）
                - "cluster": 仅聚类，自动识别说话人数量
                - "cluster_match": 聚类后与已注册说话人匹配
                - "direct_match": 每个片段直接匹配最相似的说话人
            registered_speakers: 已注册说话人名称列表（用于匹配模式）
            similarity_threshold: 相似度阈值（默认0.5，范围0-1）
        
        Returns:
            处理结果字典，包含：
            - vad_raw: VAD原始输出
            - new_segments_count: 本次新完成的VAD片段数
            - streaming_asr_history: 流式ASR历史（所有识别结果）
            - offline_asr_segments: 离线ASR片段（高精度+说话人）
            - aligned_text: 对齐后的最终文本（推荐使用）
            - session_stats: 会话统计信息
        """
        return self.processor.pipeline(
            audio_chunk=audio_chunk,
            cache=cache,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold
        )
    
    def process_chunk_streaming_only(
        self,
        audio_chunk: np.ndarray,
        cache: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理单个音频chunk（仅流式流程：VAD + 流式ASR，不包含离线ASR）
        
        适用于实时低延迟场景，不需要高精度识别和说话人识别
        
        Args:
            audio_chunk: 输入音频chunk
            cache: 会话缓存
        
        Returns:
            处理结果字典，包含：
            - vad: VAD原始输出
            - streaming_asr: 流式ASR结果（如果触发）
        """
        return self.processor.streaming_pipeline(audio_chunk=audio_chunk, cache=cache)
    
    def get_session_stats(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取会话统计信息
        
        Args:
            cache: 会话缓存
        
        Returns:
            统计信息字典
        """
        return {
            "total_audio_duration": cache.get("samples_processed", 0) / 16000.0,
            "vad_segments_count": len(cache.get("segments", [])),
            "streaming_asr_count": len(cache.get("streaming_asr_results", [])),
            "offline_asr_count": len(cache.get("offline_asr_segments", [])),
            "chunk_counter": cache.get("chunk_counter", 0)
        }
    
    def get_aligned_text(self, cache: Dict[str, Any]) -> str:
        """
        获取对齐后的最终文本
        
        Args:
            cache: 会话缓存
        
        Returns:
            对齐后的完整文本
        """
        return self.processor._align_results(cache)
    
    def get_streaming_history(self, cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取流式ASR历史结果
        
        Args:
            cache: 会话缓存
        
        Returns:
            流式ASR结果列表
        """
        return cache.get("streaming_asr_results", [])
    
    def get_offline_segments(self, cache: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        获取离线ASR片段列表
        
        Args:
            cache: 会话缓存
        
        Returns:
            离线ASR片段列表（包含说话人信息）
        """
        return cache.get("offline_asr_segments", [])
    
    def finalize(
        self,
        cache: Dict[str, Any],
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        处理音频结尾，确保所有pending的片段和最后一个窗口都被处理
        
        功能：
        1. 检查是否有pending_start_ms（未完成的VAD片段）
        2. 如果有，将其转换为完整片段（从pending_start_ms到音频结束）
        3. 处理最后一个窗口（如果有剩余的音频和片段）
        4. 返回最终结果
        
        应该在WebSocket断开时调用此方法，确保所有音频都被处理。
        
        Args:
            cache: 会话缓存
            enable_speaker_diarization: 是否启用说话人识别（默认True）
            speaker_mode: 说话人识别模式（默认"cluster"）
            registered_speakers: 已注册说话人名称列表
            similarity_threshold: 相似度阈值（默认0.5）
        
        Returns:
            最终处理结果（格式与process_chunk相同）
        """
        return self.processor.finalize(
            cache=cache,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold
        )

