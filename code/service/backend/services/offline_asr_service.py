import soundfile as sf
import numpy as np
import librosa
import io
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

from ..models.model_manager import ModelManager
from ..processors.offline_processor import OfflineASRProcessor
from ..utils.audio_constants import AudioConstants
from ..utils.logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class OfflineASRService:
    """离线 ASR 服务：对外提供简化的业务接口"""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.processor = OfflineASRProcessor(self.model_manager)

    def process_audio_array(
        self,
        audio: np.ndarray,
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        enable_text_correction: bool = True,  # 【新增参数】
    ) -> Dict[str, Any]:
        """处理 numpy 音频数组"""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)

        return self.processor.process(
            audio=audio,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold,
            enable_text_correction=enable_text_correction,
        )

    def process_audio_file(
        self,
        audio_path: str,
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        enable_text_correction: bool = True,  # 【新增参数】
    ) -> Dict[str, Any]:
        """处理音频文件"""
        audio, sample_rate = sf.read(audio_path)
        if sample_rate != AudioConstants.SAMPLE_RATE:
            logger.warning(f"[OfflineASRService] 警告：采样率为 {sample_rate}Hz，建议使用16kHz")

        return self.process_audio_array(
            audio=audio,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold,
            enable_text_correction=enable_text_correction,
        )
    
    async def process_uploaded_file(
        self,
        upload_file: UploadFile,
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        enable_text_correction: bool = True,  # 【新增参数】
    ) -> Dict[str, Any]:
        """
        处理上传的音频文件
        
        Args:
            upload_file: FastAPI UploadFile 对象
            enable_speaker_diarization: 是否启用说话人识别
            speaker_mode: 说话人识别模式
            registered_speakers: 已注册说话人列表
            similarity_threshold: 相似度阈值
            
        Returns:
            处理结果字典
        """
        # 读取音频文件
        audio_bytes = await upload_file.read()
        
        # 使用librosa加载音频（自动处理各种格式）
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=AudioConstants.SAMPLE_RATE,
            mono=True
        )
        audio = audio.astype(np.float32)
        
        logger.info(f"[OfflineASRService] 收到音频文件: {upload_file.filename}, "
              f"时长={len(audio)/AudioConstants.SAMPLE_RATE:.2f}秒")

        return self.process_audio_array(
            audio=audio,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold,
            enable_text_correction=enable_text_correction,
        )
