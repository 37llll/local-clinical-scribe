"""
离线ASR处理器模块

功能：
- VAD片段提取和处理
- 片段ASR识别和标点恢复
- 完整的离线ASR处理流程（支持说话人识别）

包含三个主要类：
1. VADSegmentProcessor - VAD片段提取
2. SegmentASRProcessor - 片段ASR识别
3. OfflineASRProcessor - 完整的离线ASR处理流程
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    from funasr.utils.vad_utils import merge_vad, slice_padding_audio_samples
except ModuleNotFoundError:
    merge_vad = None
    slice_padding_audio_samples = None

from ..models.model_manager import ModelManager
from .speaker_processor import (
    SpeakerDiarizationProcessor,
)
from ..utils.audio_constants import AudioConstants
from ..utils.logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class VADSegmentProcessor:
    """
    VAD 片段处理器（离线）
    
    功能：
    - 从完整音频中提取VAD片段
    - 合并和切片音频片段
    """

    def __init__(self, model_manager: ModelManager):
        self.vad_model = model_manager.get_vad_model()

    def extract_segments(
        self,
        audio: np.ndarray,
        max_segment_length_ms: int = 15000,
        min_segment_length_ms: int = 0,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        从音频中提取 VAD 片段
        
        Args:
            audio: 完整音频数据
            max_segment_length_ms: 最大片段长度（毫秒）
            min_segment_length_ms: 最小片段长度（毫秒）
            
        Returns:
            (speech_list, time_segments) 元组：
            - speech_list: 提取的音频片段列表
            - time_segments: 对应的时间段列表（采样点单位）
        """
        logger.info("[VAD] 开始VAD检测...")
        vad_result = self.vad_model.generate(input=audio, disable_pbar=True)

        vad_segments_ms = self._parse_vad_output(vad_result)
        if not vad_segments_ms:
            logger.warning("[VAD] 未检测到有效语音片段")
            return [], []

        logger.info(f"[VAD] 原始检测到 {len(vad_segments_ms)} 个片段")

        if merge_vad is None:
            raise ModuleNotFoundError(
                "FunASR is required for offline VAD. Install runtime dependencies "
                "before using audio endpoints."
            )

        merged_segments_ms = merge_vad(
            vad_segments_ms,
            max_length=max_segment_length_ms,
            min_length=min_segment_length_ms,
        )
        logger.info(f"[VAD] merge_vad后: {len(merged_segments_ms)} 个片段")

        speech_list, time_segments = self._slice_audio_by_vad(audio, merged_segments_ms)
        logger.info(f"[VAD] 音频切片完成，共 {len(speech_list)} 个片段")
        return speech_list, time_segments

    def _parse_vad_output(self, vad_result: Any) -> List[List[int]]:
        """
        解析 VAD 模型输出，返回毫秒段列表
        
        Args:
            vad_result: VAD模型返回的结果
            
        Returns:
            毫秒段列表，格式：[[start_ms, end_ms], ...]
        """
        if not vad_result or len(vad_result) == 0:
            return []

        vad_item = vad_result[0] if isinstance(vad_result, list) else vad_result
        if not isinstance(vad_item, dict):
            return []

        vad_data = vad_item.get("value", [])
        if not isinstance(vad_data, list):
            return []

        return vad_data

    def _slice_audio_by_vad(
        self, audio: np.ndarray, segments_ms: List[List[int]]
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        根据 VAD 时间戳切片音频
        
        Args:
            audio: 完整音频数据
            segments_ms: VAD片段时间戳列表（毫秒）
            
        Returns:
            (speech_list, time_segments) 元组：
            - speech_list: 切片后的音频片段列表
            - time_segments: 对应的时间段列表（采样点单位）
        """
        if not segments_ms:
            return [], []

        if slice_padding_audio_samples is None:
            raise ModuleNotFoundError(
                "FunASR is required for audio slicing. Install runtime dependencies "
                "before using audio endpoints."
            )

        vad_segments_for_slice = [((seg[0], seg[1]),) for seg in segments_ms]

        speech_list, _ = slice_padding_audio_samples(
            audio, len(audio), vad_segments_for_slice
        )

        time_segments_samples = [
            (int(seg[0] * AudioConstants.MS_TO_SAMPLES), int(seg[1] * AudioConstants.MS_TO_SAMPLES))
            for seg in segments_ms
        ]

        return speech_list, time_segments_samples


class SegmentASRProcessor:
    """
    片段 ASR 处理器（离线）
    
    功能：
    - 对单个音频片段进行ASR识别
    - 标点符号恢复
    """

    def __init__(self, model_manager: ModelManager):
        self.asr_model = model_manager.get_asr_offline_model()
        self.punc_model = model_manager.get_punc_model()
        self.model_manager = model_manager
        self._text_corrector = None

    @property
    def text_corrector(self):
        """懒加载文本纠错器"""
        if self._text_corrector is None:
            try:
                self._text_corrector = self.model_manager.get_text_corrector()
            except Exception as e:
                logger.warning(f"[ASR] 文本纠错器加载失败: {e}")
                self._text_corrector = None
        return self._text_corrector

    def process_segment(
        self, 
        audio_segment: np.ndarray,
        enable_text_correction: bool = True  # 【新增参数】
    ) -> str:
        """
        处理单个片段：ASR + 标点 + 文本纠错
        
        Args:
            audio_segment: 音频片段数据
            enable_text_correction: 是否启用文本纠错（默认True）
            
        Returns:
            识别后的文本（带标点符号，纠错后）
        """
        text = self._run_asr(audio_segment)
        if not text:
            return ""
        text = self._run_punctuation(text)
        
        # 【新增】文本纠错
        if enable_text_correction and self.text_corrector:
            text = self._run_text_correction(text)
        
        return text

    def _run_asr(self, audio: np.ndarray) -> str:
        """
        运行ASR识别
        
        Args:
            audio: 音频数据
            
        Returns:
            识别文本（不带标点）
        """
        try:
            result = self.asr_model.generate(input=audio, disable_pbar=True)
            if result and len(result) > 0:
                first_result = result[0] if isinstance(result, list) else result
                if isinstance(first_result, dict):
                    return first_result.get("text", "")
        except Exception as e:
            logger.error(f"[ASR] 识别失败: {e}")
        return ""

    def _run_punctuation(self, text: str) -> str:
        """
        运行标点恢复
        
        Args:
            text: 原始文本（不带标点）
            
        Returns:
            带标点的文本
        """
        if not text or not self.punc_model:
            return text

        text_no_space = text.replace(" ", "")
        if not text_no_space:
            return text

        try:
            result = self.punc_model.generate(input=text_no_space, disable_pbar=True)
            if result and len(result) > 0:
                first_result = result[0] if isinstance(result, list) else result
                if isinstance(first_result, dict):
                    return first_result.get("text", text)
        except Exception as e:
            logger.error(f"[PUNC] 标点恢复失败: {e}")
        return text

    def _run_text_correction(self, text: str) -> str:
        """
        运行文本纠错
        
        Args:
            text: 原始文本（带标点）
            
        Returns:
            纠错后的文本
        """
        if not text or not self.text_corrector:
            return text
        
        text_no_space = text.replace(" ", "")
        if not text_no_space:
            return text
        
        try:
            corrected = self.text_corrector.correct(text)
            
            # 如果有错误纠正，记录日志
            if corrected != text:
                logger.info(f"[TEXT_CORRECTION] 原文: {text}")
                logger.info(f"[TEXT_CORRECTION] 纠错: {corrected}")
            
            return corrected
        except Exception as e:
            logger.error(f"[TEXT_CORRECTION] 文本纠错失败: {e}")
            return text


class OfflineASRProcessor:
    """
    离线 ASR 处理器（主控制器）
    
    功能：
    - 标准模式：FunASR 组合模型（VAD+ASR+PUNC）
    - 说话人识别模式：手动 VAD + ASR + Speaker Diarization
    
    支持两种处理模式：
    1. 标准模式：使用FunASR组合模型，快速处理
    2. 说话人识别模式：手动VAD分段，然后ASR识别和说话人识别
    """

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.combined_model = model_manager.get_asr_offline_with_vad_punc()

        self._vad_processor: Optional[VADSegmentProcessor] = None
        self._asr_processor: Optional[SegmentASRProcessor] = None
        self._speaker_processor: Optional[SpeakerDiarizationProcessor] = None

    @property
    def vad_processor(self) -> VADSegmentProcessor:
        if self._vad_processor is None:
            self._vad_processor = VADSegmentProcessor(self.model_manager)
        return self._vad_processor

    @property
    def asr_processor(self) -> SegmentASRProcessor:
        if self._asr_processor is None:
            self._asr_processor = SegmentASRProcessor(self.model_manager)
        return self._asr_processor

    @property
    def speaker_processor(self) -> SpeakerDiarizationProcessor:
        if self._speaker_processor is None:
            self._speaker_processor = SpeakerDiarizationProcessor(self.model_manager)
        return self._speaker_processor

    def process(
        self,
        audio: np.ndarray,
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        enable_text_correction: bool = True,  # 【新增参数】
    ) -> Dict[str, Any]:
        """
        处理完整音频文件
        
        Args:
            audio: 完整音频数据
            enable_speaker_diarization: 是否启用说话人识别
            speaker_mode: 说话人识别模式（cluster/cluster_match/direct_match）
            registered_speakers: 已注册说话人名称列表
            similarity_threshold: 相似度阈值
            
        Returns:
            处理结果字典，包含：
            - text: 完整识别文本
            - duration: 音频时长（秒）
            - segments: 片段列表（如果启用说话人识别）
        """
        duration = len(audio) / AudioConstants.SAMPLE_RATE
        logger.info(f"[OfflineASR] 开始处理音频: {len(audio)}采样点 ({duration:.2f}秒)")

        if not enable_speaker_diarization:
            return self._process_standard_mode(audio, duration)

        logger.info("[OfflineASR] 说话人识别模式：开始VAD分段...")
        speech_list, time_segments = self.vad_processor.extract_segments(
            audio,
            max_segment_length_ms=15000,
            min_segment_length_ms=0,
        )

        if not speech_list:
            logger.warning("[OfflineASR] 警告：VAD未检测到语音片段，降级为标准模式")
            return self._process_standard_mode(audio, duration)

        # 记录离线VAD检测到的片段时间戳（用于对比流式VAD）
        logger.info(f"[OfflineASR] 离线VAD检测到 {len(speech_list)} 个片段:")
        for i, (start_sample, end_sample) in enumerate(time_segments, 1):
            seg_start_s = start_sample / AudioConstants.SAMPLE_RATE
            seg_end_s = end_sample / AudioConstants.SAMPLE_RATE
            seg_duration = (end_sample - start_sample) / AudioConstants.SAMPLE_RATE
            logger.info(f"[OfflineASR]   片段{i}: [{seg_start_s:.2f}s - {seg_end_s:.2f}s] ({seg_duration:.2f}s)")

        logger.info(f"[OfflineASR] 开始ASR识别，共{len(speech_list)}个片段...")
        segments = self._process_segments_asr(speech_list, time_segments)

        if not segments:
            logger.warning("[OfflineASR] 警告：ASR未识别出文本，降级为标准模式")
            return self._process_standard_mode(audio, duration)

        full_text = "".join([seg["text"] for seg in segments])
        logger.info(f"[OfflineASR] ASR完成: {len(segments)}个片段, {len(full_text)}个字符")

        logger.info("[OfflineASR] 开始说话人识别...")
        segments = self.speaker_processor.process_segments(
            audio=audio,
            segments=segments,
            mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold,
        )

        self._print_result_summary(full_text, segments)

        return {
            "text": full_text,
            "duration": duration,
            "segments": segments,
        }

    def _process_standard_mode(
        self, audio: np.ndarray, duration: float
    ) -> Dict[str, Any]:
        """
        标准模式处理：使用FunASR组合模型
        
        Args:
            audio: 完整音频数据
            duration: 音频时长（秒）
            
        Returns:
            处理结果字典
        """
        logger.info("[OfflineASR] 标准模式：调用FunASR组合模型...")
        result = self.combined_model.generate(input=audio, disable_pbar=True)

        full_text = ""
        if result and len(result) > 0:
            first_result = result[0] if isinstance(result, list) else result
            if isinstance(first_result, dict):
                full_text = first_result.get("text", "")

        logger.info(f"[OfflineASR] 识别完成: {len(full_text)}个字符")
        if full_text:
            preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
            logger.info(f"[OfflineASR] 文本预览: {preview}")

        return {"text": full_text, "duration": duration, "segments": None}

    def _process_segments_asr(
        self, speech_list: List[np.ndarray], time_segments: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """
        处理所有片段的ASR识别
        
        Args:
            speech_list: 音频片段列表
            time_segments: 时间段列表
            
        Returns:
            识别结果片段列表
        """
        segments: List[Dict[str, Any]] = []
        for i, (audio_segment, (start_sample, end_sample)) in enumerate(
            zip(speech_list, time_segments)
        ):
            # 过滤异常片段：start_sample >= end_sample
            if start_sample >= end_sample:
                logger.warning(f"[OfflineASR] ⚠️ 跳过异常片段{i+1}: start_sample({start_sample}) >= end_sample({end_sample})")
                continue
            
            try:
                text = self.asr_processor.process_segment(
                    audio_segment, 
                    enable_text_correction=True  # 【新增参数】可通过配置控制
                )
                if text:
                    segments.append(
                        {
                            "start": start_sample / AudioConstants.SAMPLE_RATE,
                            "end": end_sample / AudioConstants.SAMPLE_RATE,
                            "text": text,
                            "start_sample": start_sample,
                            "end_sample": end_sample,
                        }
                    )

                    seg_duration = (end_sample - start_sample) / AudioConstants.SAMPLE_RATE
                    text_preview = text[:60] + "..." if len(text) > 60 else text
                    logger.info(
                        f"[OfflineASR] 片段{i+1}/{len(speech_list)}: "
                        f"{start_sample/AudioConstants.SAMPLE_RATE:.2f}s-{end_sample/AudioConstants.SAMPLE_RATE:.2f}s "
                        f"({seg_duration:.2f}s) | {text_preview}"
                    )
                else:
                    # ASR识别返回空文本，记录日志
                    seg_duration = (end_sample - start_sample) / AudioConstants.SAMPLE_RATE
                    logger.warning(
                        f"[OfflineASR] ⚠️ 片段{i+1}/{len(speech_list)}: "
                        f"{start_sample/AudioConstants.SAMPLE_RATE:.2f}s-{end_sample/AudioConstants.SAMPLE_RATE:.2f}s "
                        f"({seg_duration:.2f}s) | ASR识别返回空文本，跳过"
                    )
            except Exception as e:
                logger.warning(f"[OfflineASR] 警告：片段{i+1}处理失败: {e}")
                continue

        logger.info(f"[OfflineASR] ASR处理完成: {len(segments)}/{len(speech_list)}个片段成功")
        return segments

    def _print_result_summary(self, full_text: str, segments: List[Dict]):
        """
        打印处理结果摘要
        
        Args:
            full_text: 完整识别文本
            segments: 片段列表
        """
        logger.info("[OfflineASR] " + "=" * 50)
        logger.info("[OfflineASR] 处理完成!")
        logger.info(f"[OfflineASR] 总字符数: {len(full_text)}")
        logger.info(f"[OfflineASR] 总片段数: {len(segments)}")

        speakers = set(seg.get("speaker", "未知") for seg in segments)
        logger.info(f"[OfflineASR] 说话人数: {len(speakers)} ({', '.join(sorted(speakers))})")

        if full_text:
            preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
            logger.info(f"[OfflineASR] 文本预览: {preview}")

        logger.info("[OfflineASR] " + "=" * 50)
