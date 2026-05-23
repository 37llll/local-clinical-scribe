"""
流式ASR处理器

功能：
- 封装流式ASR处理的核心逻辑
- 调用各种模型（VAD、流式ASR、离线ASR）
- 管理VAD片段和音频缓存
- 提供完整的流式ASR处理管线

包含三个主要类：
1. ModelInvoker - 调用各种模型
2. SegmentManager - 管理VAD片段
3. StreamingASRProcessor - 流式ASR处理管线
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from ..models.model_manager import ModelManager
from .offline_processor import OfflineASRProcessor
from ..utils.audio_constants import AudioConstants
from ..utils.streaming_utils import (
    AudioBufferManager,
    CacheInitializer,
    SlidingWindowManager,
    VADResultParser,
)
from ..utils.text_extractor import TextExtractor
from ..utils.result_aligner import ResultAligner
from ..utils.offline_asr_trigger import OfflineASRTrigger
from ..utils.logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class ModelInvoker:
    """职责：调用各种模型"""
    def __init__(self, vad_model, asr_online_model, asr_offline_model):
        self.vad_model = vad_model
        self.asr_online_model = asr_online_model
        self.asr_offline_model = asr_offline_model
        # 为离线VAD保存的临时cache（每次调用前清空，避免流式状态干扰）
        self._offline_vad_cache = {}

    def invoke_streaming_vad(self, audio_chunk: np.ndarray, cache: Dict):
        """调用流式 VAD 模型"""
        vad_cache = cache.get("vad_cache") if isinstance(cache, dict) else cache
        return self.vad_model.generate(
            input=audio_chunk,
            cache=vad_cache,
            is_final=False,
            chunk_size=200,
            disable_pbar=True,
        )

    def invoke_offline_vad(self, audio_chunk: np.ndarray):
        """
        调用离线 VAD 模型
        
        注意：离线模式不传cache参数，不设置is_final参数，让模型一次性处理完整音频。
        返回格式：完整的语音片段列表 [[start, end], [start, end], ...]
        """
        return self.vad_model.generate(
            input=audio_chunk,
            disable_pbar=True
            # 不传cache，不传is_final，让模型使用离线模式
        )

    def invoke_streaming_asr(self, audio_chunk: np.ndarray, cache: Dict):
        """
        调用流式 ASR 模型
        
        缓存配置：
        - chunk_size=[0, 10, 5]: ASR窗口大小600ms (10帧 * 60ms)
        - encoder_chunk_look_back=4: 编码器缓存2.4秒 (4个chunk * 600ms)
        - decoder_chunk_look_back=1: 解码器缓存0.6秒 (1个chunk * 600ms)
        """
        return self.asr_online_model.generate(
            input=audio_chunk,
            cache=cache,
            is_final=False,
            chunk_size=[0, 10, 5],
            encoder_chunk_look_back=4,  # 2.4秒缓存 (4个chunk)
            decoder_chunk_look_back=1,  # 0.6秒缓存 (1个chunk)
            disable_pbar=True
        )

    def invoke_offline_asr(self, audio_chunk: np.ndarray):
        """调用离线 ASR 模型"""
        return self.asr_offline_model.generate(
            input=audio_chunk,
            chunk_size=200,
            disable_pbar=True
        )


class SegmentManager:
    """职责：管理 VAD 片段"""
    
    def update_from_vad(self, cache: Dict, vad_result) -> List[Tuple[int, int]]:
        """
        根据 VAD 输出更新已完成的语音片段，返回本轮新完成的片段
        
        Args:
            cache: 缓存字典
            vad_result: VAD 模型返回的结果
        
        ⚠️ 重要：流式VAD模型通过cache维护内部状态，返回的时间戳是**毫秒（ms）**，
        是从音频开始累计的绝对时间。需要转换为采样点：samples = ms * 16 (16kHz采样率)
        
        Returns:
            新完成的片段列表，格式：[(start_sample, end_sample), ...]
        """
        new_segments: List[Tuple[int, int]] = []
        items = VADResultParser.extract_items(vad_result)
        pending_ms = cache.get("pending_start_ms")  # 更名以明确单位

        for item in items:
            value = VADResultParser.extract_value(item)
            if not value:
                continue

            for pair in value:
                if VADResultParser.is_start_marker(pair):
                    # VAD返回的是毫秒（ms），存储为 pending_ms
                    pending_ms = pair[0]
                    logger.debug(f"[DEBUG] 检测到开始标记: VAD={pair[0]}ms ({pair[0]/1000:.2f}s)")
                elif VADResultParser.is_end_marker(pair):
                    end_ms = pair[1]
                    logger.debug(f"[DEBUG] 检测到结束标记: VAD={end_ms}ms, pending={pending_ms}ms")
                    
                    if pending_ms is not None:
                        # 转换毫秒为采样点（16kHz采样率：1ms = 16 samples）
                        start_sample = int(pending_ms * AudioConstants.MS_TO_SAMPLES)
                        end_sample = int(end_ms * AudioConstants.MS_TO_SAMPLES)
                        
                        if end_sample > start_sample:
                            cache["segments"].append((start_sample, end_sample))
                            new_segments.append((start_sample, end_sample))
                            duration_sec = (end_ms - pending_ms) / 1000.0  # ms转秒
                            logger.debug(f"[DEBUG] ✅ 完成新片段: ({start_sample}, {end_sample})采样点, "
                                  f"[{pending_ms}ms, {end_ms}ms], 时长={duration_sec:.2f}秒")
                        else:
                            logger.warning(f"[DEBUG] ⚠️ 片段无效: end_sample({end_sample}) <= start_sample({start_sample})")
                    else:
                        logger.warning(f"[DEBUG] ⚠️ 结束标记但无pending_start_ms，跳过")
                    pending_ms = None

        cache["pending_start_ms"] = pending_ms  # 更新缓存键名
        # 清理旧键名（向后兼容）
        if "pending_start_frame" in cache:
            del cache["pending_start_frame"]
            
        if new_segments:
            logger.debug(f"[DEBUG] 本轮产生 {len(new_segments)} 个新片段")
        return new_segments


class StreamingASRProcessor:
    """流式ASR处理器类，封装所有流式ASR处理逻辑"""
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.vad_model = model_manager.get_vad_model()
        self.asr_offline_model = model_manager.get_asr_offline_model()
        self.asr_online_model = model_manager.get_asr_online_model()

        # 初始化各个管理器
        self.model_invoker = ModelInvoker(
            self.vad_model,
            self.asr_online_model,
            self.asr_offline_model
        )
        self.segment_manager = SegmentManager()
        
        # 懒加载离线ASR处理器（用于高精度识别和说话人识别）
        self._offline_processor = None
        
        # 初始化工具类
        self.text_extractor = TextExtractor()
        self.result_aligner = ResultAligner()
        self._offline_asr_trigger = None
    
    @property
    def offline_processor(self):
        """懒加载离线ASR处理器（包含VAD+ASR+说话人识别完整流程）"""
        if self._offline_processor is None:
            self._offline_processor = OfflineASRProcessor(self.model_manager)
        return self._offline_processor
    
    @property
    def offline_asr_trigger(self):
        """懒加载离线ASR触发器"""
        if self._offline_asr_trigger is None:
            self._offline_asr_trigger = OfflineASRTrigger(self.offline_processor)
        return self._offline_asr_trigger

    # 保留这些方法作为公共接口，内部委托给 ModelInvoker
    def process_streaming_vad(self, audio_chunk: np.ndarray, cache: Dict):
        """公共接口：处理流式 VAD"""
        return self.model_invoker.invoke_streaming_vad(audio_chunk, cache)

    def process_offline_vad(self, audio_chunk: np.ndarray):
        """公共接口：处理离线 VAD"""
        return self.model_invoker.invoke_offline_vad(audio_chunk)

    def process_streaming_asr(self, audio_chunk: np.ndarray, cache: Dict):
        """公共接口：处理流式 ASR"""
        return self.model_invoker.invoke_streaming_asr(audio_chunk, cache)

    def process_offline_asr(self, audio_chunk: np.ndarray):
        """公共接口：处理离线 ASR"""
        return self.model_invoker.invoke_offline_asr(audio_chunk)

    def _extract_asr_text(self, asr_result) -> str:
        """
        从ASR结果中提取文本（委托给 TextExtractor）
        
        Args:
            asr_result: ASR模型返回的结果（多种可能格式）
        
        Returns:
            识别文本字符串，如果提取失败返回空字符串
        """
        return self.text_extractor.extract_text(asr_result)
    
    def _align_results(self, cache: Dict) -> str:
        """
        对齐流式ASR和离线ASR结果，生成最终文本（委托给 ResultAligner）
        
        Args:
            cache: 包含streaming_asr_results和offline_asr_segments的缓存
        
        Returns:
            对齐后的完整文本
        """
        return self.result_aligner.align_results(cache)
    
    def _process_vad(self, audio_chunk: np.ndarray, cache: Dict):
        """
        职责：处理 VAD 并更新片段列表
        - 调用流式 VAD
        - 解析 VAD 结果，更新已完成的片段
        - 返回 VAD 结果和新完成的片段列表
        
        ⚠️ 重要：流式VAD模型通过cache维护内部状态，返回的时间戳是**毫秒（ms）**，
        是从音频开始累计的绝对时间。SegmentManager会自动转换为采样点。
        """
        # 调用流式VAD模型
        vad_result = self.model_invoker.invoke_streaming_vad(audio_chunk, cache)
        
        # 更新片段列表（VAD返回的ms会在这里转换为采样点）
        new_segments = self.segment_manager.update_from_vad(cache, vad_result)
        
        # 流式VAD结果已通过logger记录
        
        return vad_result, new_segments

    def _process_streaming_asr_if_needed(self, audio_chunk: np.ndarray, cache: Dict, samples_start_idx: int):
        """
        职责：按需处理流式 ASR
        - 累积每个VAD chunk（200ms）到ASR缓冲区
        - 每 3 个 chunk（600ms）触发一次ASR处理
        - 传入累积的600ms音频，而不是单个200ms chunk
        - 处理完后清空缓冲区，准备下一轮累积
        - 返回 streaming ASR 结果（未触发则返回 None）
        
        Args:
            audio_chunk: 当前音频chunk
            cache: 缓存字典
            samples_start_idx: 当前chunk在整个音频流中的起始采样点索引
        """
        # 累积音频到ASR缓冲区
        if len(audio_chunk) > 0:
            if len(cache["asr_online_audio_buffer"]) == 0:
                # 记录缓冲区的起始位置（第一个chunk的起始位置）
                cache["asr_online_buffer_start_idx"] = samples_start_idx
            # 将当前chunk添加到缓冲区（累积）
            cache["asr_online_audio_buffer"].extend(audio_chunk)
        
        cache["chunk_counter"] += 1
        should_process_asr = cache["chunk_counter"] % 3 == 0
        
        if should_process_asr and len(cache["asr_online_audio_buffer"]) > 0:
            # 将累积的音频列表转换为numpy数组（600ms的累积音频）
            asr_audio_chunk = np.array(cache["asr_online_audio_buffer"], dtype=np.float32)
            
            # 调用流式ASR模型，传入累积的600ms音频
            # 模型内部会使用cache保持encoder/decoder状态，但处理的是完整的600ms窗口
            asr_result = self.model_invoker.invoke_streaming_asr(
                asr_audio_chunk, cache.get("asr_online_cache")
            )
            
            # 清空ASR缓冲区，准备下一轮累积
            cache["asr_online_audio_buffer"] = []
            cache["asr_online_buffer_start_idx"] = 0
            
            return asr_result
        
        return None

    def _process_offline_asr_for_segments(
        self, 
        cache: Dict, 
        new_segments: List[Tuple[int, int]],
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        force_trigger: bool = False
    ):
        """
        为新完成的片段处理离线 ASR（委托给 OfflineASRTrigger）
        
        Args:
            cache: 流式处理缓存
            new_segments: 本轮新完成的片段列表
            enable_speaker_diarization: 是否启用说话人识别（默认True）
            speaker_mode: 说话人识别模式（cluster/cluster_match/direct_match）
            registered_speakers: 已注册说话人名称列表
            similarity_threshold: 相似度阈值
            force_trigger: 是否强制触发（finalize时使用，默认False）
        
        Returns:
            OfflineASRProcessor 的处理结果（包含text, segments等）或 None
        """
        return self.offline_asr_trigger.process_segments(
            cache=cache,
            new_segments=new_segments,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold,
            force_trigger=force_trigger
        )

    def streaming_pipeline(self, audio_chunk: np.ndarray, cache: Dict = None):
        """
        流式管线（仅VAD+流式ASR）
        - 不包含离线ASR和音频缓存
        - 适用于实时低延迟场景
        - 返回VAD结果和流式ASR结果
        
        ⚠️ 注意：流式VAD模型通过cache维护内部状态，返回的时间戳是毫秒（ms）。
        """
        if cache is None:
            cache = {}
        CacheInitializer.ensure_state(cache)

        # 记录当前chunk的起始采样点位置
        samples_start_idx = cache["samples_processed"]
        
        # 更新采样点计数（供ASR使用，但不实际缓存音频）
        cache["samples_processed"] += len(audio_chunk)

        # 处理 VAD（返回的时间戳是ms，会在内部转换为采样点）
        vad_result = self.model_invoker.invoke_streaming_vad(audio_chunk, cache)

        # 按需处理流式 ASR，传入当前chunk的起始位置
        streaming_asr_result = self._process_streaming_asr_if_needed(audio_chunk, cache, samples_start_idx)

        return {
            "vad": vad_result,
            "streaming_asr": streaming_asr_result,
        }

    def _is_segment_duplicate(
        self,
        new_seg: Dict[str, Any],
        existing_segments: List[Dict[str, Any]],
        overlap_threshold: float = 0.8  # 重叠阈值：80%
    ) -> bool:
        """
        检查新片段是否与已有片段重复
        
        Args:
            new_seg: 新片段
            existing_segments: 已有片段列表
            overlap_threshold: 重叠阈值（0-1），超过此阈值认为是重复
        
        Returns:
            True if duplicate, False otherwise
        """
        new_start = new_seg.get("start", 0)
        new_end = new_seg.get("end", 0)
        new_duration = new_end - new_start
        
        if new_duration <= 0:
            return True  # 异常片段，认为是重复
        
        # ⚠️ 改进：检查start_sample和end_sample是否都相等
        # 这样可以区分同一个起始位置但不同长度的片段
        new_start_sample = new_seg.get("start_sample", 0)
        new_end_sample = new_seg.get("end_sample", 0)
        
        for existing in existing_segments:
            if (existing.get("start_sample", 0) == new_start_sample and
                existing.get("end_sample", 0) == new_end_sample):
                return True  # 完全相同的片段
        
        # 然后检查时间戳范围是否重叠
        for existing in existing_segments:
            existing_start = existing.get("start", 0)
            existing_end = existing.get("end", 0)
            existing_duration = existing_end - existing_start
            
            if existing_duration <= 0:
                continue
            
            # 计算重叠时间
            overlap_start = max(new_start, existing_start)
            overlap_end = min(new_end, existing_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # 计算重叠比例（相对于新片段的时长）
            overlap_ratio = overlap_duration / new_duration if new_duration > 0 else 0
            
            if overlap_ratio >= overlap_threshold:
                logger.warning(f"[Pipeline] 检测到重复片段:")
                logger.warning(f"  新片段: [{new_start:.2f}s - {new_end:.2f}s]")
                logger.warning(f"  已有片段: [{existing_start:.2f}s - {existing_end:.2f}s]")
                logger.warning(f"  重叠比例: {overlap_ratio:.2%}")
                return True
        
        return False

    def pipeline(
        self, 
        audio_chunk: np.ndarray, 
        cache: Dict = None,
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5
    ):
        """
        流式ASR处理管线（累积结果模式）
        
        核心特性：
        ⚠️ 每次调用返回从WebSocket连接开始到现在的**所有累积结果**，而不是单次chunk的结果
        
        处理流程：
        1. 音频缓存管理（60秒滑动窗口）
        2. 流式VAD检测（每200ms chunk）
        3. 流式ASR识别（每600ms触发，低延迟）→ 累积到streaming_asr_results
        4. 离线ASR识别（片段完成时触发，高精度）→ 累积到offline_asr_segments
        5. 结果对齐（基于VAD时间戳，优先使用离线高精度结果）
        
        Args:
            audio_chunk: 输入音频chunk (推荐200ms, 3200采样点 @ 16kHz)
            cache: 流式处理状态缓存（每个WebSocket连接一套cache），包含：
                - audio_buffer: 音频数据缓存 [(start_idx, audio_data), ...]
                - segments: VAD检测到的完成片段 [(start_sample, end_sample), ...] ⚠️采样点单位
                - vad_cache: VAD模型内部状态（模型内部维护，返回ms时间戳）
                - asr_online_cache: 流式ASR模型内部状态
                - streaming_asr_results: 累积的流式ASR结果列表 ⭐
                - offline_asr_segments: 累积的离线ASR片段列表 ⭐
                - samples_processed: 已处理的总采样点数
                - pending_start_ms: 待配对的VAD起始时间（毫秒）
                - chunk_counter: chunk计数器（用于ASR触发）
            enable_speaker_diarization: 是否启用说话人识别（默认True）
            speaker_mode: 说话人识别模式（默认"cluster"）
                - "cluster": 仅聚类，自动识别说话人数量
                - "cluster_match": 聚类后与已注册说话人匹配
                - "direct_match": 每个片段直接匹配最相似的说话人
            registered_speakers: 已注册说话人名称列表（用于匹配模式）
            similarity_threshold: 相似度阈值（默认0.5，范围0-1）
        
        Returns:
            {
                # 原始调试信息
                "vad_raw": VAD原始输出（流式标记），
                "new_segments_count": 本次新完成的VAD片段数,
                
                # 累积的完整结果（核心返回数据）⭐
                "streaming_asr_history": [        # 流式ASR历史（所有识别结果）
                    {"text": str, "timestamp": float, "chunk_idx": int},
                    ...
                ],
                "offline_asr_segments": [         # 离线ASR片段（高精度+说话人）
                    {
                        "start": float,           # 绝对开始时间（秒）
                        "end": float,             # 绝对结束时间（秒）
                        "text": str,              # 片段文本（带标点）
                        "speaker": str,           # 说话人标识
                        "start_sample": int,      # 绝对开始采样点
                        "end_sample": int,        # 绝对结束采样点
                        "similarity": float       # 相似度分数（如有匹配）
                    },
                    ...
                ],
                "aligned_text": str,              # 对齐后的最终文本（推荐使用）⭐
                
                # 会话统计信息
                "session_stats": {
                    "total_audio_duration": float,    # 总音频时长（秒）
                    "vad_segments_count": int,        # VAD检测的片段数
                    "streaming_asr_count": int,       # 流式ASR结果数
                    "offline_asr_count": int,         # 离线ASR片段数
                    "chunk_counter": int              # 处理的chunk数
                }
            }
            
        使用建议：
        - 前端应该使用`aligned_text`作为最终显示文本
        - 如果需要说话人信息，使用`offline_asr_segments`
        - 如果需要实时反馈，可以展示`streaming_asr_history`的最新项
        - `session_stats`可用于监控处理进度和音频时长
        """
        # ==================== Step 1: 初始化缓存状态 ====================
        if cache is None:
            cache = {}
        CacheInitializer.ensure_state(cache)
        # 确保所有必需字段存在：vad_cache, asr_online_cache, segments, 
        # audio_buffer, samples_processed, pending_start_ms, chunk_counter 等

        # ==================== Step 2: 音频缓存管理 ====================
        # Step 2.1: 记录当前chunk的起始采样点位置（在append之前）
        # 用途：为流式ASR提供时间戳参考
        samples_start_idx = cache["samples_processed"]
        
        # Step 2.2: 累积音频到缓存（用于后续离线ASR和60秒滑动窗口）
        # AudioBufferManager.append 会：
        #   - 将当前chunk存储为 (start_idx, audio_data) 对
        #   - 更新 samples_processed 计数器
        AudioBufferManager.append(cache, audio_chunk)
        
        # 调试信息：当前缓存状态
        logger.debug(f"[Pipeline] Chunk输入: {len(audio_chunk)}采样点 "
              f"({len(audio_chunk)/16000*1000:.1f}ms), "
              f"累积位置: {samples_start_idx/16000:.2f}s - {cache['samples_processed']/16000:.2f}s")

        # ==================== Step 3: 流式VAD处理（语音活动检测）====================
        # 功能：检测语音片段的开始和结束边界
        # 输入：200ms音频chunk
        # 输出：
        #   - vad_result: VAD原始输出（流式标记格式 [start, -1] 或 [-1, end]）
        #   - new_segments: 本轮新完成的片段列表 [(start, end), ...]
        # 
        # 内部逻辑：
        #   1. 调用流式VAD模型（通过vad_cache维护状态）
        #   2. 解析VAD输出，识别片段开始标记 [s, -1] 和结束标记 [-1, e]
        #   3. 配对开始/结束标记，形成完整片段 (start, end)
        #   4. 将完整片段添加到 cache["segments"] 列表
        vad_result, new_segments = self._process_vad(audio_chunk, cache)
        
        logger.info(f"[Pipeline] VAD结果: {len(new_segments)}个新完成片段, "
              f"总片段数: {len(cache['segments'])}")

        # ==================== Step 3.5: 持续维护滑动窗口（清理旧片段）====================
        # 功能：基于已处理位置清理旧片段，避免无限累积
        # 策略：
        #   - 如果已完成第一次触发，清理"上次触发结束位置"之前的旧片段
        #   - 保留一定的重叠区域（以防需要重新处理）
        # 
        # ⚠️ 核心改进：基于"已处理位置"而不是"最后一个片段"来清理
        segments = cache.get("segments", [])
        first_trigger_done = cache.get("first_offline_trigger_done", False)
        last_trigger_end = cache.get("last_offline_trigger_end", 0)
        
        if segments and first_trigger_done and last_trigger_end > 0:
            # 清理策略：保留上次触发结束位置之后的片段
            # 留一些重叠（例如10秒），避免边界问题
            keep_after = last_trigger_end - 10 * 16000  # 保留触发结束前10秒
            
            if keep_after > 0:
                old_count = len(segments)
                SlidingWindowManager.cleanup_old_segments(cache, keep_after)
                new_count = len(cache.get("segments", []))
                
                if old_count != new_count:
                    logger.info(f"[Pipeline] 🧹 清理旧片段：清理了{old_count - new_count}个，"
                          f"保留{new_count}个（{keep_after/16000:.2f}s之后的片段）")
                    
                    # 同时清理离线精细VAD片段
                    offline_processed = cache.get("offline_processed_segments", [])
                    if offline_processed:
                        new_offline = [
                            (start, end) for start, end in offline_processed
                            if end > keep_after
                        ]
                        if len(new_offline) != len(offline_processed):
                            cache["offline_processed_segments"] = new_offline
                            logger.info(f"[Pipeline] 🧹 清理了{len(offline_processed) - len(new_offline)}个旧的离线精细片段")

        # ==================== Step 4: 流式ASR处理（实时识别）====================
        # 功能：提供低延迟的实时文本输出
        # 触发条件：每累积3个chunk（600ms）触发一次
        # 
        # 处理流程：
        #   1. 累积每个200ms chunk到 asr_online_audio_buffer
        #   2. chunk_counter 计数器 +1
        #   3. 当 counter % 3 == 0 时：
        #      - 将累积的600ms音频传入流式ASR模型
        #      - 模型通过 asr_online_cache 维护encoder/decoder状态
        #      - 清空 asr_online_audio_buffer，准备下一轮累积
        #   4. 返回识别文本（或None如果未触发）
        # 
        # 配置参数：
        #   - chunk_size=[0, 10, 5]: ASR窗口600ms（10帧*60ms）
        #   - encoder_chunk_look_back=4: 编码器回看2.4秒
        #   - decoder_chunk_look_back=1: 解码器回看0.6秒
        streaming_asr_result = self._process_streaming_asr_if_needed(
            audio_chunk, cache, samples_start_idx
        )
        
        # Step 4.1: 累积流式ASR结果
        if streaming_asr_result:
            asr_text = self._extract_asr_text(streaming_asr_result)
            if asr_text:
                timestamp = cache['samples_processed'] / 16000.0
                chunk_idx = cache['chunk_counter']
                
                # 累积流式ASR结果到cache
                cache["streaming_asr_results"].append({
                    "text": asr_text,
                    "timestamp": timestamp,  # 当前时间点（秒）
                    "chunk_idx": chunk_idx
                })
                logger.info(f"[Pipeline] 流式ASR触发（第{chunk_idx}个chunk）: {asr_text}")

        # ==================== Step 5: 离线ASR处理（高精度识别）====================
        # 功能：当VAD片段完成且触发条件满足时，进行高精度识别
        # 触发条件（在OfflineASRTrigger中判断）：
        #   - 有新VAD片段完成（必须条件）
        #   - 且达到触发条件：
        #     * 第一次：窗口累加到55-65s
        #     * 后续：距离上次触发25-35s（步长）
        # 
        # 设计理念：
        #   - 流式VAD持续生成片段
        #   - 有新片段时检查是否达到触发条件
        #   - 触发后清理旧片段，避免累积
        # 
        # 处理流程（在 _process_offline_asr_for_segments 中）：
        #   Step 5.1: 判断是否应该触发（基于窗口长度/步长）
        #   Step 5.2: 计算窗口范围
        #     - begin: 从离线VAD结果找（上次结束+30s）
        #     - end: 从流式VAD找（累加60s）
        #   Step 5.3: 验证音频数据完整性
        #   Step 5.4: 音频缓存裁剪（内存优化）
        #   Step 5.5: 提取窗口音频
        #   Step 5.6: 调用 OfflineASRProcessor（VAD+ASR+PUNC+说话人识别）
        #   Step 5.7: 时间戳转换（窗口相对 → 全局绝对）
        #   Step 5.8: 清理旧片段（重要！避免累积）
        offline_asr_result = self._process_offline_asr_for_segments(
            cache, new_segments,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers,
            similarity_threshold=similarity_threshold
        )
        
        # Step 5.1: 累积离线ASR结果到cache
        if offline_asr_result and offline_asr_result.get("segments"):
            new_offline_segments = offline_asr_result["segments"]
            # 累积新的离线ASR片段（去重：避免重复累积）
            added_count = 0
            skipped_count = 0
            for seg in new_offline_segments:
                if self._is_segment_duplicate(seg, cache["offline_asr_segments"]):
                    skipped_count += 1
                    logger.debug(f"[Pipeline] 跳过重复片段: [{seg.get('start', 0):.2f}s - {seg.get('end', 0):.2f}s]")
                    continue
                cache["offline_asr_segments"].append(seg)
                added_count += 1
            
            # ⚠️ 重要：按时间戳排序，确保结果顺序正确
            if added_count > 0:
                cache["offline_asr_segments"].sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))
            
            logger.info(f"[Pipeline] 离线ASR完成: 新增{added_count}个片段，跳过{skipped_count}个重复片段，"
                  f"累积总计{len(cache['offline_asr_segments'])}个片段")

        # ==================== Step 6: 对齐结果并返回累积结果 ====================
        # 基于VAD时间戳，用高精度的离线ASR结果修正流式ASR结果
        aligned_result = self._align_results(cache)
        
        return {
            # 原始结果（用于调试）
            "vad_raw": vad_result,                         # VAD原始输出（流式标记）
            "new_segments_count": len(new_segments),       # 本次新完成的VAD片段数
            
            # 累积的完整结果（前端使用）
            "streaming_asr_history": cache["streaming_asr_results"],   # 流式ASR历史（所有识别结果）
            "offline_asr_segments": cache["offline_asr_segments"],     # 离线ASR片段（高精度+说话人）
            "aligned_text": aligned_result,                             # 对齐后的最终结果
            
            # 会话统计信息
            "session_stats": {
                "total_audio_duration": cache['samples_processed'] / 16000.0,  # 总音频时长（秒）
                "vad_segments_count": len(cache['segments']),                  # VAD检测的片段数
                "streaming_asr_count": len(cache["streaming_asr_results"]),    # 流式ASR结果数
                "offline_asr_count": len(cache["offline_asr_segments"]),       # 离线ASR片段数
                "chunk_counter": cache['chunk_counter']                        # 处理的chunk数
            }
        }
    
    def finalize(
        self,
        cache: Dict,
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
        
        Args:
            cache: 流式处理缓存
            enable_speaker_diarization: 是否启用说话人识别
            speaker_mode: 说话人识别模式
            registered_speakers: 已注册说话人名称列表
            similarity_threshold: 相似度阈值
        
        Returns:
            最终处理结果（格式与pipeline方法相同）
        """
        logger.info(f"\n[Pipeline] " + "="*60)
        logger.info(f"[Pipeline] 🏁 开始finalize处理（音频结束）")
        total_duration = cache.get("samples_processed", 0) / 16000.0
        logger.info(f"[Pipeline] 总音频时长: {total_duration:.2f}s")
        
        # 确保缓存状态已初始化
        CacheInitializer.ensure_state(cache)
        
        # ==================== Step 1: 处理pending_start_ms（未完成的VAD片段）====================
        pending_start_ms = cache.get("pending_start_ms")
        samples_processed = cache.get("samples_processed", 0)
        
        if pending_start_ms is not None and samples_processed > 0:
            # 将pending片段转换为完整片段（从pending_start_ms到音频结束）
            start_sample = int(pending_start_ms * AudioConstants.MS_TO_SAMPLES)
            end_sample = samples_processed
            
            if end_sample > start_sample:
                logger.info(f"[Pipeline] 📌 发现pending片段: [{pending_start_ms}ms, 音频结束]")
                logger.info(f"[Pipeline]    转换为采样点: [{start_sample}, {end_sample}] "
                      f"({start_sample/16000:.2f}s - {end_sample/16000:.2f}s)")
                
                # 添加到segments列表
                cache["segments"].append((start_sample, end_sample))
                new_segments = [(start_sample, end_sample)]
                
                # 清空pending_start_ms
                cache["pending_start_ms"] = None
                
                # 处理这个最后的片段（强制触发离线ASR）
                logger.info(f"[Pipeline] 处理pending片段（强制触发离线ASR）...")
                offline_asr_result = self._process_offline_asr_for_segments(
                    cache, new_segments,
                    enable_speaker_diarization=enable_speaker_diarization,
                    speaker_mode=speaker_mode,
                    registered_speakers=registered_speakers,
                    similarity_threshold=similarity_threshold,
                    force_trigger=True  # ⚠️ 强制触发
                )
                
                # 累积离线ASR结果
                if offline_asr_result and offline_asr_result.get("segments"):
                    new_offline_segments = offline_asr_result["segments"]
                    added_count = 0
                    for seg in new_offline_segments:
                        if not self._is_segment_duplicate(seg, cache["offline_asr_segments"]):
                            cache["offline_asr_segments"].append(seg)
                            added_count += 1
                    
                    # 按时间戳排序
                    if added_count > 0:
                        cache["offline_asr_segments"].sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))
                    
                    logger.info(f"[Pipeline] 离线ASR完成: 新增{added_count}个片段")
            else:
                logger.warning(f"[Pipeline] ⚠️ pending片段无效: end_sample({end_sample}) <= start_sample({start_sample})")
        else:
            if pending_start_ms is None:
                logger.info(f"[Pipeline] ℹ️ 没有pending片段需要处理")
            else:
                logger.warning(f"[Pipeline] ⚠️ samples_processed为0，无法处理pending片段")
        
        # ==================== Step 2: 处理最后一个窗口（如果有剩余的片段和音频）====================
        # 检查是否还有未处理的片段
        segments = cache.get("segments", [])
        audio_buffer = cache.get("audio_buffer", [])
        offline_segments = cache.get("offline_asr_segments", [])
        
        if segments and audio_buffer:
            # 找出所有未处理的片段（通过比较segments和已处理的offline_segments）
            # 简单方法：如果segments中有片段，但还没有被处理（因为滑动窗口条件未满足），
            # 那么在finalize时应该处理所有剩余的片段
            
            # 获取已处理的片段起始位置集合（用于快速查找）
            processed_start_samples = {seg["start_sample"] for seg in offline_segments}
            
            # 找出未处理的片段（segments中不在已处理列表中的）
            unprocessed_segments = [
                (s, e) for s, e in segments 
                if s not in processed_start_samples
            ]
            
            if unprocessed_segments:
                logger.info(f"[Pipeline] 📦 发现未处理的片段: {len(unprocessed_segments)}个")
                for i, (s, e) in enumerate(unprocessed_segments, 1):
                    logger.info(f"[Pipeline]   片段{i}: [{s/16000:.2f}s, {e/16000:.2f}s]")
                
                # 处理这些未处理的片段（强制触发离线ASR，不论是否满足触发条件）
                logger.info(f"[Pipeline] 处理最后一个窗口（包含所有未处理的片段）...")
                logger.info(f"[Pipeline] ⚠️ Finalize阶段：强制触发离线ASR，不论是否满足触发条件")
                offline_asr_result = self._process_offline_asr_for_segments(
                    cache, unprocessed_segments,
                    enable_speaker_diarization=enable_speaker_diarization,
                    speaker_mode=speaker_mode,
                    registered_speakers=registered_speakers,
                    similarity_threshold=similarity_threshold,
                    force_trigger=True  # ⚠️ 强制触发
                )
                
                # 累积离线ASR结果
                if offline_asr_result and offline_asr_result.get("segments"):
                    new_offline_segments = offline_asr_result["segments"]
                    added_count = 0
                    for seg in new_offline_segments:
                        if not self._is_segment_duplicate(seg, cache["offline_asr_segments"]):
                            cache["offline_asr_segments"].append(seg)
                            added_count += 1
                    
                    # 按时间戳排序
                    if added_count > 0:
                        cache["offline_asr_segments"].sort(key=lambda x: (x.get("start", 0), x.get("end", 0)))
                    
                    logger.info(f"[Pipeline] 最后一个窗口处理完成: 新增{added_count}个片段")
            else:
                logger.info(f"[Pipeline] ℹ️ 所有片段都已被处理")
        else:
            if not segments:
                logger.info(f"[Pipeline] ℹ️ 没有片段需要处理")
            if not audio_buffer:
                logger.warning(f"[Pipeline] ℹ️ audio_buffer为空，无法处理片段")
        
        # ==================== Step 3: 对齐结果并返回最终结果 ====================
        aligned_result = self._align_results(cache)
        
        logger.info(f"[Pipeline] ✅ Finalize完成")
        logger.info(f"[Pipeline] 最终离线ASR片段总数: {len(cache['offline_asr_segments'])}")
        logger.info(f"[Pipeline] " + "="*60)
        
        return {
            # 最终结果
            "streaming_asr_history": cache["streaming_asr_results"],
            "offline_asr_segments": cache["offline_asr_segments"],
            "aligned_text": aligned_result,
            
            # 会话统计信息
            "session_stats": {
                "total_audio_duration": cache['samples_processed'] / 16000.0,
                "vad_segments_count": len(cache['segments']),
                "streaming_asr_count": len(cache["streaming_asr_results"]),
                "offline_asr_count": len(cache["offline_asr_segments"]),
                "chunk_counter": cache['chunk_counter']
            }
        }

