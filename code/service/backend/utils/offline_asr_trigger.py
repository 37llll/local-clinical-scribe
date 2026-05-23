"""
离线 ASR 触发工具

功能：
- 管理离线 ASR 的触发逻辑
- 维护 60 秒滑动窗口
- 处理音频缓存和窗口提取
- 时间戳转换（窗口相对时间 → 全局绝对时间）
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from ..processors.offline_processor import OfflineASRProcessor
from .audio_constants import AudioConstants
from .streaming_utils import (
    AudioBufferManager,
    SlidingWindowManager,
)
from .logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class OfflineASRTrigger:
    """
    离线 ASR 触发器
    
    功能：
    - 管理离线 ASR 的触发条件
    - 维护 60 秒滑动窗口
    - 处理音频缓存和窗口提取
    - 调用 OfflineASRProcessor
    - 转换时间戳（窗口相对 → 全局绝对）
    """
    
    def __init__(self, offline_processor: OfflineASRProcessor):
        """
        初始化离线 ASR 触发器
        
        Args:
            offline_processor: OfflineASRProcessor 实例
        """
        self.offline_processor = offline_processor
    
    def process_segments(
        self,
        cache: Dict[str, Any],
        new_segments: List[Tuple[int, int]],
        enable_speaker_diarization: bool = True,
        speaker_mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5,
        force_trigger: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        为新完成的片段处理离线 ASR
        
        处理流程：
        1. 检查触发条件（基于滑动窗口策略：累加更新或步长触发）
        2. 计算滑动窗口范围（支持步长和累加策略）
        3. 验证音频数据完整性
        4. 音频缓存裁剪（内存优化）
        5. 提取窗口音频
        6. 调用 OfflineASRProcessor
        7. 时间戳转换（窗口相对 → 全局绝对）⚠️ 关键：确保时间戳对齐
        8. 更新触发状态
        
        Args:
            cache: 流式处理缓存
            new_segments: 本轮新完成的片段列表
            enable_speaker_diarization: 是否启用说话人识别
            speaker_mode: 说话人识别模式
            registered_speakers: 已注册说话人名称列表
            similarity_threshold: 相似度阈值
            force_trigger: 是否强制触发（finalize时使用，默认False）
        
        Returns:
            OfflineASRProcessor 的处理结果（包含 text, segments 等）或 None
        """
        # ========== Step 1: 检查触发条件 ==========
        if not new_segments:
            return None
        
        # 使用新的滑动窗口策略判断是否应该触发
        should_trigger = SlidingWindowManager.should_trigger_offline_asr(cache, new_segments, force_trigger)
        if not should_trigger:
            logger.debug(f"[OfflineASR] ⏸️ 收到{len(new_segments)}个新片段，但未达到触发条件，暂不处理")
            return None
        
        logger.info(f"\n[OfflineASR] " + "="*60)
        logger.info(f"[OfflineASR] 🎬 触发离线ASR处理：{len(new_segments)}个新片段")
        logger.info(f"[OfflineASR] 新片段时间: {[(s/16000, e/16000) for s, e in new_segments]}")
        logger.info(f"[OfflineASR] 当前总片段数: {len(cache['segments'])}")
        
        # ========== Step 2: 计算滑动窗口范围 ==========
        window_start, window_end = SlidingWindowManager.calculate_window_range(cache)
        if window_start is None or window_end is None:
            logger.warning(f"[OfflineASR] ⚠️ 无法计算窗口范围，跳过")
            return None
        
        # 验证window_start值是否合理
        segments = cache.get("segments", [])
        if segments:
            first_segment_start = segments[0][0]
            if window_start < first_segment_start:
                logger.warning(f"[OfflineASR] ⚠️⚠️⚠️ window_start({window_start/16000:.2f}s) < 第一个片段起始({first_segment_start/16000:.2f}s)")
                logger.warning(f"[OfflineASR]     这可能导致时间戳转换错误！")
            elif window_start > first_segment_start:
                logger.info(f"[OfflineASR] ℹ️ window_start({window_start/16000:.2f}s) > 第一个片段起始({first_segment_start/16000:.2f}s)")
                logger.info(f"[OfflineASR]     这是正常的（窗口调整后），将裁剪第一个片段的前部分")
        
        # 验证音频数据完整性
        audio_buffer_start, audio_buffer_end = self._get_audio_buffer_range(cache)
        if audio_buffer_start is None or audio_buffer_end is None:
            logger.warning(f"[OfflineASR] ⚠️ audio_buffer为空，跳过")
            return None
        
        # 调整窗口结束位置（如果超出实际缓存）
        window_end = self._adjust_window_end(window_end, audio_buffer_end)
        
        # 音频缓存裁剪（内存优化）
        self._trim_audio_buffer(cache, window_start)
        
        # 提取窗口音频
        audio_window = self._extract_window_audio(cache, window_start, window_end)
        if audio_window is None:
            return None
        
        # 记录离线ASR触发信息到专用日志
        # 提取窗口内的流式VAD片段
        window_vad_segments = [
            (start, end) for start, end in cache.get("segments", [])
            if end > window_start and start < window_end
        ]
        trigger_type = "finalize" if force_trigger else "normal"
        # 离线ASR触发信息已通过logger记录
        
        # 调用 OfflineASRProcessor
        offline_result = self._invoke_offline_processor(
            audio_window,
            cache,
            enable_speaker_diarization,
            speaker_mode,
            registered_speakers,
            similarity_threshold
        )
        
        # ========== Step 7: 时间戳转换（窗口相对 → 全局绝对）==========
        if offline_result:
            self._convert_timestamps(offline_result, window_start)
            self._print_result_summary(offline_result, cache)
            
            # 离线ASR结果已通过logger记录
            
            # ========== Step 8: 保存离线处理的精细VAD片段 ==========
            # 这些精细片段用于下次窗口计算时的起始位置（避免重复处理）
            segments = offline_result.get("segments") or []
            if segments:
                for seg in segments:
                    seg_start = seg.get("start_sample", 0)
                    seg_end = seg.get("end_sample", 0)
                    if seg_start < seg_end:  # 过滤异常片段
                        # 检查是否已存在（去重）
                        if not any(existing[0] == seg_start 
                                  for existing in cache.get("offline_processed_segments", [])):
                            cache["offline_processed_segments"].append((seg_start, seg_end))
                
                logger.info(f"[OfflineASR] 保存了{len(segments)}个精细VAD片段到cache "
                      f"(累计{len(cache['offline_processed_segments'])}个)")
            
            # ========== Step 9: 更新触发状态 ==========
            # 判断是否是第一次正式触发（达到60s）
            first_trigger_done = cache.get("first_offline_trigger_done", False)
            window_duration_sec = (window_end - window_start) / 16000.0
            
            # 如果窗口时长在55-65s之间，且还未完成第一次触发，则视为第一次正式触发
            is_first_formal_trigger = (not first_trigger_done and 
                                       55 <= window_duration_sec <= 65)
            
            SlidingWindowManager.update_trigger_state(cache, window_end, is_first_formal_trigger)
            
            # ========== Step 10: 清理旧片段（基于已处理位置）==========
            # 策略：清理window_end之前的片段，确保只保留未来要处理的片段
            # ⚠️ 关键修复：不保留重叠，直接清理已处理的范围，避免窗口回溯
            if first_trigger_done or is_first_formal_trigger:
                # 清理流式VAD片段：保留window_end之后的片段（留一点重叠用于边界保护）
                keep_after_streaming = window_end - 5 * 16000  # window_end前5秒开始保留
                
                if keep_after_streaming > 0:
                    old_count = len(cache.get("segments", []))
                    SlidingWindowManager.cleanup_old_segments(cache, keep_after_streaming)
                    new_count = len(cache.get("segments", []))
                    
                    if old_count != new_count:
                        logger.info(f"[OfflineASR] 清理了{old_count - new_count}个旧的流式VAD片段，"
                              f"保留{new_count}个片段（{keep_after_streaming/16000:.2f}s之后）")
                
                # ⚠️ 关键修复：清理离线精细VAD片段时，只保留window_end之后的片段
                # 这样可以确保下次查找时不会回溯到已处理的范围
                last_trigger_end = cache.get("last_offline_trigger_end", 0)
                keep_after_offline = last_trigger_end  # 只保留上次触发结束之后的片段
                
                offline_processed = cache.get("offline_processed_segments", [])
                new_offline_processed = [
                    (start, end) for start, end in offline_processed 
                    if start >= keep_after_offline  # ← 关键：用start而不是end，确保起始位置在上次结束之后
                ]
                removed_offline = len(offline_processed) - len(new_offline_processed)
                if removed_offline > 0:
                    cache["offline_processed_segments"] = new_offline_processed
                    logger.info(f"[OfflineASR] 清理了{removed_offline}个旧的离线精细片段，"
                          f"保留{len(new_offline_processed)}个片段（起始>{keep_after_offline/16000:.2f}s）")
        
        logger.info(f"[OfflineASR] " + "="*60)
        return offline_result
    
    def _get_audio_buffer_range(
        self,
        cache: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        获取 audio_buffer 的时间范围
        
        Args:
            cache: 流式处理缓存
        
        Returns:
            (start, end) 元组，如果 buffer 为空返回 (None, None)
        """
        if not cache.get('audio_buffer'):
            return None, None
        
        first_start = cache['audio_buffer'][0][0]
        last_start = cache['audio_buffer'][-1][0]
        last_end = last_start + len(cache['audio_buffer'][-1][1])
        
        return first_start, last_end
    
    def _adjust_window_end(
        self,
        window_end: int,
        audio_buffer_end: int
    ) -> int:
        """
        调整窗口结束位置（如果超出实际缓存）
        
        Args:
            window_end: 窗口结束位置（采样点）
            audio_buffer_end: 实际缓存结束位置（采样点）
        
        Returns:
            调整后的窗口结束位置
        """
        if window_end > audio_buffer_end:
            shortage = window_end - audio_buffer_end
            logger.warning(f"[OfflineASR] ⚠️ 窗口结束({window_end/16000:.2f}s) 超出 buffer结束({audio_buffer_end/16000:.2f}s)")
            logger.warning(f"[OfflineASR]    缺少{shortage/16000:.2f}s，使用buffer实际结束位置")
            return audio_buffer_end
        return window_end
    
    def _trim_audio_buffer(self, cache: Dict[str, Any], window_start: int):
        """
        音频缓存裁剪（内存优化）
        
        Args:
            cache: 流式处理缓存
            window_start: 窗口起始位置（采样点）
        """
        logger.info(f"[OfflineASR] 执行trim操作：丢弃 {window_start/16000:.2f}s 之前的音频")
        AudioBufferManager.trim(cache, window_start)
        logger.info(f"[OfflineASR] trim后: audio_buffer剩余{len(cache['audio_buffer'])}个chunk")
    
    def _extract_window_audio(
        self,
        cache: Dict[str, Any],
        window_start: int,
        window_end: int
    ) -> Optional[np.ndarray]:
        """
        提取窗口音频
        
        Args:
            cache: 流式处理缓存
            window_start: 窗口起始位置（采样点）
            window_end: 窗口结束位置（采样点）
        
        Returns:
            窗口音频数据，如果提取失败返回 None
        """
        audio_window = AudioBufferManager.extract(cache, window_start, window_end)
        
        if len(audio_window) == 0:
            logger.warning(f"[OfflineASR] ⚠️ 提取的窗口音频为空，跳过")
            return None
        
        expected_length = window_end - window_start
        if len(audio_window) < expected_length:
            shortage = expected_length - len(audio_window)
            logger.warning(f"[OfflineASR] ⚠️ 音频长度不足：期望{expected_length}，实际{len(audio_window)}，"
                  f"缺少{shortage}采样点({shortage/16000:.2f}s)")
        
        logger.info(f"[OfflineASR] 提取窗口音频成功: {len(audio_window)}采样点 ({len(audio_window)/16000:.2f}s)")
        return audio_window
    
    def _invoke_offline_processor(
        self,
        audio_window: np.ndarray,
        cache: Dict[str, Any],
        enable_speaker_diarization: bool,
        speaker_mode: str,
        registered_speakers: Optional[List[str]],
        similarity_threshold: float
    ) -> Optional[Dict[str, Any]]:
        """
        调用 OfflineASRProcessor
        
        Args:
            audio_window: 窗口音频数据
            cache: 流式处理缓存
            enable_speaker_diarization: 是否启用说话人识别
            speaker_mode: 说话人识别模式
            registered_speakers: 已注册说话人名称列表
            similarity_threshold: 相似度阈值
        
        Returns:
            OfflineASRProcessor 的处理结果或 None
        """
        logger.info(f"[OfflineASR] 调用 OfflineASRProcessor...")
        logger.info(f"[OfflineASR] 传入音频: {len(audio_window)}采样点 ({len(audio_window)/16000:.2f}s)")
        logger.info(f"[OfflineASR] 期望涵盖流式VAD检测到的 {len(cache['segments'])} 个片段")
        logger.info(f"[OfflineASR] 配置: 说话人识别={enable_speaker_diarization}, "
              f"模式={speaker_mode}, 相似度阈值={similarity_threshold}")
        
        try:
            offline_result = self.offline_processor.process(
                audio=audio_window,
                enable_speaker_diarization=enable_speaker_diarization,
                speaker_mode=speaker_mode,
                registered_speakers=registered_speakers,
                similarity_threshold=similarity_threshold
            )
            
            # 检查返回的片段数
            if offline_result:
                segments = offline_result.get("segments") or []
                returned_segments = len(segments)
                logger.warning(f"[OfflineASR] ⚠️ OfflineASRProcessor 返回了 {returned_segments} 个片段 "
                      f"(流式VAD检测到 {len(cache['segments'])} 个)")
                
                if returned_segments < len(cache['segments']):
                    logger.warning(f"[OfflineASR] ⚠️ 警告：离线VAD检测到的片段数少于流式VAD！")
                    logger.warning(f"[OfflineASR]     可能原因：")
                    logger.warning(f"[OfflineASR]     1. 离线VAD合并了多个片段（max_segment_length设置）")
                    logger.warning(f"[OfflineASR]     2. 音频窗口中间有长时间静音，被VAD忽略")
                    logger.warning(f"[OfflineASR]     3. 窗口音频数据不完整")
            
            return offline_result
            
        except Exception as e:
            logger.error(f"[OfflineASR] ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _convert_timestamps(
        self,
        offline_result: Dict[str, Any],
        window_start: int
    ):
        """
        时间戳转换（窗口相对时间 → 全局绝对时间）
        
        Args:
            offline_result: OfflineASRProcessor 的处理结果
            window_start: 窗口起始位置（采样点）
        """
        if not offline_result:
            return
        
        segments = offline_result.get("segments") or []
        if not segments:
            return
        
        logger.info(f"[OfflineASR] 开始时间戳转换：窗口起始位置 {window_start/16000:.2f}s ({window_start}采样点)")
        
        # 验证window_start值是否合理
        if window_start < 0:
            logger.warning(f"[OfflineASR] ⚠️⚠️⚠️ window_start为负值: {window_start}")
        
        # 检查window_start是否与segments[0][0]一致（如果cache中有segments）
        # 注意：这里不能直接访问cache，因为参数中没有传递cache
        # 但可以通过日志提示检查
        
        for i, segment in enumerate(segments, 1):
            # 将相对时间戳转换为绝对时间戳
            relative_start = segment.get("start", 0)
            relative_end = segment.get("end", 0)
            
            # 调试：打印原始值
            if "start_sample" in segment and "end_sample" in segment:
                logger.debug(f"[OfflineASR]   片段{i}原始采样点: [{segment['start_sample']}, {segment['end_sample']}]")
            
            # 转换为全局时间（秒）
            segment["start"] = window_start / AudioConstants.SAMPLE_RATE + relative_start
            segment["end"] = window_start / AudioConstants.SAMPLE_RATE + relative_end
            
            # 同时保存采样点位置（便于后续使用）
            segment["start_sample"] = int(segment["start"] * AudioConstants.SAMPLE_RATE)
            segment["end_sample"] = int(segment["end"] * AudioConstants.SAMPLE_RATE)
            
            # 检测异常：如果开始时间>结束时间
            if segment["start"] > segment["end"]:
                logger.warning(f"[OfflineASR]   ⚠️⚠️⚠️ 片段{i}时间戳异常！开始>结束！")
                logger.warning(f"[OfflineASR]       相对时间: [{relative_start:.2f}s, {relative_end:.2f}s]")
                logger.warning(f"[OfflineASR]       绝对时间: [{segment['start']:.2f}s, {segment['end']:.2f}s]")
                logger.warning(f"[OfflineASR]       window_start: {window_start/16000:.2f}s")
                logger.warning(f"[OfflineASR]       文本: {segment.get('text', '')[:50]}")
            
            # 检测时间戳回退：如果当前片段开始时间 < 前一片段结束时间
            if i > 1:
                prev_segment = segments[i-2]
                if segment["start"] < prev_segment["end"]:
                    logger.warning(f"[OfflineASR]   ⚠️⚠️⚠️ 片段{i}时间戳回退！")
                    logger.warning(f"[OfflineASR]       前一片段: [{prev_segment['start']:.2f}s - {prev_segment['end']:.2f}s]")
                    logger.warning(f"[OfflineASR]       当前片段: [{segment['start']:.2f}s - {segment['end']:.2f}s]")
                    logger.warning(f"[OfflineASR]       window_start: {window_start/16000:.2f}s")
                    logger.warning(f"[OfflineASR]       相对时间: [{relative_start:.2f}s, {relative_end:.2f}s]")
            
            logger.debug(f"[OfflineASR]   片段{i}时间戳转换: "
                  f"相对[{relative_start:.2f}s, {relative_end:.2f}s] → "
                  f"绝对[{segment['start']:.2f}s, {segment['end']:.2f}s] | "
                  f"{segment.get('speaker', 'N/A')} | {segment.get('text', '')[:30]}...")
    
    def _print_result_summary(
        self,
        offline_result: Dict[str, Any],
        cache: Dict[str, Any]
    ):
        """
        打印处理结果摘要
        
        Args:
            offline_result: OfflineASRProcessor 的处理结果
            cache: 流式处理缓存
        """
        text = offline_result.get("text") or ""
        text_len = len(text)
        segments = offline_result.get("segments") or []
        segment_count = len(segments)
        logger.info(f"[OfflineASR] 处理完成: {text_len}个字符, {segment_count}个片段")
        
        if segments:
            speakers = set(seg.get("speaker", "未知") for seg in segments)
            logger.info(f"[OfflineASR] 检测到说话人: {', '.join(sorted(speakers))}")

