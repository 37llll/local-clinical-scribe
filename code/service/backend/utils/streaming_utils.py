"""
流式ASR处理工具类

职责：提供流式ASR处理所需的各种工具函数和辅助类
- CacheInitializer: 初始化缓存状态
- AudioBufferManager: 管理音频缓存
- VADResultParser: 解析VAD结果
- SlidingWindowManager: 维护滑动窗口
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .audio_constants import AudioConstants
from .logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class CacheInitializer:
    """
    职责：初始化缓存状态
    
    设计原则：一个WebSocket连接 → 一套cache → 一个完整的流式音频会话
    
    数据流说明：
    ┌─────────────────────────────────────────────────────────────┐
    │  WebSocket连接 (一个完整的音频会话)                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  音频流输入 → audio_buffer (60秒滑动窗口)                    │
    │       ↓                                                      │
    │  流式VAD → segments (检测到的语音片段)                       │
    │       ↓                                                      │
    │  ┌──────────────────┐     ┌──────────────────┐            │
    │  │  流式ASR路径      │     │  离线ASR路径      │            │
    │  │  (每600ms)       │     │  (VAD片段完成时)  │            │
    │  └────────┬─────────┘     └────────┬─────────┘            │
    │           ↓                         ↓                       │
    │  streaming_asr_results      offline_asr_segments           │
    │  (实时低延迟)                (高精度+说话人)                │
    │           └──────────┬──────────┘                          │
    │                      ↓                                      │
    │              aligned_text (对齐后的最终结果)                │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    
    关键特性：
    - 累积模式：每次pipeline调用返回从连接开始到现在的所有累积结果
    - 对齐机制：优先使用高精度的离线ASR结果，未覆盖部分使用流式ASR
    - 实时性：流式ASR提供低延迟反馈，离线ASR提供高精度修正
    """
    @staticmethod
    def ensure_state(cache: Dict):
        """
        初始化所有必需的状态字段
        
        Cache结构说明：
        1. 模型状态缓存：模型内部维护的状态（VAD/ASR模型需要）
        2. 音频缓存：完整的音频流（用于60秒滑动窗口和离线ASR）
        3. 流式VAD状态：VAD检测的语音片段（用于触发离线ASR）
        4. 流式ASR状态：ASR音频缓冲和触发控制（每600ms触发一次）
        5. 识别结果累积：流式和离线识别的最终结果（核心返回数据）
        6. 进度追踪：采样点计数/chunk计数器（VAD通过cache维护状态，返回毫秒时间戳）
        """
        
        # ==================== 1. 模型状态缓存（模型内部维护）====================
        cache.setdefault("vad_cache", {})           # VAD模型的内部状态（流式处理需要）
        cache.setdefault("asr_online_cache", {})    # 流式ASR模型的内部状态（encoder/decoder缓存）
        
        # ==================== 2. 音频缓存（完整音频流）====================
        cache.setdefault("audio_buffer", [])        # 完整音频缓存：[(start_idx, audio_chunk), ...]
                                                     # 用于60秒滑动窗口和离线ASR
        
        # ==================== 3. 流式VAD状态（语音片段检测）====================
        cache.setdefault("pending_start_ms", None)     # 待配对的VAD起始时间（毫秒，用于片段配对）
        cache.setdefault("segments", [])               # 已完成的VAD片段：[(start_sample, end_sample), ...]
                                                        # 注意：存储的是采样点位置，从VAD返回的ms转换而来
                                                        # 用于触发离线ASR和维护60秒滑动窗口
        
        # ==================== 4. 流式ASR状态（实时识别控制）====================
        cache.setdefault("asr_online_audio_buffer", [])         # ASR音频缓冲区：累积3个chunk=600ms
        cache.setdefault("asr_online_buffer_start_idx", 0)      # ASR缓冲区的起始采样点索引
        cache.setdefault("chunk_counter", 0)                    # chunk计数器（每3个触发一次ASR）
        
        # ==================== 5. 识别结果累积（最终输出）====================
        cache.setdefault("streaming_asr_results", [])  # 流式ASR识别结果列表（低延迟，实时输出）
                                                        # 格式：[{"text": str, "timestamp": float, "chunk_idx": int}, ...]
        cache.setdefault("offline_asr_segments", [])   # 离线ASR识别的片段列表（高精度，包含说话人信息）
                                                        # 格式：[{"start": float, "end": float, "text": str, "speaker": str, ...}, ...]
        
        # ==================== 6. 进度追踪（状态索引）====================
        cache.setdefault("samples_processed", 0)    # 已处理的总采样点数（用于音频索引）
        # 注意：流式VAD通过模型内部cache维护状态，返回的时间戳是绝对毫秒值
        # 不需要手动维护 frames_processed 来转换VAD结果
        
        # ==================== 7. 离线ASR触发状态（滑动窗口和步长控制）====================
        cache.setdefault("last_offline_trigger_end", 0)     # 上次离线ASR触发的窗口结束位置（采样点）
        cache.setdefault("last_offline_update_end", 0)      # 上次累加更新的窗口结束位置（采样点）
        cache.setdefault("first_offline_trigger_done", False)  # 是否已完成第一次正式触发（达到60s）
        cache.setdefault("offline_processed_segments", [])  # 离线ASR处理过的精细VAD片段
                                                             # 格式：[(start_sample, end_sample), ...]
                                                             # 这些是离线ASR内部VAD返回的精细片段（全局绝对时间戳）


class AudioBufferManager:
    """职责：管理音频缓存"""
    @staticmethod
    def append(cache: Dict, audio_chunk: np.ndarray):
        """追加音频块到缓存"""
        start = cache["samples_processed"]
        cache["audio_buffer"].append((start, audio_chunk))
        cache["samples_processed"] += len(audio_chunk)

    @staticmethod
    def trim(cache: Dict, keep_from: int):
        """裁剪缓存，只保留从 keep_from 开始的音频段"""
        trimmed = []
        for start, buf in cache["audio_buffer"]:
            end = start + len(buf)
            if end <= keep_from:
                continue
            if start < keep_from < end:
                offset = keep_from - start
                trimmed.append((keep_from, buf[offset:]))
            else:
                trimmed.append((start, buf))
        cache["audio_buffer"] = trimmed

    @staticmethod
    def extract(cache: Dict, start_sample: int, end_sample: int) -> np.ndarray:
        """从缓存中提取指定范围的音频"""
        pieces: List[np.ndarray] = []
        for start, buf in cache["audio_buffer"]:
            buf_end = start + len(buf)
            if buf_end <= start_sample or start >= end_sample:
                continue
            s = max(start_sample, start)
            e = min(end_sample, buf_end)
            pieces.append(buf[s - start : e - start])
        if not pieces:
            return np.array([], dtype=np.float32)
        return np.concatenate(pieces)


class VADResultParser:
    """职责：解析 VAD 结果"""
    @staticmethod
    def extract_items(vad_result) -> List:
        """从 VAD 结果中提取 items 列表"""
        if not vad_result:
            return []
        if isinstance(vad_result, list):
            return vad_result
        return vad_result.get("result") or []

    @staticmethod
    def extract_value(item) -> Optional[List]:
        """从 item 中提取 value 列表"""
        if isinstance(item, dict):
            return item.get("value")
        return None

    @staticmethod
    def is_start_marker(pair: List) -> bool:
        """判断是否为片段开始标记 [start, -1]"""
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return False
        s, e = pair
        return s is not None and s >= 0 and (e is None or e < 0)

    @staticmethod
    def is_end_marker(pair: List) -> bool:
        """判断是否为片段结束标记 [-1, end]"""
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return False
        s, e = pair
        return e is not None and e >= 0 and (s is None or s < 0)

    @staticmethod
    def convert_to_absolute_frames(vad_result, frames_offset: int):
        """
        将 VAD 结果中的相对帧索引转换为绝对帧索引
        
        ⚠️ 废弃警告：此方法基于错误的假设。FunASR的流式VAD返回的是**毫秒（ms）**，
        不是帧索引。当VAD使用cache时，返回的已经是绝对毫秒时间戳，无需转换。
        
        保留此方法仅用于向后兼容某些特殊场景（如不使用cache的VAD调用）。
        正常的流式处理应使用 SegmentManager.update_from_vad 方法。
        
        Args:
            vad_result: VAD 模型返回的结果（List格式）
            frames_offset: 偏移量（单位取决于实际使用场景）
        
        Returns:
            转换后的 VAD 结果
        """
        if not vad_result:
            return vad_result
        
        # VAD模型始终返回List，不会返回Dict（已通过官方源码验证）
        if not isinstance(vad_result, list):
            raise TypeError(f"VAD结果格式错误：期望List，实际收到{type(vad_result)}")
        
        items = VADResultParser.extract_items(vad_result)
        if not items:
            return vad_result
        
        # 深拷贝结果以避免修改原始数据
        converted_result = []
        for item in items:
            converted_item = item.copy() if isinstance(item, dict) else item
            value = VADResultParser.extract_value(converted_item)
            if value:
                converted_value = []
                for pair in value:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        s, e = pair
                        # 转换相对帧索引为绝对帧索引
                        if s is not None and s >= 0:
                            s = frames_offset + s
                        if e is not None and e >= 0:
                            e = frames_offset + e
                        converted_value.append([s, e])
                    else:
                        converted_value.append(pair)
                if isinstance(converted_item, dict):
                    converted_item["value"] = converted_value
            converted_result.append(converted_item)
        
        return converted_result


class SlidingWindowManager:
    """
    职责：维护滑动窗口（支持步长和累加策略）
    
    滑动窗口策略：
    1. 窗口长度：55-65s（VAD片段累加近似值，目标60s）
    2. 步长：25-35s（近似30s）
    3. 第一个窗口前：每10-15s累加更新（例如：[0s,12s]→[0s,28s]→[0s,45s]→[0s,60s]）
    4. 第一个窗口：累加到60s左右触发
    5. 后续窗口：每步长30s触发一次
    
    时间戳对齐关键点：
    - window_start必须是VAD片段的起始位置（用于时间戳转换）
    - window_end必须是VAD片段的结束位置
    """
    # 窗口长度配置（秒）
    WINDOW_MIN_SECONDS = 55
    WINDOW_MAX_SECONDS = 65
    WINDOW_TARGET_SECONDS = 60
    
    # 步长配置（秒）
    STEP_MIN_SECONDS = 25
    STEP_MAX_SECONDS = 35
    STEP_TARGET_SECONDS = 30
    
    # 累加更新间隔（秒）
    UPDATE_MIN_SECONDS = 10
    UPDATE_MAX_SECONDS = 15
    
    SAMPLES_PER_SECOND = 16000

    @staticmethod
    def should_trigger_offline_asr(cache: Dict, new_segments: List[Tuple[int, int]], force_trigger: bool = False) -> bool:
        """
        判断是否应该触发离线ASR
        
        策略：
        1. 第一个窗口前：每10-15s累加更新
        2. 第一个窗口：累加到55-65s触发（目标60s）
        3. 后续窗口：距离上次触发25-35s时触发（目标30s步长）
        
        ⚠️ 核心设计：只在有新VAD片段时才检查触发条件
           - 流式VAD持续生成片段
           - 有新片段时检查是否达到触发条件
           - 不需要"强制触发"机制（除了finalize时）
        
        Args:
            cache: 流式处理缓存
            new_segments: 本轮新完成的片段列表
            force_trigger: 是否强制触发（finalize时使用，默认False）
        
        Returns:
            True表示应该触发离线ASR，False表示不触发
        """
        # finalize时强制触发，不论是否满足触发条件
        if force_trigger:
            logger.info(f"[SlidingWindow] 🔥 强制触发（finalize阶段）")
            return True
        
        # 只在有新片段时检查（符合用户设计逻辑）
        if not new_segments:
            return False
        
        segments = cache.get("segments", [])
        if not segments:
            return False
        
        # 计算当前窗口的总时长
        current_window_start = segments[0][0]
        current_window_end = segments[-1][1]
        current_duration_samples = current_window_end - current_window_start
        current_duration_sec = current_duration_samples / SlidingWindowManager.SAMPLES_PER_SECOND
        
        first_trigger_done = cache.get("first_offline_trigger_done", False)
        last_trigger_end = cache.get("last_offline_trigger_end", 0)
        last_update_end = cache.get("last_offline_update_end", 0)
        
        if not first_trigger_done:
            # ========== 第一个窗口前的累加更新策略 ==========
            # 检查是否应该进行累加更新（10-15s间隔）
            update_duration_samples = current_window_end - last_update_end
            update_duration_sec = update_duration_samples / SlidingWindowManager.SAMPLES_PER_SECOND
            
            # 检查是否达到第一个正式窗口的触发条件（55-65s）
            if (SlidingWindowManager.WINDOW_MIN_SECONDS <= current_duration_sec <= 
                SlidingWindowManager.WINDOW_MAX_SECONDS):
                logger.info(f"[SlidingWindow] ✅ 第一个窗口触发条件满足：时长={current_duration_sec:.2f}s")
                return True
            
            # 如果还没到60s，检查是否应该累加更新
            if (SlidingWindowManager.UPDATE_MIN_SECONDS <= update_duration_sec <= 
                SlidingWindowManager.UPDATE_MAX_SECONDS):
                logger.info(f"[SlidingWindow] 🔄 累加更新触发：时长={current_duration_sec:.2f}s "
                      f"(距上次更新{update_duration_sec:.2f}s)")
                return True
            
            # 如果窗口已经超过65s，说明触发条件一直没满足，这时触发
            if current_duration_sec > SlidingWindowManager.WINDOW_MAX_SECONDS:
                logger.warning(f"[SlidingWindow] ⚠️ 窗口超过{SlidingWindowManager.WINDOW_MAX_SECONDS}s，触发处理: "
                      f"时长={current_duration_sec:.2f}s")
                return True
            
            return False
        
        else:
            # ========== 后续窗口的步长策略 ==========
            # 计算距离上次触发的步长
            step_duration_samples = current_window_end - last_trigger_end
            step_duration_sec = step_duration_samples / SlidingWindowManager.SAMPLES_PER_SECOND
            
            # 检查步长是否在25-35s范围内
            if (SlidingWindowManager.STEP_MIN_SECONDS <= step_duration_sec <= 
                SlidingWindowManager.STEP_MAX_SECONDS):
                logger.info(f"[SlidingWindow] ✅ 步长触发条件满足：步长={step_duration_sec:.2f}s")
                return True
            
            # 如果步长已经超过35s，说明触发条件一直没满足，这时触发
            if step_duration_sec > SlidingWindowManager.STEP_MAX_SECONDS:
                logger.warning(f"[SlidingWindow] ⚠️ 步长超过{SlidingWindowManager.STEP_MAX_SECONDS}s，触发处理: "
                      f"步长={step_duration_sec:.2f}s")
                return True
            
            return False

    @staticmethod
    def calculate_window_range(cache: Dict) -> Tuple[Optional[int], Optional[int]]:
        """
        计算滑动窗口的范围
        
        策略：
        1. 第一个窗口：从流式VAD第一个片段开始，到最后一个片段结束
        2. 后续窗口（重要改进）：
           - 起始位置：从离线处理过的精细VAD片段中找（避免重复处理）
           - 结束位置：从流式VAD片段中找
           - 目标：[上次结束+30s附近的离线VAD片段起始, 当前流式VAD片段结束]
        
        ⚠️ 时间戳对齐关键：
        - window_start必须对应某个VAD片段的起始位置（离线VAD精细片段或流式VAD片段）
        - 这样离线ASR内部VAD返回的相对时间戳才能正确转换为全局时间戳
        
        Returns:
            (window_start, window_end) 元组：
            - window_start: 窗口起始位置（采样点，必须是VAD片段的起始位置）
            - window_end: 窗口结束位置（采样点，必须是VAD片段的结束位置）
        """
        segments = cache.get("segments", [])  # 流式VAD片段
        if not segments:
            return None, None
        
        first_trigger_done = cache.get("first_offline_trigger_done", False)
        last_trigger_end = cache.get("last_offline_trigger_end", 0)
        offline_processed = cache.get("offline_processed_segments", [])  # 离线处理过的精细片段
        
        if not first_trigger_done:
            # ========== 第一个窗口：从头开始累加（使用流式VAD片段）==========
            window_start = segments[0][0]  # 第一个流式VAD片段的起始位置
            window_end = segments[-1][1]    # 最后一个流式VAD片段的结束位置
            
            logger.info(f"[SlidingWindow] 第一个窗口计算（流式VAD）: "
                  f"[{window_start/16000:.2f}s, {window_end/16000:.2f}s], "
                  f"时长={(window_end-window_start)/16000:.2f}s, "
                  f"包含{len(segments)}个流式VAD片段")
            
            return window_start, window_end
        
        else:
            # ========== 后续窗口：混合使用离线VAD和流式VAD ==========
            # 计算目标起始位置（步长30s）
            target_start_sample = last_trigger_end + SlidingWindowManager.STEP_TARGET_SECONDS * 16000
            target_start_sec = target_start_sample / 16000.0
            
            # ⚠️ 关键修复：确保window_start永远不会小于last_trigger_end
            # 即使从offline_processed_segments中查找，也要确保不回溯到已处理的范围
            safe_target_sample = max(target_start_sample, last_trigger_end)
            safe_target_sec = safe_target_sample / 16000.0
            
            # Step 1: 从离线处理过的精细片段中找起始位置
            window_start = None
            offline_start_found = False
            
            if offline_processed:
                # 从离线精细片段中找第一个起始位置 >= safe_target_sample 的片段
                # safe_target 确保不会回溯到last_trigger_end之前
                for seg_start, seg_end in offline_processed:
                    if seg_start >= safe_target_sample:
                        window_start = seg_start
                        offline_start_found = True
                        logger.info(f"[SlidingWindow] ✅ 从离线精细VAD片段中找到起始位置: "
                              f"{window_start/16000:.2f}s (目标{target_start_sec:.2f}s, "
                              f"安全阈值{safe_target_sec:.2f}s)")
                        break
            
            # Step 2: 如果离线片段中没找到，从流式VAD片段中找
            if window_start is None:
                for seg_start, seg_end in segments:
                    if seg_start >= safe_target_sample:  # ← 使用safe_target防止回溯
                        window_start = seg_start
                        logger.info(f"[SlidingWindow] 从流式VAD片段中找到起始位置: "
                              f"{window_start/16000:.2f}s (目标{target_start_sec:.2f}s, "
                              f"安全阈值{safe_target_sec:.2f}s)")
                        break
            
            # Step 3: 如果还是没找到，使用第一个流式VAD片段
            if window_start is None:
                window_start = segments[0][0]
                logger.warning(f"[SlidingWindow] ⚠️ 未找到合适起始片段，使用第一个流式VAD片段: "
                      f"{window_start/16000:.2f}s")
            
            # Step 4: 从流式VAD片段中找结束位置（累加到55-65s）
            target_end_sample = window_start + SlidingWindowManager.WINDOW_TARGET_SECONDS * 16000
            window_end = segments[-1][1]  # 默认到最后一个流式VAD片段
            
            for seg_start, seg_end in segments:
                # 如果这个片段的结束位置超过了目标结束位置
                if seg_end >= target_end_sample:
                    window_end = seg_end
                    break
            
            window_duration_sec = (window_end - window_start) / 16000.0
            step_duration_sec = (window_start - last_trigger_end) / 16000.0
            
            logger.info(f"[SlidingWindow] 后续窗口计算: "
                  f"[{window_start/16000:.2f}s, {window_end/16000:.2f}s], "
                  f"时长={window_duration_sec:.2f}s, "
                  f"步长={step_duration_sec:.2f}s")
            
            if offline_start_found:
                logger.info(f"[SlidingWindow] ℹ️ 起始位置使用离线精细VAD片段（避免重复处理）")
            
            return window_start, window_end
    
    @staticmethod
    def update_trigger_state(cache: Dict, window_end: int, is_first_trigger: bool):
        """
        更新触发状态
        
        Args:
            cache: 流式处理缓存
            window_end: 本次窗口的结束位置
            is_first_trigger: 是否是第一次正式触发（达到60s）
        """
        if is_first_trigger:
            cache["first_offline_trigger_done"] = True
            cache["last_offline_trigger_end"] = window_end
            logger.info(f"[SlidingWindow] ✅ 第一次正式触发完成，记录触发结束位置: {window_end/16000:.2f}s")
        else:
            if cache.get("first_offline_trigger_done", False):
                # 后续触发
                cache["last_offline_trigger_end"] = window_end
                logger.info(f"[SlidingWindow] ✅ 后续触发完成，更新触发结束位置: {window_end/16000:.2f}s")
            else:
                # 累加更新
                cache["last_offline_update_end"] = window_end
                logger.info(f"[SlidingWindow] 🔄 累加更新完成，记录更新结束位置: {window_end/16000:.2f}s")
    
    @staticmethod
    def cleanup_old_segments(cache: Dict, keep_after: int):
        """
        清理旧的VAD片段（内存优化）
        
        策略：保留keep_after之后的所有片段
        
        Args:
            cache: 流式处理缓存
            keep_after: 保留此采样点之后的片段
        """
        segments = cache.get("segments", [])
        if not segments:
            return
        
        # 找到第一个结束位置在keep_after之后的片段
        new_segments = []
        for seg_start, seg_end in segments:
            if seg_end > keep_after:
                new_segments.append((seg_start, seg_end))
        
        removed_count = len(segments) - len(new_segments)
        if removed_count > 0:
            cache["segments"] = new_segments
            logger.info(f"[SlidingWindow] 清理了{removed_count}个旧片段，"
                  f"保留{len(new_segments)}个片段（{keep_after/16000:.2f}s之后）")

