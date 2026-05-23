"""
结果对齐工具

功能：
- 对齐流式 ASR 和离线 ASR 结果
- 生成最终文本（优先使用高精度离线结果）
- 处理时间戳覆盖和文本合并
"""

from typing import Dict, List, Any


class ResultAligner:
    """
    结果对齐器
    
    功能：
    - 对齐流式 ASR 和离线 ASR 结果
    - 优先使用高精度的离线 ASR 结果
    - 对于未被离线 ASR 覆盖的时间段，使用流式 ASR 结果
    - 按时间顺序拼接所有文本
    """
    
    @staticmethod
    def align_results(cache: Dict[str, Any]) -> str:
        """
        对齐流式 ASR 和离线 ASR 结果，生成最终文本
        
        策略：
        1. 如果有离线 ASR 结果，优先使用（高精度+标点+说话人）
        2. 对于还没有离线 ASR 结果的时间段，使用流式 ASR 结果（实时性）
        3. 按时间顺序拼接所有文本
        
        Args:
            cache: 包含 streaming_asr_results 和 offline_asr_segments 的缓存
                - streaming_asr_results: 流式 ASR 结果列表
                - offline_asr_segments: 离线 ASR 片段列表
        
        Returns:
            对齐后的完整文本
        """
        streaming_results = cache.get("streaming_asr_results", [])
        offline_segments = cache.get("offline_asr_segments", [])
        
        # 如果没有任何结果，返回空
        if not streaming_results and not offline_segments:
            return ""
        
        # 如果只有流式结果，直接拼接
        if not offline_segments:
            return ResultAligner._join_streaming_results(streaming_results)
        
        # 如果只有离线结果，按时间顺序拼接（带说话人标签）
        if not streaming_results:
            return ResultAligner._join_offline_segments(offline_segments)
        
        # 同时有流式和离线结果：优先使用离线结果
        return ResultAligner._merge_results(streaming_results, offline_segments)
    
    @staticmethod
    def _join_streaming_results(streaming_results: List[Dict]) -> str:
        """
        拼接流式 ASR 结果
        
        Args:
            streaming_results: 流式 ASR 结果列表
        
        Returns:
            拼接后的文本
        """
        return "".join([r["text"] for r in streaming_results])
    
    @staticmethod
    def _join_offline_segments(offline_segments: List[Dict]) -> str:
        """
        拼接离线 ASR 片段（带说话人标签）
        
        Args:
            offline_segments: 离线 ASR 片段列表
        
        Returns:
            拼接后的文本
        """
        sorted_segments = sorted(offline_segments, key=lambda x: x["start"])
        return "".join([
            f"[{seg.get('speaker', '未知')}] {seg['text']}"
            for seg in sorted_segments
        ])
    
    @staticmethod
    def _merge_results(
        streaming_results: List[Dict],
        offline_segments: List[Dict]
    ) -> str:
        """
        合并流式和离线结果
        
        策略：对于每个离线 ASR 片段覆盖的时间范围，忽略流式 ASR 结果
        
        Args:
            streaming_results: 流式 ASR 结果列表
            offline_segments: 离线 ASR 片段列表
        
        Returns:
            合并后的文本
        """
        result_parts = []
        
        # 按时间排序离线片段
        sorted_offline = sorted(offline_segments, key=lambda x: x["start"])
        
        # 记录离线 ASR 已覆盖的时间范围
        covered_ranges = [(seg["start"], seg["end"]) for seg in sorted_offline]
        
        # 添加离线 ASR 结果（带说话人标签）
        for seg in sorted_offline:
            speaker = seg.get("speaker", "未知")
            text = seg.get("text", "")
            result_parts.append({
                "time": seg["start"],
                "text": f"[{speaker}] {text}",
                "type": "offline"
            })
        
        # 添加流式 ASR 结果（只添加未被离线 ASR 覆盖的部分）
        for stream_result in streaming_results:
            timestamp = stream_result["timestamp"]
            text = stream_result["text"]
            
            # 检查是否被离线 ASR 覆盖
            is_covered = ResultAligner._is_timestamp_covered(timestamp, covered_ranges)
            
            if not is_covered:
                result_parts.append({
                    "time": timestamp,
                    "text": text,
                    "type": "streaming"
                })
        
        # 按时间排序并拼接
        result_parts.sort(key=lambda x: x["time"])
        return "".join([part["text"] for part in result_parts])
    
    @staticmethod
    def _is_timestamp_covered(timestamp: float, covered_ranges: List[tuple]) -> bool:
        """
        检查时间戳是否被覆盖范围覆盖
        
        Args:
            timestamp: 时间戳（秒）
            covered_ranges: 覆盖范围列表 [(start, end), ...]
        
        Returns:
            如果被覆盖返回 True，否则返回 False
        """
        return any(start <= timestamp <= end for start, end in covered_ranges)

