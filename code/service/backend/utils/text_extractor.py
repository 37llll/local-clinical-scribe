"""
文本提取工具

功能：
- 从 ASR 模型结果中提取文本
- 支持多种 ASR 结果格式
"""

from typing import Any


class TextExtractor:
    """
    ASR 结果文本提取器
    
    功能：
    - 从 ASR 模型返回的结果中提取文本
    - 支持多种结果格式（列表、字典、字符串等）
    """
    
    @staticmethod
    def extract_text(asr_result: Any) -> str:
        """
        从 ASR 结果中提取文本
        
        Args:
            asr_result: ASR 模型返回的结果（多种可能格式）
                - 列表格式: [{"text": "..."}, ...] 或 ["...", ...]
                - 字典格式: {"text": "..."}
                - 字符串格式: "..."
                - None 或空值
        
        Returns:
            识别文本字符串，如果提取失败返回空字符串
        """
        if not asr_result:
            return ""
        
        # 处理列表格式
        if isinstance(asr_result, list):
            if len(asr_result) == 0:
                return ""
            first_item = asr_result[0]
            
            # 列表中的字典
            if isinstance(first_item, dict):
                return first_item.get("text", "")
            # 列表中的字符串
            elif isinstance(first_item, str):
                return first_item
        
        # 处理字典格式
        if isinstance(asr_result, dict):
            return asr_result.get("text", "")
        
        # 处理字符串格式
        if isinstance(asr_result, str):
            return asr_result
        
        return ""

