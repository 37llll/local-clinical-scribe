"""
WebSocket 音频消息解析器

功能：
- 解析 WebSocket 消息中的音频数据
- 支持二进制消息（bytes）和 JSON 消息（text）
- 统一的错误处理和音频数据提取
- 音频格式转换和重采样
"""

import json
import numpy as np
from typing import Tuple, Optional
from fastapi import WebSocket, WebSocketDisconnect

from .audio_processing import resample_audio
from .audio_constants import AudioConstants
from .logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class AudioMessageParser:
    """
    WebSocket 音频消息解析器
    
    功能：
    - 解析 WebSocket 消息（二进制或 JSON 格式）
    - 提取音频数据和采样率
    - 统一错误处理和音频重采样
    """
    
    def __init__(self, target_sample_rate: int = None):
        """
        初始化解析器
        
        Args:
            target_sample_rate: 目标采样率（默认使用 AudioConstants.SAMPLE_RATE）
        """
        self.target_sample_rate = target_sample_rate or AudioConstants.SAMPLE_RATE
    
    def parse_message(
        self, 
        message: dict
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        解析 WebSocket 消息，提取音频数据和采样率
        
        Args:
            message: WebSocket 消息字典，包含 "bytes" 或 "text" 字段
            
        Returns:
            (audio, sample_rate) 元组：
            - audio: 音频数据（numpy数组，float32），如果解析失败返回 None
            - sample_rate: 采样率（int），如果解析失败返回 None
            
        支持的输入格式：
        1. 二进制消息：{"bytes": b"...", "type": "websocket.receive"}
           - 默认采样率：16kHz
        2. JSON消息：{"text": '{"audio": [...], "sample_rate": 16000}', "type": "websocket.receive"}
           - 可以指定采样率
        """
        raw_bytes = message.get("bytes")
        raw_text = message.get("text")
        
        # 处理二进制消息
        if raw_bytes is not None:
            try:
                audio = np.frombuffer(raw_bytes, dtype=np.float32)
                return audio, self.target_sample_rate
            except Exception as e:
                logger.error(f"[AudioMessageParser] 二进制消息解析失败: {e}")
                return None, None
        
        # 处理 JSON 文本消息
        if raw_text is not None:
            try:
                payload = json.loads(raw_text)
                
                # 提取音频数据
                if isinstance(payload, dict):
                    audio_list = payload.get("audio")
                    sample_rate = int(payload.get("sample_rate", self.target_sample_rate))
                else:
                    # 如果 payload 直接是数组
                    audio_list = payload
                    sample_rate = self.target_sample_rate
                
                if audio_list is None:
                    return None, None
                
                audio = np.array(audio_list, dtype=np.float32)
                return audio, sample_rate
                
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.error(f"[AudioMessageParser] JSON消息解析失败: {e}")
                return None, None
        
        # 既没有 bytes 也没有 text
        return None, None
    
    async def parse_and_resample(
        self,
        message: dict,
        websocket: WebSocket
    ) -> Optional[np.ndarray]:
        """
        解析消息并重采样到目标采样率，包含错误处理
        
        Args:
            message: WebSocket 消息字典
            websocket: WebSocket 连接对象（用于发送错误消息）
            
        Returns:
            重采样后的音频数据（16kHz），如果解析失败返回 None
            
        注意：
            - 如果解析失败，会自动发送错误消息到客户端
            - 如果 WebSocket 断开，会抛出 WebSocketDisconnect 异常
        """
        audio, sample_rate = self.parse_message(message)
        
        # 检查解析结果
        if audio is None:
            try:
                await websocket.send_json({"error": "无法解析音频数据"})
            except (WebSocketDisconnect, RuntimeError):
                pass  # 连接已断开，无需处理
            return None
        
        if len(audio) == 0:
            try:
                await websocket.send_json({"error": "音频数据为空"})
            except (WebSocketDisconnect, RuntimeError):
                pass
            return None
        
        # 重采样到目标采样率
        try:
            audio_16k = resample_audio(audio, sample_rate, self.target_sample_rate)
            return audio_16k
        except Exception as e:
            logger.error(f"[AudioMessageParser] 音频重采样失败: {e}")
            try:
                await websocket.send_json({"error": f"音频处理失败: {str(e)}"})
            except (WebSocketDisconnect, RuntimeError):
                pass
            return None
    
    @staticmethod
    async def send_error(websocket: WebSocket, error_message: str) -> bool:
        """
        发送错误消息到客户端
        
        Args:
            websocket: WebSocket 连接对象
            error_message: 错误消息
            
        Returns:
            是否发送成功（如果连接已断开返回 False）
        """
        try:
            await websocket.send_json({"error": error_message})
            return True
        except (WebSocketDisconnect, RuntimeError):
            return False

