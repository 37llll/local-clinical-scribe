"""
API 模块

功能：
- 提供所有 REST 和 WebSocket API 接口
- streaming_asr: 流式语音识别接口
- offline_asr: 离线语音识别接口
- speaker: 说话人注册管理接口
"""

from backend.api.streaming_asr import router as streaming_asr_router
from backend.api.offline_asr import router as offline_asr_router
from backend.api.speaker import router as speaker_router

__all__ = [
    "streaming_asr_router",
    "offline_asr_router",
    "speaker_router",
]

