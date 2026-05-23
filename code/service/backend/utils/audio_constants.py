"""
音频处理相关常量

功能：
- 定义音频处理中使用的常量
- 提供采样率和时间单位转换的常量

说明：
FunASR 流式 VAD 返回毫秒（ms），常用换算：
- ms → samples: samples = ms * 16  （16kHz）
- samples → ms: ms = samples / 16
- samples → seconds: seconds = samples / 16000
"""


class AudioConstants:
    """音频处理相关常量类"""
    SAMPLE_RATE = 16000
    MS_TO_SAMPLES = 16  # 1ms = 16 samples @16kHz

