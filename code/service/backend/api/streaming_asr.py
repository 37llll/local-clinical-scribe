"""
流式语音识别 API

功能：
- 流式VAD检测（实时检测语音片段）
- 流式ASR识别（实时语音转文字）
- 完整流式处理管线（VAD + 流式ASR + 离线ASR + 说话人识别）

提供三个 WebSocket 接口：
1. /vad - 仅流式VAD检测
2. /streaming_asr - 流式VAD + 流式ASR（低延迟）
3. /pipeline - 完整处理管线（累积结果模式）
"""

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional, List

from backend.models.model_manager import ModelManager
from backend.services.streaming_asr_service import StreamingASRService
from backend.utils.audio_message_parser import AudioMessageParser
from backend.utils.audio_constants import AudioConstants
from backend.utils.file_utils import parse_speaker_list
from backend.utils.logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()

router = APIRouter(prefix="/api/stream_asr", tags=["流式语音识别"])

# Streaming models are loaded lazily when a websocket endpoint is used.
_asr_service = None


def get_asr_service() -> StreamingASRService:
    global _asr_service
    if _asr_service is None:
        model_manager = ModelManager()
        _asr_service = StreamingASRService(model_manager)
    return _asr_service

# 音频消息解析器（单例）
audio_parser = AudioMessageParser(target_sample_rate=AudioConstants.SAMPLE_RATE)


@router.websocket("/vad")
async def vad(websocket: WebSocket):
    """
    WebSocket 接口：流式VAD检测
    
    **功能说明：**
    实时接收音频数据流，使用流式VAD模型检测语音片段，返回时间戳。
    
    **输入格式：**
    - 二进制消息：16kHz PCM float32 音频数据
    - JSON消息：{"audio": [floats], "sample_rate": 16000}
    
    **输出格式：**
    ```json
    {
        "result": [
            [start_ms, -1],      // 检测到片段开始
            [-1, end_ms],        // 检测到片段结束
            [start_ms, end_ms]   // 完整片段（开始+结束）
        ]
    }
    ```
    
    **时间戳说明：**
    - 单位：毫秒（ms）
    - 流式VAD通过cache维护内部状态，返回的已经是绝对毫秒时间戳
    - 无需手动转换帧索引
    
    **使用场景：**
    - 仅需要VAD检测，不需要ASR识别
    - 前端需要实时知道语音片段的开始和结束时间
    
    **示例：**
    ```javascript
    const ws = new WebSocket('ws://localhost:63100/api/stream_asr/vad');
    ws.send(JSON.stringify({
        audio: [0.1, 0.2, ...],  // float32数组
        sample_rate: 16000
    }));
    ```
    """
    await websocket.accept()
    asr_service = get_asr_service()
    # VAD cache：流式VAD通过cache维护内部状态，返回的已经是绝对毫秒时间戳
    vad_cache = asr_service.create_session_cache()
    
    try:
        while True:
            message = await websocket.receive()
            # 客户端主动断开时直接退出循环
            if message.get("type") == "websocket.disconnect":
                break
            
            # 使用 AudioMessageParser 解析消息并重采样
            audio_16k = await audio_parser.parse_and_resample(message, websocket)
            if audio_16k is None:
                continue
            
            # 调用流式VAD（返回的时间戳已经是毫秒，通过cache维护绝对位置）
            vad_result = asr_service.processor.process_streaming_vad(audio_16k, vad_cache)
            
            try:
                await websocket.send_json({"result": vad_result})
            except (WebSocketDisconnect, RuntimeError):
                break
                
    except WebSocketDisconnect:
        # 客户端主动断开，无需额外处理
        return


@router.websocket("/streaming_asr")
async def streaming_asr(websocket: WebSocket):
    """
    WebSocket 接口：流式VAD + 流式ASR
    
    **功能说明：**
    实时接收音频数据流，进行流式VAD检测和流式ASR识别，返回实时识别结果。
    不包含离线ASR处理，延迟最低。
    
    **输入格式：**
    - 二进制消息：16kHz PCM float32 音频数据
    - JSON消息：{"audio": [floats], "sample_rate": 16000}
    
    **输出格式：**
    ```json
    {
        "result": {
            "vad": [...],               // 本 chunk 的 VAD 原始输出（流式标记，格式同 /vad）
            "streaming_asr": ...        // 本 chunk 触发的流式 ASR 原始结果（每 3 个 chunk 触发一次，未触发时为 null）
        }
    }
    ```
    
    **使用场景：**
    - 需要实时低延迟的语音识别
    - 不需要高精度和标点符号
    - 不需要说话人识别
    
    **示例：**
    ```javascript
    const ws = new WebSocket('ws://localhost:63100/api/stream_asr/streaming_asr');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const r = data.result;
        if (r.streaming_asr != null) {
            // 从流式 ASR 原始结果中提取文本（格式取决于模型返回）
            console.log('识别结果:', r.streaming_asr);
        }
    };
    ```
    """
    await websocket.accept()
    asr_service = get_asr_service()
    # 每个WebSocket连接维护自己的流式cache（包含VAD/ASR缓存，不包含音频缓存）
    streaming_cache = asr_service.create_session_cache()
    
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            
            # 使用 AudioMessageParser 解析消息并重采样
            audio_16k = await audio_parser.parse_and_resample(message, websocket)
            if audio_16k is None:
                continue

            # 调用流式管线（仅VAD+流式ASR）
            result = asr_service.process_chunk_streaming_only(audio_16k, cache=streaming_cache)
            
            try:
                await websocket.send_json({"result": result})
            except (WebSocketDisconnect, RuntimeError):
                break
                
    except WebSocketDisconnect:
        return


@router.websocket("/pipeline")
async def pipeline(websocket: WebSocket):
    """
    WebSocket 接口：完整流式处理管线（累积结果模式）
    
    **功能说明：**
    实时接收音频数据流，进行完整的流式处理：
    1. 流式VAD检测语音片段
    2. 流式ASR实时识别（低延迟）
    3. 离线ASR高精度识别（VAD片段完成时）
    4. 说话人识别（可选，通过查询参数配置）
    5. 文本对齐和合并
    
    每次返回累积的完整结果，不是单次chunk的结果。
    
    **连接参数（查询参数，可选）：**
    - `enable_speaker_diarization`: 是否启用说话人识别（默认true）
      - `true` 或 `1`: 启用说话人识别
      - `false` 或 `0`: 禁用说话人识别
    - `speaker_mode`: 说话人识别模式（默认"cluster"）
      - `"cluster"`: 仅聚类，自动识别说话人数量
      - `"cluster_match"`: 聚类后与已注册说话人匹配
      - `"direct_match"`: 每个片段直接匹配最相似的说话人
    - `registered_speakers`: 已注册说话人名称，逗号分隔（如："doctor,patient"）
      - 仅在 `speaker_mode` 为 `cluster_match` 或 `direct_match` 时有效
    - `similarity_threshold`: 相似度阈值（默认0.5，范围0-1）
      - 仅在匹配模式（`cluster_match` 或 `direct_match`）时有效
    
    **输入格式：**
    - 二进制消息：16kHz PCM float32 音频数据
    - JSON消息：{"audio": [floats], "sample_rate": 16000}
    
    **输出格式：**
    ```json
    {
        "status": "success",
        
        // VAD相关结果
        "vad_raw": [...],                    // VAD原始输出（流式标记，仅在检测到边界时有数据）
        "vad_segments": [                    // VAD片段列表（累积所有已完成的片段）
            {
                "start": 1.23,               // 开始时间（秒）
                "end": 3.45                  // 结束时间（秒）
            }
        ],
        
        // 最终文本（推荐使用）
        "aligned_text": "完整对齐后的文本（流式+离线混合）",
        
        // 离线ASR片段（高精度+标点+说话人）
        "offline_segments": [
            {
                "start": 1.23,           // 开始时间（秒）
                "end": 3.45,              // 结束时间（秒）
                "text": "识别文本（带标点）",
                "speaker": "speaker_0",   // 说话人标识
                "start_sample": 19680,   // 开始采样点
                "end_sample": 55200,     // 结束采样点
                "similarity": 0.85       // 相似度分数（如果有匹配）
            }
        ],
        
        // 流式ASR历史（实时低延迟结果）
        "streaming_history": [
            {
                "text": "识别文本",
                "timestamp": 1.23,       // 时间戳（秒）
                "chunk_idx": 0           // chunk索引
            }
        ],
        
        // 会话统计信息
        "stats": {
            "total_audio_duration": 60.5,    // 总音频时长（秒）
            "vad_segments_count": 5,          // VAD检测的片段数
            "streaming_asr_count": 10,        // 流式ASR结果数
            "offline_asr_count": 5,            // 离线ASR片段数
            "chunk_counter": 302              // 处理的chunk数
        },
        
        // 调试信息
        "debug": {
            "new_segments_count": 1           // 本次新完成的VAD片段数
        }
    }
    ```
    
    **Cache设计：**
    - 每个WebSocket连接维护一套独立的pipeline_cache
    - 一个连接 = 一个完整的音频会话 = 一套完整的累积结果
    - cache包含：
      * 模型状态（VAD/ASR模型的内部cache）
      * 音频缓冲（60秒滑动窗口）
      * VAD片段（语音段检测结果）
      * 流式ASR结果累积（实时结果历史）
      * 离线ASR结果累积（高精度结果）
    - 连接断开时，cache自动释放
    
    **使用建议：**
    1. 前端应该使用`aligned_text`作为最终显示文本
    2. 如果需要说话人信息，使用`offline_segments`
    3. 如果需要实时性，可以展示`streaming_history`的最新结果
    4. 如果需要VAD片段信息，使用`vad_segments`（累积所有已完成的片段）
    5. `vad_raw`包含原始VAD输出（流式标记），主要用于调试
    6. `stats`可用于监控处理进度
    
    **示例（无说话人识别）：**
    ```javascript
    const ws = new WebSocket('ws://localhost:63100/api/stream_asr/pipeline?enable_speaker_diarization=false');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        document.getElementById('text').textContent = data.aligned_text;
    };
    ```
    
    **示例（使用说话人识别和已注册说话人匹配）：**
    ```javascript
    const ws = new WebSocket('ws://localhost:63100/api/stream_asr/pipeline?enable_speaker_diarization=true&speaker_mode=cluster_match&registered_speakers=doctor,patient&similarity_threshold=0.5');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // 显示最终文本
        document.getElementById('text').textContent = data.aligned_text;
        // 显示说话人信息
        data.offline_segments.forEach(segment => {
            console.log(`${segment.speaker}: ${segment.text}`);
        });
    };
    ```
    """
    await websocket.accept()
    asr_service = get_asr_service()
    
    # 解析查询参数（说话人识别配置）
    query_params = websocket.query_params
    
    # 解析 enable_speaker_diarization（默认True）
    enable_speaker_diarization = True
    if "enable_speaker_diarization" in query_params:
        value = query_params["enable_speaker_diarization"].lower()
        enable_speaker_diarization = value in ("true", "1", "yes")
    
    # 解析 speaker_mode（默认"cluster"）
    speaker_mode = query_params.get("speaker_mode", "cluster")
    if speaker_mode not in ("cluster", "cluster_match", "direct_match"):
        speaker_mode = "cluster"  # 默认值
    
    # 解析 registered_speakers（逗号分隔的字符串）
    registered_speakers_str = query_params.get("registered_speakers")
    registered_speakers: Optional[List[str]] = parse_speaker_list(registered_speakers_str)
    
    # 解析 similarity_threshold（默认0.5）
    similarity_threshold = 0.5
    if "similarity_threshold" in query_params:
        try:
            similarity_threshold = float(query_params["similarity_threshold"])
            # 限制范围在0-1之间
            similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        except (ValueError, TypeError):
            similarity_threshold = 0.5  # 默认值
    
    # 记录配置信息（用于调试）
    logger.info(f"[Pipeline] 说话人识别配置: enable={enable_speaker_diarization}, mode={speaker_mode}, "
          f"registered_speakers={registered_speakers}, threshold={similarity_threshold}")
    
    # 每个WebSocket连接维护自己的管线cache（包含VAD/ASR/audio缓存）
    pipeline_cache = asr_service.create_session_cache()
    
    try:
        while True:
            message = await websocket.receive()

            # 判断是否有结束信号
            if json.loads(message.get("text", "{}")).get("audio_end", False):
                logger.info("Pipeline 收到音频结束信号，开始finalize处理...")
                final_result = asr_service.finalize(
                    cache=pipeline_cache,
                    enable_speaker_diarization=enable_speaker_diarization,
                    speaker_mode=speaker_mode,
                    registered_speakers=registered_speakers,
                    similarity_threshold=similarity_threshold
                )
                
                # 格式化VAD片段
                vad_segments_formatted = []
                segments = pipeline_cache.get("segments", [])
                for start_sample, end_sample in segments:
                    vad_segments_formatted.append({
                        "start": start_sample / AudioConstants.SAMPLE_RATE,
                        "end": end_sample / AudioConstants.SAMPLE_RATE,
                    })
                
                # 发送最终结果
                final_response = {
                    "status": "final",
                    "vad_segments": vad_segments_formatted,
                    "aligned_text": final_result["aligned_text"],
                    "offline_segments": final_result["offline_asr_segments"],
                    "streaming_history": final_result["streaming_asr_history"],
                    "stats": final_result["session_stats"],
                }
                
                await websocket.send_json(final_response)
                
                logger.info(f"[Pipeline] Finalize处理完成")
            else:
                
                # 使用 AudioMessageParser 解析消息并重采样
                audio_16k = await audio_parser.parse_and_resample(message, websocket)
                if audio_16k is None:
                    continue

                # 调用pipeline处理（返回累积结果，传递说话人识别参数）
                result = asr_service.process_chunk(
                    audio_16k,
                    cache=pipeline_cache,
                    enable_speaker_diarization=enable_speaker_diarization,
                    speaker_mode=speaker_mode,
                    registered_speakers=registered_speakers,
                    similarity_threshold=similarity_threshold
                )
                
                # 格式化VAD片段（从cache中读取，转换为秒单位格式）
                vad_segments_formatted = []
                segments = pipeline_cache.get("segments", [])
                for start_sample, end_sample in segments:
                    vad_segments_formatted.append({
                        "start": start_sample / AudioConstants.SAMPLE_RATE,      # 开始时间（秒）
                        "end": end_sample / AudioConstants.SAMPLE_RATE,          # 结束时间（秒）
                    })
                
                # 返回格式化的结果
                response = {
                    "status": "success",
                    
                    # VAD相关结果
                    "vad_raw": result.get("vad_raw"),                           # VAD原始输出（流式标记）
                    "vad_segments": vad_segments_formatted,                    # VAD片段列表（格式化，累积所有片段）
                    
                    # 累积的完整结果（推荐前端使用）
                    "aligned_text": result["aligned_text"],                    # 对齐后的最终文本（流式+离线混合）
                    "offline_segments": result["offline_asr_segments"],        # 离线ASR片段（高精度+说话人）
                    "streaming_history": result["streaming_asr_history"],      # 流式ASR历史记录
                    
                    # 会话统计信息
                    "stats": result["session_stats"],
                    
                    # 调试信息（可选）
                    "debug": {
                        "new_segments_count": result["new_segments_count"],    # 本次新完成的VAD片段数
                    }
                }
                await websocket.send_json(response)

            
    except WebSocketDisconnect:
        logger.info("Pipeline WebSocket连接断开")
        return

