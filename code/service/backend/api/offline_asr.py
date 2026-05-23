"""
离线语音识别 API

功能：
- 上传完整音频文件进行非流式识别
- 支持说话人识别（聚类/匹配）
- 返回高精度识别结果（带标点符号）

提供 REST API 接口：
- POST /offline_asr - 上传音频文件进行识别
"""

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from backend.models.model_manager import ModelManager
from backend.services.offline_asr_service import OfflineASRService
from backend.utils.file_utils import parse_speaker_list
from backend.utils.logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()

router = APIRouter(prefix="/api/offline_asr", tags=["离线语音识别"])

# ASR models are loaded lazily on the first audio request. This keeps the
# product API and docs usable on machines that have not downloaded models yet.
_offline_asr_service = None


def get_offline_asr_service() -> OfflineASRService:
    global _offline_asr_service
    if _offline_asr_service is None:
        model_manager = ModelManager()
        _offline_asr_service = OfflineASRService(model_manager)
    return _offline_asr_service


@router.post("/offline_asr")
async def offline_asr(
    audio_file: UploadFile = File(..., description="音频文件（支持wav, mp3等格式）"),
    enable_speaker_diarization: bool = Form(True, description="是否启用说话人识别（默认True）"),
    speaker_mode: str = Form("cluster", description="说话人识别模式：cluster/cluster_match/direct_match"),
    registered_speakers: str = Form(None, description="已注册说话人名称，逗号分隔（如：'doctor,patient'）"),
    similarity_threshold: float = Form(0.5, description="相似度阈值（默认0.5）")
):
    """
    非流式ASR接口：上传完整音频文件，返回识别结果（支持说话人识别）
    
    **功能说明：**
    1. 非流式VAD检测语音片段
    2. 对每个片段调用非流式ASR
    3. 标点补全
    4. 可选：说话人识别（聚类/匹配）
    
    **输入参数：**
    - `audio_file`: 音频文件（支持wav, mp3等格式，librosa会自动处理）
    - `enable_speaker_diarization`: 是否启用说话人识别（默认True）
    - `speaker_mode`: 说话人识别模式
      - `"cluster"`: 仅聚类，自动识别说话人数量（默认）
      - `"cluster_match"`: 聚类后与已注册说话人匹配
      - `"direct_match"`: 直接匹配每个片段
    - `registered_speakers`: 已注册说话人名称，逗号分隔（如："doctor,patient"）
    - `similarity_threshold`: 相似度阈值（默认0.5，范围0-1）
    
    **输出格式：**
    ```json
    {
        "success": true,
        "text": "完整识别文本（带标点符号）",
        "duration": 60.5,              // 音频时长（秒）
        "segments": [                   // 如果启用说话人识别
            {
                "start": 1.23,          // 开始时间（秒）
                "end": 3.45,            // 结束时间（秒）
                "text": "识别文本（带标点）",
                "speaker": "speaker_0", // 说话人标识
                "similarity": 0.85,     // 相似度分数（如果有匹配）
                "start_sample": 19680,  // 开始采样点
                "end_sample": 55200     // 结束采样点
            },
            ...
        ]
    }
    ```
    
    **错误响应：**
    ```json
    {
        "success": false,
        "error": "错误信息"
    }
    ```
    
    **使用场景：**
    - 上传完整音频文件进行识别
    - 需要高精度识别结果（带标点符号）
    - 需要说话人识别（区分不同说话人）
    - 不需要实时性，可以等待完整处理
    
    **示例请求：**
    ```bash
    curl -X POST "http://localhost:63100/api/offline_asr/offline_asr" \
         -F "audio_file=@audio.wav" \
         -F "enable_speaker_diarization=true" \
         -F "speaker_mode=cluster_match" \
         -F "registered_speakers=doctor,patient" \
         -F "similarity_threshold=0.5"
    ```
    
    **示例响应：**
    ```json
    {
        "success": true,
        "text": "你好，我是医生。请问你有什么症状？",
        "duration": 5.2,
        "segments": [
            {
                "start": 0.0,
                "end": 2.1,
                "text": "你好，我是医生。",
                "speaker": "doctor",
                "similarity": 0.92,
                "start_sample": 0,
                "end_sample": 33600
            },
            {
                "start": 2.5,
                "end": 5.2,
                "text": "请问你有什么症状？",
                "speaker": "doctor",
                "similarity": 0.88,
                "start_sample": 40000,
                "end_sample": 83200
            }
        ]
    }
    ```
    """
    try:
        logger.info(f"[API] 说话人识别: {enable_speaker_diarization}, 模式: {speaker_mode}")
        
        # 解析已注册说话人列表
        registered_speakers_list = parse_speaker_list(registered_speakers)
        if registered_speakers_list:
            logger.info(f"[API] 已注册说话人: {registered_speakers_list}")
        
        # 调用 Service 层处理上传的文件
        result = await get_offline_asr_service().process_uploaded_file(
            upload_file=audio_file,
            enable_speaker_diarization=enable_speaker_diarization,
            speaker_mode=speaker_mode,
            registered_speakers=registered_speakers_list,
            similarity_threshold=similarity_threshold
        )
        
        return JSONResponse(content={
            "success": True,
            **result
        })
        
    except Exception as e:
        logger.error(f"[API] 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

