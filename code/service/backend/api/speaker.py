"""
说话人注册管理 API

功能：
- 注册说话人声纹（从音频文件提取声纹特征）
- 查询已注册说话人列表
- 删除指定说话人
- 验证音频是否为指定说话人

提供 REST API 接口：
- POST /enroll - 注册说话人声纹
- GET /speakers - 查询已注册说话人列表
- DELETE /speakers/{speaker_name} - 删除指定说话人
- POST /verify - 验证音频是否为指定说话人
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from backend.models.model_manager import ModelManager
from backend.utils.file_utils import (
    TempFileManager,
    validate_file_format
)

router = APIRouter(prefix="/api/speaker", tags=["说话人注册管理"])

# 说话人注册处理器（懒加载，首次使用时自动初始化）
enrollment_processor = None


def get_enrollment_processor():
    """
    获取说话人注册处理器（懒加载）
    
    Returns:
        说话人注册处理器实例
    """
    global enrollment_processor
    if enrollment_processor is None:
        model_manager = ModelManager()
        enrollment_processor = model_manager.get_enrollment_model()
    return enrollment_processor


@router.post("/enroll", summary="注册说话人声纹")
async def enroll_speaker(
    audio_file: UploadFile = File(..., description="音频文件（WAV格式，16kHz，建议≥15秒）"),
    speaker_name: str = Form(..., description="说话人名称（作为唯一标识）"),
    overwrite: bool = Form(False, description="是否覆盖已存在的说话人")
):
    """
    注册说话人声纹
    
    **功能说明：**
    从上传的音频文件中提取说话人的声纹特征（embedding），
    保存到本地文件系统，用于后续的说话人识别和验证。
    
    **输入参数：**
    - `audio_file`: WAV音频文件（推荐16kHz采样率，时长≥15秒，单声道）
    - `speaker_name`: 说话人名称（字符串，不能为空，作为唯一标识）
    - `overwrite`: 是否覆盖已存在的说话人（默认False）
    
    **输出格式（成功）：**
    ```json
    {
        "success": true,
        "message": "说话人 'doctor' 注册成功",
        "data": {
            "speaker_name": "doctor",
            "embedding_path": "/path/to/speaker_embedding/doctor.npy",
            "audio_duration": 18.5,
            "embedding_dim": 192
        }
    }
    ```
    
    **输出格式（失败 - 已存在）：**
    ```json
    {
        "success": false,
        "message": "说话人 'doctor' 已存在，如需覆盖请设置overwrite=True",
        "data": null
    }
    ```
    
    **输出格式（失败 - 其他错误）：**
    ```json
    {
        "detail": "注册失败: 错误信息"
    }
    ```
    
    **使用场景：**
    - 注册医生、患者等特定说话人的声纹
    - 为说话人识别功能准备声纹库
    - 支持后续的说话人匹配和验证
    
    **注意事项：**
    - 音频文件必须是WAV格式
    - 建议音频时长≥15秒，以获得更好的声纹特征
    - 要求使用16kHz采样率
    - speaker_name不能包含特殊字符（建议使用字母、数字、下划线）
    
    **示例请求：**
    ```bash
    curl -X POST "http://localhost:63100/api/speaker/enroll" \
         -F "audio_file=@doctor_voice.wav" \
         -F "speaker_name=doctor" \
         -F "overwrite=false"
    ```
    """
    try:
        # 获取处理器
        processor = get_enrollment_processor()
        
        # 检查文件格式
        if not validate_file_format(audio_file.filename, ('.wav', '.WAV')):
            raise HTTPException(
                status_code=400,
                detail="只支持WAV格式音频文件"
            )
        
        # 使用临时文件管理器保存上传的文件
        with TempFileManager(suffix=".wav") as tmp_manager:
            tmp_path = await tmp_manager.save_upload_file(audio_file)
        
            # 调用注册处理器
            result = processor.enroll(
                audio_path=tmp_path,
                speaker_name=speaker_name,
                overwrite=overwrite
            )
            
            if result["success"]:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "message": result["message"],
                        "data": {
                            "speaker_name": result["speaker_name"],
                            "embedding_path": result["embedding_path"],
                            "audio_duration": result["audio_duration"],
                            "embedding_dim": result["embedding_dim"]
                        }
                    }
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": result["message"],
                        "data": None
                    }
                )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"注册失败: {str(e)}"
        )


@router.get("/speakers", summary="查询已注册说话人列表")
async def list_speakers():
    """
    查询所有已注册说话人列表
    
    **功能说明：**
    返回所有已注册说话人的名称列表，以及相关的统计信息。
    
    **输入参数：**
    无（GET请求）
    
    **输出格式（成功）：**
    ```json
    {
        "success": true,
        "data": {
            "speakers": ["doctor", "patient", "nurse"],
            "count": 3,
            "embedding_dir": "/path/to/speaker_embedding"
        }
    }
    ```
    
    **输出格式（失败）：**
    ```json
    {
        "success": false,
        "message": "查询失败: 错误信息",
        "data": null
    }
    ```
    
    **使用场景：**
    - 查看当前已注册的所有说话人
    - 获取说话人数量统计
    - 前端展示说话人列表供用户选择
    
    **示例请求：**
    ```bash
    curl -X GET "http://localhost:63100/api/speaker/speakers"
    ```
    
    **示例响应：**
    ```json
    {
        "success": true,
        "data": {
            "speakers": ["doctor", "patient"],
            "count": 2,
            "embedding_dir": "data/speaker_embedding"
        }
    }
    ```
    """
    try:
        # 获取处理器
        processor = get_enrollment_processor()
        
        # 查询列表
        result = processor.list_speakers()
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": {
                        "speakers": result["speakers"],
                        "count": result["count"],
                        "embedding_dir": result["embedding_dir"]
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": result.get("error", "查询失败"),
                    "data": None
                }
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"查询失败: {str(e)}"
        )


@router.delete("/speakers/{speaker_name}", summary="删除指定说话人")
async def delete_speaker(speaker_name: str):
    """
    删除指定说话人的声纹文件
    
    **功能说明：**
    从文件系统中删除指定说话人的声纹特征文件。
    
    **输入参数：**
    - `speaker_name`: 说话人名称（路径参数）
    
    **输出格式（成功）：**
    ```json
    {
        "success": true,
        "message": "说话人 'doctor' 已删除",
        "data": {
            "speaker_name": "doctor"
        }
    }
    ```
    
    **输出格式（失败 - 不存在）：**
    ```json
    {
        "success": false,
        "message": "说话人 'unknown' 不存在",
        "data": null
    }
    ```
    
    **输出格式（失败 - 其他错误）：**
    ```json
    {
        "detail": "删除失败: 错误信息"
    }
    ```
    
    **使用场景：**
    - 删除不再需要的说话人声纹
    - 清理测试数据
    - 重新注册说话人（先删除再注册）
    
    **示例请求：**
    ```bash
    curl -X DELETE "http://localhost:63100/api/speaker/speakers/doctor"
    ```
    
    **示例响应：**
    ```json
    {
        "success": true,
        "message": "说话人 'doctor' 已删除",
        "data": {
            "speaker_name": "doctor"
        }
    }
    ```
    """
    try:
        # 获取处理器
        processor = get_enrollment_processor()
        
        # 删除说话人
        result = processor.delete_speaker(speaker_name)
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": result["message"],
                    "data": {
                        "speaker_name": result["speaker_name"]
                    }
                }
            )
        else:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": result["message"],
                    "data": None
                }
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除失败: {str(e)}"
        )


@router.post("/verify", summary="验证说话人身份")
async def verify_speaker(
    audio_file: UploadFile = File(..., description="待验证的音频文件"),
    speaker_name: str = Form(..., description="要验证的说话人名称"),
    threshold: Optional[float] = Form(None, description="相似度阈值（0-1，默认0.5）")
):
    """
    验证音频是否为指定说话人
    
    **功能说明：**
    从上传的音频文件中提取声纹特征，与指定说话人的注册声纹进行比对，
    返回相似度分数和是否匹配的判断。
    
    **输入参数：**
    - `audio_file`: WAV音频文件（待验证的音频）
    - `speaker_name`: 要验证的说话人名称（必须是已注册的说话人）
    - `threshold`: 相似度阈值（可选，默认0.5，范围0-1）
      - 相似度 >= threshold：判定为匹配
      - 相似度 < threshold：判定为不匹配
    
    **输出格式（成功）：**
    ```json
    {
        "success": true,
        "data": {
            "speaker_name": "doctor",
            "is_match": true,
            "similarity": 0.85,
            "threshold": 0.5
        }
    }
    ```
    
    **输出格式（失败 - 说话人不存在）：**
    ```json
    {
        "success": false,
        "message": "说话人 'unknown' 不存在",
        "data": null
    }
    ```
    
    **输出格式（失败 - 其他错误）：**
    ```json
    {
        "detail": "验证失败: 错误信息"
    }
    ```
    
    **使用场景：**
    - 身份验证：验证音频是否为特定说话人
    - 说话人确认：在语音识别前确认说话人身份
    - 安全验证：验证访问权限
    
    **注意事项：**
    - 音频文件必须是WAV格式
    - speaker_name必须是已注册的说话人
    - threshold可以根据实际需求调整（值越高，匹配要求越严格）
    
    **示例请求：**
    ```bash
    curl -X POST "http://localhost:63100/api/speaker/verify" \
         -F "audio_file=@test_audio.wav" \
         -F "speaker_name=doctor" \
         -F "threshold=0.5"
    ```
    
    **示例响应：**
    ```json
    {
        "success": true,
        "data": {
            "speaker_name": "doctor",
            "is_match": true,
            "similarity": 0.85,
            "threshold": 0.5
        }
    }
    ```
    """
    try:
        # 获取处理器
        processor = get_enrollment_processor()
        
        # 检查文件格式
        if not validate_file_format(audio_file.filename, ('.wav', '.WAV')):
            raise HTTPException(
                status_code=400,
                detail="只支持WAV格式音频文件"
            )
        
        # 使用临时文件管理器保存上传的文件
        with TempFileManager(suffix=".wav") as tmp_manager:
            tmp_path = await tmp_manager.save_upload_file(audio_file)
        
            # 调用验证处理器
            result = processor.verify_speaker(
                audio_path=tmp_path,
                speaker_name=speaker_name,
                threshold=threshold
            )
            
            if result["success"]:
                return JSONResponse(
                    status_code=200,
                    content={
                        "success": True,
                        "data": {
                            "speaker_name": result["speaker_name"],
                            "is_match": result["is_match"],
                            "similarity": result["similarity"],
                            "threshold": result["threshold"]
                        }
                    }
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": result.get("error", "验证失败"),
                        "data": None
                    }
                )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"验证失败: {str(e)}"
        )

