"""
FastAPI application entrypoint for Local Clinical Scribe.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.streaming_asr import router as streaming_asr_router
from backend.api.offline_asr import router as offline_asr_router
from backend.api.speaker import router as speaker_router
from backend.api.clinical_note import router as clinical_note_router

app = FastAPI(
    title="Local Clinical Scribe",
    description=(
        "Local-first clinical conversation transcription, speaker separation, "
        "and structured note drafting."
    ),
    version="0.1.0",
)

# 允许跨域，便于前端联调
# 注意：CORSMiddleware 不会影响 WebSocket 连接，WebSocket 有自己的协议
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,  # WebSocket 可能需要 credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(streaming_asr_router)
app.include_router(offline_asr_router)
app.include_router(speaker_router)
app.include_router(clinical_note_router)


@app.get("/")
async def root():
    """
    根路径接口
    
    Returns:
        服务基本信息，包括文档地址和健康检查地址
    """
    return {
        "message": "Local Clinical Scribe API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    """
    健康检查接口
    
    Returns:
        服务健康状态
    """
    return {"status": "healthy", "service": "local-clinical-scribe"}

