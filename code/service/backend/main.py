"""
FastAPI application entrypoint for Local Clinical Scribe.
"""

import importlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.api.clinical_note import router as clinical_note_router

APP_VERSION = "0.4.0"
SERVICE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = SERVICE_DIR / "web"
optional_router_errors = []

app = FastAPI(
    title="Local Clinical Scribe",
    description=(
        "Local-first clinical conversation transcription, speaker separation, "
        "and structured note drafting."
    ),
    version=APP_VERSION,
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
app.include_router(clinical_note_router)

if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")


def include_optional_router(module_name: str, capability: str):
    """Register optional audio routers only when runtime deps are installed."""
    try:
        module = importlib.import_module(module_name)
        app.include_router(module.router)
    except ModuleNotFoundError as exc:
        optional_router_errors.append(
            {
                "capability": capability,
                "module": module_name,
                "error": str(exc),
            }
        )


include_optional_router("backend.api.offline_asr", "offline_asr")
include_optional_router("backend.api.streaming_asr", "streaming_asr")
include_optional_router("backend.api.speaker", "speaker_management")


@app.get("/")
async def root():
    """
    根路径接口
    
    Returns:
        服务基本信息，包括文档地址和健康检查地址
    """
    return {
        "message": "Local Clinical Scribe API",
        "version": APP_VERSION,
        "app": "/app",
        "docs": "/docs",
        "health": "/health",
        "capabilities": "/capabilities",
    }


@app.get("/app", include_in_schema=False)
async def web_app():
    """Serve the product workbench."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """
    健康检查接口
    
    Returns:
        服务健康状态
    """
    return {"status": "healthy", "service": "local-clinical-scribe"}


@app.get("/capabilities")
async def capabilities():
    """Show which optional runtime capabilities are currently available."""
    unavailable = optional_router_errors
    unavailable_keys = {item["capability"] for item in unavailable}
    all_optional = ["offline_asr", "streaming_asr", "speaker_management"]
    return {
        "core": {
            "clinical_note_drafting": True,
            "encounter_store": True,
            "markdown_json_export": True,
        },
        "optional_audio": {
            key: key not in unavailable_keys for key in all_optional
        },
        "unavailable": unavailable,
    }

