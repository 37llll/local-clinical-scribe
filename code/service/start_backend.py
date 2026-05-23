#!/usr/bin/env python3
"""
启动后端服务脚本
"""
import os
import sys

# 添加项目路径到 sys.path
# 获取当前脚本所在目录（service目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.dirname(current_dir)  # code 目录
project_root = os.path.dirname(code_dir)  # 项目根目录
sys.path.insert(0, current_dir)  # 添加 service 目录到路径

# 在导入其他模块之前初始化logger
from backend.utils.logger_manager import LoggerManager
logger = LoggerManager.init_backend_logger()

from backend.main import app
import uvicorn
from config import SERVER_NAME, BACKEND_PORT

if __name__ == "__main__":
    logger.info(f"\n{'='*70}")
    logger.info("🚀 启动 FastAPI 后端服务")
    logger.info(f"{'='*70}")
    logger.info(f"API 文档: http://{SERVER_NAME}:{BACKEND_PORT}/docs")
    logger.info(f"健康检查: http://{SERVER_NAME}:{BACKEND_PORT}/health")
    logger.info(f"{'='*70}\n")
    
    uvicorn.run(
        app,
        host=SERVER_NAME,
        port=BACKEND_PORT,
        log_level="info"
    )

