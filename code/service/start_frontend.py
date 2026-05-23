#!/usr/bin/env python3
"""
启动前端服务脚本
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
logger = LoggerManager.init_frontend_logger()

# 直接导入 frontend 目录下的前端
from frontend.app import demo
from config import FRONTEND_PORT, BACKEND_URL

if __name__ == "__main__":
    logger.info(f"\n{'='*70}")
    logger.info("🚀 启动 Gradio 前端服务")
    logger.info(f"{'='*70}")
    logger.info(f"前端地址: http://0.0.0.0:{FRONTEND_PORT}")
    logger.info(f"后端 API: {BACKEND_URL}")
    logger.info(f"{'='*70}\n")
    
    inbrowser = os.environ.get("LAUNCH_INBROWSER", "true").lower() in ("1", "true", "yes")
    demo.launch(
        server_name="0.0.0.0",
        server_port=FRONTEND_PORT,
        share=False,
        inbrowser=inbrowser
    )

