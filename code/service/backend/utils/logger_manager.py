"""
通用日志管理器

功能：
- 统一的日志记录接口
- 支持不同日志级别（DEBUG, INFO, WARNING, ERROR）
- 同时输出到控制台和文件
- 每次启动时覆盖旧日志文件
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


class LoggerManager:
    """日志管理器"""
    
    _backend_logger: Optional[logging.Logger] = None
    _frontend_logger: Optional[logging.Logger] = None
    
    @staticmethod
    def _get_log_dir() -> Path:
        """获取日志目录路径"""
        # 项目根目录/logs
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    @staticmethod
    def init_backend_logger(log_file: str = "backend.log") -> logging.Logger:
        """
        初始化后端日志记录器
        
        Args:
            log_file: 日志文件名
            
        Returns:
            Logger实例
        """
        if LoggerManager._backend_logger is not None:
            return LoggerManager._backend_logger
        
        log_dir = LoggerManager._get_log_dir()
        log_path = log_dir / log_file
        
        # 创建logger
        logger = logging.getLogger("backend")
        logger.setLevel(logging.DEBUG)
        
        # 清除已有的handlers（避免重复添加）
        logger.handlers.clear()
        
        # 文件handler（覆盖模式）
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 写入启动信息
        logger.info("=" * 70)
        logger.info("🚀 后端服务启动")
        logger.info(f"日志文件: {log_path}")
        logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        LoggerManager._backend_logger = logger
        return logger
    
    @staticmethod
    def init_frontend_logger(log_file: str = "frontend.log") -> logging.Logger:
        """
        初始化前端日志记录器
        
        Args:
            log_file: 日志文件名
            
        Returns:
            Logger实例
        """
        if LoggerManager._frontend_logger is not None:
            return LoggerManager._frontend_logger
        
        log_dir = LoggerManager._get_log_dir()
        log_path = log_dir / log_file
        
        # 创建logger
        logger = logging.getLogger("frontend")
        logger.setLevel(logging.DEBUG)
        
        # 清除已有的handlers（避免重复添加）
        logger.handlers.clear()
        
        # 文件handler（覆盖模式）
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 写入启动信息
        logger.info("=" * 70)
        logger.info("🚀 前端服务启动")
        logger.info(f"日志文件: {log_path}")
        logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
        
        LoggerManager._frontend_logger = logger
        return logger
    
    @staticmethod
    def get_backend_logger() -> logging.Logger:
        """获取后端logger（如果未初始化则初始化）"""
        if LoggerManager._backend_logger is None:
            return LoggerManager.init_backend_logger()
        return LoggerManager._backend_logger
    
    @staticmethod
    def get_frontend_logger() -> logging.Logger:
        """获取前端logger（如果未初始化则初始化）"""
        if LoggerManager._frontend_logger is None:
            return LoggerManager.init_frontend_logger()
        return LoggerManager._frontend_logger
