"""
模型管理器（单例模式）

功能：
- 统一管理所有模型实例（VAD、ASR、说话人识别、标点恢复等）
- 提供模型的懒加载和单例访问
- 支持批量加载和重新加载模型
"""
import warnings
from funasr import AutoModel as FunASRAutoModel

warnings.filterwarnings('ignore')

from config import (
    VAD_MODEL_DIR, SV_MODEL_DIR, ASR_MODEL_DIR, ASR_ONLINE_MODEL_DIR, PUNC_MODEL_DIR,
    TEXT_CORRECTOR_MODEL_DIR, TEXT_CORRECTOR_TOKENIZER_DIR,  # 【新增】
    DEVICE
)
from ..utils.logger_manager import LoggerManager

logger = LoggerManager.get_backend_logger()


class ModelManager:
    """
    模型管理器（单例模式）
    
    功能：
    - 统一管理所有模型实例
    - 提供模型的懒加载机制
    - 确保每个模型只加载一次
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            # 初始化所有模型变量为 None
            self._vad_model = None
            self._sv_model = None
            self._asr_offline_model = None
            self._asr_online_model = None
            self._punc_model = None
            self._recognition_model = None
            self._enrollment_model = None
            self._text_corrector = None  # 【新增】
    
    # ========== 基础模型获取方法 ==========
    
    def get_vad_model(self) -> FunASRAutoModel:
        """
        获取VAD模型（统一管理）
        
        Returns:
            VAD模型实例（如果未加载则自动加载）
        """
        if not hasattr(self, '_vad_model') or self._vad_model is None:
            logger.info("🔄 加载VAD模型...")
            self._vad_model = FunASRAutoModel(model=VAD_MODEL_DIR, device=DEVICE, disable_update=True, disable_pbar=True)
            logger.info("✅ VAD模型加载完成")
        return self._vad_model
    
    def get_sv_model(self) -> FunASRAutoModel:
        """
        获取说话人识别模型（统一管理）
        
        Returns:
            说话人识别模型实例（如果未加载则自动加载）
        """
        if not hasattr(self, '_sv_model') or self._sv_model is None:
            logger.info("🔄 加载说话人识别模型...")
            self._sv_model = FunASRAutoModel(model=SV_MODEL_DIR, device=DEVICE, disable_update=True, disable_pbar=True)
            logger.info("✅ 说话人识别模型加载完成")
        return self._sv_model
    
    def get_asr_offline_model(self) -> FunASRAutoModel:
        """
        获取非流式ASR模型（统一管理）
        
        Returns:
            非流式ASR模型实例（如果未加载则自动加载）
        """
        if not hasattr(self, '_asr_offline_model') or self._asr_offline_model is None:
            logger.info("🔄 加载非流式ASR模型...")
            self._asr_offline_model = FunASRAutoModel(model=ASR_MODEL_DIR, device=DEVICE, disable_update=True, disable_pbar=True)
            logger.info("✅ 非流式ASR模型加载完成")
        return self._asr_offline_model
    
    def get_asr_online_model(self) -> FunASRAutoModel:
        """
        获取流式ASR模型（统一管理）
        
        Returns:
            流式ASR模型实例（如果未加载则自动加载）
        """
        if not hasattr(self, '_asr_online_model') or self._asr_online_model is None:
            logger.info("🔄 加载流式ASR模型...")
            self._asr_online_model = FunASRAutoModel(model=ASR_ONLINE_MODEL_DIR, device=DEVICE, disable_update=True, disable_pbar=True)
            logger.info("✅ 流式ASR模型加载完成")
        return self._asr_online_model
    
    def get_punc_model(self) -> FunASRAutoModel:
        """
        获取标点恢复模型（统一管理）
        
        Returns:
            标点恢复模型实例（如果未加载则自动加载）
        """
        if not hasattr(self, '_punc_model') or self._punc_model is None:
            logger.info("🔄 加载标点恢复模型...")
            self._punc_model = FunASRAutoModel(model=PUNC_MODEL_DIR, device=DEVICE, disable_update=True, disable_pbar=True)
            logger.info("✅ 标点恢复模型加载完成")
        return self._punc_model

    def get_text_corrector(self):
        """
        获取文本纠错器（统一管理）
        
        Returns:
            BERTTextCorrector实例（如果未加载则自动加载）
        """
        if not hasattr(self, '_text_corrector') or self._text_corrector is None:
            logger.info("🔄 加载文本纠错器...")
            from .text_corrector import BERTTextCorrector
            self._text_corrector = BERTTextCorrector(
                TEXT_CORRECTOR_MODEL_DIR, 
                tokenizer_path=TEXT_CORRECTOR_TOKENIZER_DIR
            )
            logger.info("✅ 文本纠错器加载完成")
        return self._text_corrector
    
    def get_asr_offline_with_vad_punc(self) -> FunASRAutoModel:
        """
        获取组合的非流式ASR模型（包含VAD+ASR+PUNC）
        
        用于处理完整音频文件，自动进行：
        1. VAD分段
        2. ASR识别
        3. 标点补全
        
        适合处理任意长度的音频。
        """
        if not hasattr(self, '_asr_offline_combined') or self._asr_offline_combined is None:
            logger.info("🔄 加载组合的非流式ASR模型（VAD+ASR+PUNC）...")
            self._asr_offline_combined = FunASRAutoModel(
                model=ASR_MODEL_DIR,
                vad_model=VAD_MODEL_DIR,
                punc_model=PUNC_MODEL_DIR,
                vad_kwargs={"max_single_segment_time": 60000},  # VAD最大切割60秒
                device=DEVICE,
                disable_update=True,
                disable_pbar=True
            )
            logger.info("✅ 组合的非流式ASR模型加载完成")
        return self._asr_offline_combined
    
    # ========== 高级处理器获取方法 ==========
    
    def get_recognition_model(self):
        """
        获取说话人识别处理器（统一管理，单例模式）
        
        注意：重构后，此方法返回 SpeakerClusteringProcessor
        如果需要其他处理器，请直接导入：
        - SpeakerEmbeddingProcessor: 特征提取
        - SpeakerClusteringProcessor: 聚类分析
        - SpeakerMatchingProcessor: 匹配已注册说话人
        - SpeakerDiarizationProcessor: 完整流程
        """
        if not hasattr(self, '_recognition_model') or self._recognition_model is None:
            from ..processors.speaker_processor import SpeakerClusteringProcessor
            logger.info("🔄 初始化说话人聚类处理器...")
            self._recognition_model = SpeakerClusteringProcessor(self)
            logger.info("✅ 说话人聚类处理器初始化完成")
        return self._recognition_model
    
    def get_enrollment_model(self):
        """获取说话人注册服务（统一管理，单例模式）"""
        if not hasattr(self, '_enrollment_model') or self._enrollment_model is None:
            from ..services.speaker_service import SpeakerEnrollmentService
            logger.info("🔄 初始化说话人注册服务...")
            self._enrollment_model = SpeakerEnrollmentService(self)
            logger.info("✅ 说话人注册服务初始化完成")
        return self._enrollment_model
    
    # ========== 组合模型加载函数 ==========
    
    def load_all_models(self):
        """
        加载所有模型
        
        包括：
        - VAD模型
        - 说话人识别模型（SV）
        - 非流式ASR模型
        - 流式ASR模型
        - 标点恢复模型（PUNC）
        - 组合模型（VAD+ASR+PUNC）
        - 说话人识别处理器
        - 说话人注册处理器
        """
        logger.info("\n" + "="*60)
        logger.info("📦 加载所有模型...")
        logger.info("="*60)
        
        try:
            logger.info("\n[1/9] 加载VAD模型...")
            self.get_vad_model()
            
            logger.info("\n[2/9] 加载说话人识别模型（SV）...")
            self.get_sv_model()
            
            logger.info("\n[3/9] 加载非流式ASR模型...")
            self.get_asr_offline_model()
            
            logger.info("\n[4/9] 加载流式ASR模型...")
            self.get_asr_online_model()
            
            logger.info("\n[5/9] 加载标点恢复模型（PUNC）...")
            self.get_punc_model()
            
            logger.info("\n[6/9] 加载组合模型（VAD+ASR+PUNC）...")
            self.get_asr_offline_with_vad_punc()
            
            logger.info("\n[7/9] 初始化说话人识别处理器...")
            self.get_recognition_model()
            
            logger.info("\n[8/9] 初始化说话人注册处理器...")
            self.get_enrollment_model()

            logger.info("\n[9/9] 加载文本纠错器（可选）...")
            self.get_text_corrector()
            
            logger.info("\n✅ 所有模型加载完成")
            logger.info("="*60 + "\n")
        except Exception as e:
            logger.error(f"\n❌ 模型加载失败: {e}")
            raise
    
    def reload_models(self):
        """
        重新加载已缓存的模型（用于模型更新）
        
        先重置 VAD、SV、ASR 离线/在线、标点、识别处理器、注册服务等实例，再调用 load_all_models 重新加载。
        注意：组合模型 _asr_offline_combined 未在此处重置，仍由 get_asr_offline_with_vad_punc 内部懒加载逻辑复用或创建。
        """
        logger.info("\n" + "="*60)
        logger.info("🔄 重新加载所有模型...")
        logger.info("="*60)
        
        # 重置已缓存的模型实例（不含 _asr_offline_combined）
        self._vad_model = None
        self._sv_model = None
        self._asr_offline_model = None
        self._asr_online_model = None
        self._punc_model = None
        self._recognition_model = None
        self._enrollment_model = None
        self._text_corrector = None
        
        # 立即重新加载所有模型
        try:
            self.load_all_models()
            logger.info("✅ 所有模型重新加载完成")
        except Exception as e:
            logger.error(f"❌ 模型重新加载失败: {e}")
            raise

