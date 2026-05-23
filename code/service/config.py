"""
Runtime configuration for Local Clinical Scribe.
"""
import os

# 项目根目录（Docker 下通过环境变量 PROJECT_ROOT 传入，如 /app）
PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# 模型路径配置
MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained_models")
## 语音活动检测模型
VAD_MODEL_DIR = os.path.join(MODEL_DIR, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
## 说话人识别模型
SV_MODEL_DIR = os.path.join(MODEL_DIR, "speech_campplus_sv_zh-cn_16k-common")
## 语音识别模型-非流式
ASR_MODEL_DIR = os.path.join(MODEL_DIR, "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
## 语音识别模型-流式
ASR_ONLINE_MODEL_DIR = os.path.join(MODEL_DIR, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")
## 标点恢复模型
PUNC_MODEL_DIR = os.path.join(MODEL_DIR, "punc_ct-transformer_cn-en-common-vocab471067-large")

## 文本纠错模型（BERT）- ONNX模型路径
TEXT_CORRECTOR_MODEL_DIR = os.path.join(MODEL_DIR, "bert")
## 文本纠错模型（BERT）- Tokenizer路径
TEXT_CORRECTOR_TOKENIZER_DIR = os.path.join(MODEL_DIR, "bert/bert_corrector/1")

# 数据路径配置
# Runtime data is intentionally kept under data/ by default and ignored by git.
EMBEDDING_DIR = os.environ.get(
    "EMBEDDING_DIR", os.path.join(PROJECT_ROOT, "data", "speaker_embedding")
)
AUDIO_DIR = os.environ.get("AUDIO_DIR", os.path.join(PROJECT_ROOT, "data", "audio"))
ENCOUNTER_DIR = os.environ.get(
    "ENCOUNTER_DIR", os.path.join(PROJECT_ROOT, "data", "encounters")
)

# 设备配置（Docker 下可通过环境变量 CUDA_DEVICE 传入，如 cuda:0）
_PREFERRED_DEVICE = os.environ.get(
    "LOCAL_CLINICAL_SCRIBE_DEVICE",
    os.environ.get("CUDA_DEVICE", "cpu"),
)

def get_device():
    """
    智能设备选择函数
    - 如果配置了CUDA且CUDA可用，使用CUDA
    - 如果配置了CUDA但CUDA不可用，自动fallback到CPU
    - 如果配置了CPU，直接使用CPU
    """
    if _PREFERRED_DEVICE.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return _PREFERRED_DEVICE
            else:
                # 注意：config.py在logger初始化之前执行，所以这里保留print
                print(f"⚠️  CUDA不可用（配置: {_PREFERRED_DEVICE}），自动切换到CPU模式")
                return "cpu"
        except Exception as e:
            # 注意：config.py在logger初始化之前执行，所以这里保留print
            print(f"⚠️  检查CUDA时出错: {e}，使用CPU模式")
            return "cpu"
    return _PREFERRED_DEVICE

# 导出设备配置（自动检测CUDA可用性）
DEVICE = get_device()

# 模型加载配置
PRELOAD_MODELS = os.environ.get("PRELOAD_MODELS", "false").lower() in (
    "1",
    "true",
    "yes",
)  # 默认懒加载，便于启动文档和结构化草稿服务

# 服务配置（Docker 下前端通过 BACKEND_URL 环境变量指向后端服务名，如 http://backend:63100）
SERVER_NAME = "0.0.0.0"
BACKEND_PORT = int(os.environ.get("BACKEND_PORT", "63100"))
FRONTEND_PORT = int(os.environ.get("FRONTEND_PORT", "63101"))
BACKEND_URL = os.environ.get("BACKEND_URL") or f"http://localhost:{BACKEND_PORT}"

