import os
from modelscope.hub.snapshot_download import snapshot_download

def download_funasr_models(models_to_download, download_dir):
    """
    Downloads a list of FunASR models from ModelScope Hub to a specified directory.

    Args:
        models_to_download (dict): A dictionary where keys are model purposes and
                                   values are model IDs on ModelScope Hub.
        download_dir (str): The local directory to save the models to.
    """
    print(f"开始下载模型到 '{download_dir}' 目录...")

    # 确保目标目录存在
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"已创建目录: {download_dir}")

    for purpose, model_id in models_to_download.items():
        print("\n" + "="*50)
        print(f"正在下载: {purpose}")
        print(f"模型 ID: {model_id}")
        
        # 为每个模型创建一个独立的子目录，避免文件冲突
        model_path = os.path.join(download_dir, model_id.split('/')[1])
        
        try:
            # 执行下载
            # cache_dir 参数指定了总的缓存位置
            # local_dir 参数指定了本次下载的目标位置
            snapshot_download(
                model_id=model_id,
                cache_dir=download_dir, # 使用指定目录作为缓存根目录
                local_dir=model_path,   # 将模型文件直接下载到这个清晰的子目录
                # revision="master" # 可以指定版本，默认是 master
            )
            print(f"✅ '{purpose}' 模型下载成功！")
            print(f"   存放路径: {model_path}")
        except Exception as e:
            print(f"❌ 下载 '{purpose}' 模型时出错: {e}")
        print("="*50)

    print("\n所有模型下载任务已完成！")


if __name__ == "__main__":
    # --- v0.1 model set ---
    # Keep the default download aligned with the runtime config.
    MODELS = {
        "VAD (语音活动检测)": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "Streaming ASR (实时语音识别)": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "Offline ASR (离线高精度识别)": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "Speaker Verification (声纹识别)": "iic/speech_campplus_sv_zh-cn_16k-common",
        "Punc CT-Transformer": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    }
    
    # --- 下载目录配置 ---
    # 将模型保存在当前目录（脚本所在目录）中
    PRETRAINED_MODEL_DIR = "."
    
    # 执行下载函数
    download_funasr_models(MODELS, PRETRAINED_MODEL_DIR)
