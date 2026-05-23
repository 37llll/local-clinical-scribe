"""
文本纠错模块

功能：
- 提供统一的文本纠错接口（基类设计）
- 封装BERT纠错模型（基于ONNX推理）
- 内置ONNX推理客户端，无需外部依赖
- 支持批处理和单句纠错
- 与项目日志系统集成
- 懒加载机制，支持在ModelManager中统一管理
"""

import os
import sys
import time
import operator
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast

# 处理相对导入（支持作为脚本直接运行）
try:
    from ..utils.logger_manager import LoggerManager
    logger = LoggerManager.get_backend_logger()
except ImportError:
    # 作为脚本运行时，使用loguru
    from loguru import logger


# ==================== 工具函数 ====================

def get_errors(corrected_text: str, origin_text: str) -> Tuple[str, List]:
    """获取纠错的详细信息"""
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '"', '"', ''', ''', '琊', '\n', '…', '—', '擤']:
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            continue
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


# ==================== ONNX推理客户端（内置） ====================

class ONNXInferenceClient:
    """
    ONNX模型推理客户端（内置版本）
    
    支持直接使用ONNX Runtime进行推理
    用于BertForMaskedLM（MacBERT纠错模型）
    """

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """
        初始化ONNX推理客户端

        Args:
            model_path: ONNX模型文件路径或目录路径
            tokenizer_path: Tokenizer路径，如果为None则使用model_path
        """
        self.tokenizer_path = tokenizer_path or model_path

        logger.info(f"[ONNXClient] 加载BERT ONNX模型: {model_path}")

        # 如果是目录，优先查找model_int8.onnx，否则使用model.onnx
        model_path_obj = Path(model_path)
        if model_path_obj.is_dir():
            onnx_int8_file = model_path_obj / "model_int8.onnx"
            onnx_file = model_path_obj / "model.onnx"

            if onnx_int8_file.exists():
                self.model_path = str(onnx_int8_file)
                logger.info(f"[ONNXClient] 使用INT8量化模型: {self.model_path}")
            elif onnx_file.exists():
                self.model_path = str(onnx_file)
                logger.info(f"[ONNXClient] 使用标准模型: {self.model_path}")
            else:
                raise FileNotFoundError(
                    f"在 {model_path} 中未找到 model_int8.onnx 或 model.onnx"
                )
        else:
            self.model_path = model_path

        # 创建ONNX Runtime会话
        self.session = ort.InferenceSession(self.model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_path)

        logger.info("[ONNXClient] ONNX推理客户端初始化完成")

    def predict(self, texts: List[str], max_length: int = 128) -> List[np.ndarray]:
        """
        批量预测，返回logits

        Args:
            texts: 输入文本列表
            max_length: 最大输入序列长度

        Returns:
            logits列表，每个元素shape为 [sequence_length, vocab_size]
        """
        start_time = time.time()

        # Tokenize输入
        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        # 准备ONNX Runtime输入
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64)
        }

        # 执行推理
        outputs = self.session.run(None, ort_inputs)

        # 获取logits（第一个输出）
        logits = outputs[0]  # shape: [batch_size, sequence_length, vocab_size]

        inference_time = time.time() - start_time
        logger.debug(f"[ONNXClient] 推理耗时: {inference_time:.4f}s ({len(texts)}个文本)")

        # 返回每个样本的logits
        return [logits[i] for i in range(len(texts))]

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_type': 'BERT (ONNX)',
            'model_path': self.model_path,
            'tokenizer_path': self.tokenizer_path,
            'input_names': [inp.name for inp in self.session.get_inputs()],
            'output_names': [out.name for out in self.session.get_outputs()]
        }


class TextCorrectorBase(ABC):
    """文本纠错基类"""
    
    @abstractmethod
    def correct(self, text: str, **kwargs) -> str:
        """单句纠错"""
        pass
    
    @abstractmethod
    def correct_batch(self, texts: List[str], **kwargs) -> List[str]:
        """批量纠错"""
        pass
    
    @abstractmethod
    def correct_with_details(self, text: str, **kwargs) -> Dict:
        """单句纠错（带详细信息）"""
        pass


class BERTTextCorrector(TextCorrectorBase):
    """基于BERT的文本纠错器（MacBERT模型）"""
    
    def __init__(
        self, 
        model_path: str, 
        tokenizer_path: Optional[str] = None,
        max_length: int = 128,
        batch_size: int = 32
    ):
        logger.info(f"[TextCorrector] 初始化BERT文本纠错器...")
        logger.info(f"[TextCorrector] 模型路径: {model_path}")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self._client = None
        
        logger.info(f"[TextCorrector] BERT文本纠错器初始化完成")
    
    @property
    def client(self):
        """懒加载ONNX推理客户端"""
        if self._client is None:
            logger.info("[TextCorrector] 首次使用，加载ONNX推理客户端...")
            try:
                # 使用内置的ONNXInferenceClient
                self._client = ONNXInferenceClient(self.model_path, self.tokenizer_path)
                logger.info("[TextCorrector] ONNX推理客户端加载完成")
            except Exception as e:
                logger.error(f"[TextCorrector] 加载ONNX推理客户端失败: {e}")
                raise
        return self._client
    
    def correct(self, text: str, **kwargs) -> str:
        """单句纠错"""
        if not text or not text.strip():
            return text
        result = self.correct_with_details(text, **kwargs)
        return result['target']
    
    def correct_batch(self, texts: List[str], **kwargs) -> List[str]:
        """批量纠错"""
        if not texts:
            return []
        results = self.correct_batch_with_details(texts, **kwargs)
        return [result['target'] for result in results]
    
    def correct_with_details(self, text: str, **kwargs) -> Dict:
        """单句纠错（带详细信息）"""
        if not text or not text.strip():
            return {'source': text, 'target': text, 'errors': [], 'has_error': False}
        
        results = self.correct_batch_with_details([text], **kwargs)
        return results[0] if results else {
            'source': text, 'target': text, 'errors': [], 'has_error': False
        }
    
    def correct_batch_with_details(
        self, 
        texts: List[str], 
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> List[Dict]:
        """批量纠错（带详细信息）"""
        if not texts:
            return []
        
        max_length = max_length or self.max_length
        batch_size = batch_size or self.batch_size
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_results = self._predict(batch, max_length)
                
                for src, (corrected_text, details) in zip(batch, batch_results):
                    results.append({
                        'source': src,
                        'target': corrected_text,
                        'errors': details,
                        'has_error': len(details) > 0
                    })
        except Exception as e:
            logger.error(f"[TextCorrector] 批量纠错失败: {e}")
            for text in texts:
                results.append({
                    'source': text, 'target': text, 'errors': [], 'has_error': False
                })
        
        return results
    
    def _predict(self, texts: List[str], max_length: int = 128) -> List[Tuple[str, List]]:
        """预测句子纠错结果（内部方法）"""
        try:
            logits_list = self.client.predict(texts, max_length)
            results = []
            for logits, text in zip(logits_list, texts):
                _text = self.client.tokenizer.decode(
                    logits.argmax(axis=-1),
                    skip_special_tokens=True
                ).replace(' ', '')
                corrected_text = _text[:len(text)]
                corrected_text, details = get_errors(corrected_text, text)
                results.append((corrected_text, details))
            return results
        except Exception as e:
            logger.error(f"[TextCorrector] 预测失败: {e}")
            return [(text, []) for text in texts]


class TextCorrectorManager:
    """文本纠错管理器"""
    
    def __init__(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._corrector = None
    
    def get_corrector(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None) -> BERTTextCorrector:
        """获取文本纠错器（懒加载）"""
        if model_path:
            self.model_path = model_path
        if tokenizer_path is not None:
            self.tokenizer_path = tokenizer_path
        
        if self._corrector is None:
            if not self.model_path:
                raise ValueError("未指定BERT模型路径")
            
            logger.info("[TextCorrectorManager] 创建BERT文本纠错器...")
            self._corrector = BERTTextCorrector(self.model_path, tokenizer_path=self.tokenizer_path)
            logger.info("[TextCorrectorManager] BERT文本纠错器创建完成")
        
        return self._corrector
    
    def reload_corrector(self, model_path: Optional[str] = None, tokenizer_path: Optional[str] = None):
        """重新加载文本纠错器"""
        logger.info("[TextCorrectorManager] 重新加载文本纠错器...")
        if model_path:
            self.model_path = model_path
        if tokenizer_path is not None:
            self.tokenizer_path = tokenizer_path
        self._corrector = None
        self.get_corrector()
        logger.info("[TextCorrectorManager] 文本纠错器重新加载完成")



# ==================== 测试函数 ====================

def test_text_corrector():
    """
    测试文本纠错器
    
    使用方法：
    1. 确保BERT模型已经存在于指定路径
    2. 运行: python text_corrector.py
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试BERT文本纠错器')
    parser.add_argument(
        '--model_path',
        type=str,
        default=os.path.join("pretrained_models", "bert"),
        help='BERT模型目录或ONNX文件路径（默认：pretrained_models/bert，目录下需含 model_int8.onnx 或 model.onnx）'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=os.path.join("pretrained_models", "bert", "bert_corrector", "1"),
        help='Tokenizer 目录（默认：pretrained_models/bert/bert_corrector/1）'
    )
    parser.add_argument(
        '--test_mode',
        type=str,
        choices=['single', 'batch', 'details', 'all'],
        default='all',
        help='测试模式（single:单句, batch:批量, details:详细信息, all:全部测试）'
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🧪 开始测试BERT文本纠错器")
    logger.info("=" * 60)
    
    # 检查模型路径
    model_path = args.model_path
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型路径不存在: {model_path}")
        logger.info("\n提示：请确保BERT模型已经部署到指定路径")
        logger.info("或使用 --model_path 参数指定正确的路径")
        return
    
    logger.info(f"📁 模型路径: {model_path}")
    tokenizer_path = args.tokenizer_path
    logger.info(f"📁 Tokenizer 路径: {tokenizer_path}")
    
    # 测试用例
    test_cases = [
        "今天新情很好",
        "你找到你最喜欢的工作，我也很高心。",
        "医生给我开了布落芬",
        "这个方案很不措",
        "他的成积非常优秀",
        "我们要尊守交通规则"
    ]
    
    try:
        # 初始化纠错器
        logger.info("\n🔄 初始化文本纠错器...")
        corrector = BERTTextCorrector(model_path, tokenizer_path=tokenizer_path)
        logger.info("✅ 文本纠错器初始化完成\n")
        
        # 测试1: 单句纠错
        if args.test_mode in ['single', 'all']:
            logger.info("=" * 60)
            logger.info("📝 测试1: 单句纠错")
            logger.info("=" * 60)
            for text in test_cases[:3]:
                logger.info(f"\n原文: {text}")
                corrected = corrector.correct(text)
                logger.info(f"纠错: {corrected}")
                if corrected != text:
                    logger.info("✅ 检测到错误并纠正")
                else:
                    logger.info("ℹ️  未检测到错误")
        
        # 测试2: 批量纠错
        if args.test_mode in ['batch', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("📦 测试2: 批量纠错")
            logger.info("=" * 60)
            logger.info(f"\n批量处理 {len(test_cases)} 个句子...")
            corrected_texts = corrector.correct_batch(test_cases)
            
            for i, (src, tgt) in enumerate(zip(test_cases, corrected_texts), 1):
                logger.info(f"\n[{i}] 原文: {src}")
                logger.info(f"[{i}] 纠错: {tgt}")
                if src != tgt:
                    logger.info(f"[{i}] ✅ 已纠正")
        
        # 测试3: 详细信息
        if args.test_mode in ['details', 'all']:
            logger.info("\n" + "=" * 60)
            logger.info("🔍 测试3: 纠错详细信息")
            logger.info("=" * 60)
            for text in test_cases[:3]:
                logger.info(f"\n原文: {text}")
                result = corrector.correct_with_details(text)
                logger.info(f"纠错: {result['target']}")
                
                if result['has_error']:
                    logger.info(f"错误数量: {len(result['errors'])}")
                    for err in result['errors']:
                        orig_char, corr_char, start, end = err
                        logger.info(f"  位置 {start}: '{orig_char}' → '{corr_char}'")
                else:
                    logger.info("ℹ️  未检测到错误")
        
        # 测试4: 使用Manager
        if args.test_mode == 'all':
            logger.info("\n" + "=" * 60)
            logger.info("🔧 测试4: 使用TextCorrectorManager")
            logger.info("=" * 60)
            manager = TextCorrectorManager(model_path, tokenizer_path=tokenizer_path)
            corrector_from_manager = manager.get_corrector()
            
            test_text = "这个产品的质量很不措"
            logger.info(f"\n原文: {test_text}")
            corrected = corrector_from_manager.correct(test_text)
            logger.info(f"纠错: {corrected}")
        
        # 性能测试
        if args.test_mode == 'all':
            logger.info("\n" + "=" * 60)
            logger.info("⚡ 测试5: 性能测试")
            logger.info("=" * 60)
            
            # 测试单句性能
            start_time = time.time()
            for _ in range(10):
                corrector.correct(test_cases[0])
            single_time = (time.time() - start_time) / 10
            logger.info(f"\n单句平均耗时: {single_time*1000:.2f}ms")
            
            # 测试批量性能
            batch_texts = test_cases * 5  # 30个句子
            start_time = time.time()
            corrector.correct_batch(batch_texts)
            batch_time = time.time() - start_time
            logger.info(f"批量处理 {len(batch_texts)} 个句子耗时: {batch_time*1000:.2f}ms")
            logger.info(f"平均每句: {batch_time*1000/len(batch_texts):.2f}ms")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 所有测试完成！")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ 文件未找到: {e}")
        logger.info("\n请检查模型文件是否存在于指定路径")
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    test_text_corrector()
