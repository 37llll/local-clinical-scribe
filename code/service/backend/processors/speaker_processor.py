"""
说话人处理器模块

功能：
- 说话人特征提取和相似度计算
- 说话人聚类分析
- 匹配已注册说话人
- 完整的说话人分离流程

包含4个处理器类：
1. SpeakerEmbeddingProcessor - 特征提取和相似度计算
2. SpeakerClusteringProcessor - 聚类分析
3. SpeakerMatchingProcessor - 匹配已注册说话人
4. SpeakerDiarizationProcessor - 完整的说话人分离流程
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
import warnings
import torch

warnings.filterwarnings('ignore')

from ..utils.logger_manager import LoggerManager
logger = LoggerManager.get_backend_logger()


# ==================== 1. SpeakerEmbeddingProcessor ====================

class SpeakerEmbeddingProcessor:
    """
    说话人特征提取处理器
    
    功能：
    - 从音频中提取说话人特征向量（embedding）
    - 计算两个特征向量之间的余弦相似度
    """
    
    def __init__(self, model_manager):
        """
        初始化说话人特征提取处理器
        
        Args:
            model_manager: ModelManager实例，用于获取SV模型
        """
        self.model_manager = model_manager
        self.sv_model = model_manager.get_sv_model()
        
        logger.info("[SpeakerEmbeddingProcessor] 初始化完成")
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        从音频中提取说话人特征向量（embedding）
        
        Args:
            audio: 音频数据，numpy数组，shape=(samples,)，16kHz采样率
        
        Returns:
            embedding: 说话人特征向量，numpy数组，shape=(embedding_dim,)
        
        Raises:
            ValueError: 如果音频长度不足或格式不正确
        """
        if len(audio) == 0:
            raise ValueError("音频数据为空")
        
        # 检查音频时长（建议至少1秒）
        duration = len(audio) / 16000
        if duration < 1.0:
            logger.warning(f"[警告] 音频时长过短 ({duration:.2f}秒)，可能影响特征质量")
        
        try:
            # 调用FunASR的SV模型提取特征
            # FunASR的SV模型会返回embedding向量
            result = self.sv_model.generate(
                input=audio,
                disable_pbar=True
            )
            
            # 解析结果
            if not result or len(result) == 0:
                raise ValueError("SV模型返回空结果")
            
            first_result = result[0] if isinstance(result, list) else result
            
            # 尝试从不同的字段提取embedding
            embedding = None
            if isinstance(first_result, dict):
                # 尝试常见的字段名
                for key in ['embedding', 'spk_embedding', 'feature']:
                    if key in first_result:
                        embedding = first_result[key]
                        break
            
            if embedding is None:
                raise ValueError(f"无法从SV模型结果中提取embedding: {first_result}")
            
            # 转换为numpy数组
            if not isinstance(embedding, np.ndarray):
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                else:
                    embedding = np.array(embedding)
            
            # 确保是一维数组
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            logger.info(f"[SpeakerEmbeddingProcessor] 提取embedding成功，维度: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"[错误] 提取embedding失败: {e}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个说话人特征向量的余弦相似度
        
        Args:
            embedding1: 第一个特征向量
            embedding2: 第二个特征向量
        
        Returns:
            similarity: 相似度分数，范围[0, 1]，越接近1表示越相似
        
        Note:
            余弦相似度计算公式: similarity = 1 - cosine_distance
            - 1.0 表示完全相同
            - 0.5 表示中等相似（默认阈值）
            - 0.0 表示完全不同
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(f"特征向量维度不匹配: {embedding1.shape} vs {embedding2.shape}")
        
        # 归一化（如果还未归一化）
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 计算余弦相似度
        cosine_distance = cosine(embedding1, embedding2)
        similarity = 1 - cosine_distance
        
        # 确保在[0, 1]范围内
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)


# ==================== 2. SpeakerClusteringProcessor ====================

class SpeakerClusteringProcessor:
    """
    说话人聚类处理器
    
    功能：
    - 对多个说话人特征向量进行聚类分析
    - 支持多种聚类算法（SpectralCluster、K-Means、DBSCAN、自动选择）
    """
    
    def __init__(self, model_manager):
        """
        初始化说话人聚类处理器
        
        Args:
            model_manager: ModelManager实例（用于兼容性，实际不需要）
        """
        self.model_manager = model_manager
        
        # 相似度阈值（可调整）
        self.similarity_threshold = 0.5
        
        logger.info("[SpeakerClusteringProcessor] 初始化完成")
    
    def cluster_speakers(
        self,
        embeddings: List[np.ndarray],
        method: str = 'spectral',
        min_clusters: int = 2,
        max_clusters: int = 4,
        preset_num: Optional[int] = None
    ) -> Tuple[List[int], int, Optional[float]]:
        """
        对多个说话人特征向量进行聚类分析
        
        Args:
            embeddings: 特征向量列表，每个元素是一个embedding数组
            method: 聚类方法
                - 'spectral': FunASR官方SpectralCluster谱聚类（默认，推荐）
                - 'auto': 自动选择最优聚类数（使用K-Means）
                - 'kmeans': K-Means聚类
                - 'dbscan': DBSCAN聚类（基于密度）
            min_clusters: 最小聚类数
            max_clusters: 最大聚类数
            preset_num: 预设的说话人数量（如果指定，则直接使用该数量）
        
        Returns:
            labels: 每个embedding对应的说话人标签，例如 [0, 1, 0, 2, 1, ...]
            n_speakers: 识别出的说话人数量
            score: 聚类质量分数（SpectralCluster返回None，其他算法返回轮廓系数）
        
        Raises:
            ValueError: 如果embeddings数量不足或格式不正确
        """
        if len(embeddings) < 2:
            # 只有一个或零个片段，无需聚类
            return [0] * len(embeddings), min(1, len(embeddings)), 1.0
        
        # 转换为numpy矩阵
        X = np.array(embeddings)
        
        if X.ndim != 2:
            raise ValueError(f"embeddings格式错误，期望2D数组，实际: {X.shape}")
        
        logger.info(f"[聚类] 输入{len(embeddings)}个片段，特征维度: {X.shape[1]}")
        
        # 如果指定了预设说话人数量，使用SpectralCluster（如果是spectral方法）或K-Means
        if preset_num is not None:
            logger.info(f"[聚类] 使用预设说话人数量: {preset_num}")
            if method == 'spectral':
                return self._spectral_cluster(X, min_clusters=preset_num, max_clusters=preset_num)
            else:
                return self._kmeans_cluster(X, n_clusters=preset_num)
        
        # 根据方法选择聚类算法
        if method == 'spectral':
            return self._spectral_cluster(X, min_clusters, max_clusters)
        elif method == 'dbscan':
            return self._dbscan_cluster(X)
        elif method == 'kmeans':
            # 使用中间值作为聚类数
            n_clusters = (min_clusters + max_clusters) // 2
            return self._kmeans_cluster(X, n_clusters=n_clusters)
        else:  # method == 'auto'
            return self._auto_cluster(X, min_clusters, max_clusters)
    
    def _spectral_cluster(
        self, 
        X: np.ndarray, 
        min_clusters: int, 
        max_clusters: int
    ) -> Tuple[List[int], int, Optional[float]]:
        """
        使用FunASR官方SpectralCluster进行谱聚类
        
        这是FunASR官方推荐的聚类算法，效果优于传统的K-Means和层次聚类。
        
        Args:
            X: 特征矩阵 shape=(n_samples, n_features)
            min_clusters: 最小说话人数
            max_clusters: 最大说话人数
        
        Returns:
            labels, n_speakers, None (SpectralCluster不返回score)
        """
        try:
            from funasr.models.campplus.cluster_backend import SpectralCluster
            
            # 转换为torch tensor
            embeddings_tensor = torch.from_numpy(X).float()
            
            # 创建SpectralCluster实例
            clusterer = SpectralCluster(
                min_num_spks=min_clusters,
                max_num_spks=max_clusters
            )
            
            # 执行聚类（SpectralCluster使用__call__方法，不是forward）
            logger.info(f"[SpectralCluster] 执行谱聚类，说话人范围: {min_clusters}-{max_clusters}")
            labels = clusterer(embeddings_tensor, oracle_num=None)
            
            # 转换回numpy
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            # 计算说话人数量
            n_speakers = len(np.unique(labels))
            
            logger.info(f"[SpectralCluster] 聚类完成：识别出{n_speakers}个说话人")
            
            # 计算轮廓系数并显示对比
            score = None
            try:
                if n_speakers > 1:
                    score = silhouette_score(X, labels)
                    logger.info(f"[SpectralCluster] 当前聚类({n_speakers}人)轮廓系数: {score:.3f}")
                
                # 显示不同聚类数的轮廓系数对比
                logger.info(f"[SpectralCluster] 轮廓系数对比（供参考）：")
                for n in range(min_clusters, max_clusters + 1):
                    try:
                        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                        test_labels = kmeans.fit_predict(X)
                        test_score = silhouette_score(X, test_labels)
                        marker = " ← SpectralCluster选择" if n == n_speakers else ""
                        logger.info(f"  {n}个说话人: 轮廓系数={test_score:.3f}{marker}")
                    except:
                        pass
            except:
                pass
            
            return labels.tolist(), n_speakers, score
            
        except ImportError as e:
            logger.warning(f"[警告] SpectralCluster不可用: {e}")
            logger.warning(f"[警告] 降级使用自动K-Means聚类")
            # 降级到自动K-Means（尝试多个聚类数，选择最优）
            return self._auto_cluster(X, min_clusters, max_clusters)
        except Exception as e:
            logger.error(f"[错误] SpectralCluster失败: {e}")
            logger.warning(f"[警告] 降级使用自动K-Means聚类")
            # 降级到自动K-Means（尝试多个聚类数，选择最优）
            return self._auto_cluster(X, min_clusters, max_clusters)
    
    def _kmeans_cluster(self, X: np.ndarray, n_clusters: int) -> Tuple[List[int], int, float]:
        """
        使用K-Means进行聚类
        
        Args:
            X: 特征矩阵 shape=(n_samples, n_features)
            n_clusters: 聚类数量
        
        Returns:
            labels, n_clusters, score
        """
        # 确保聚类数不超过样本数
        n_clusters = min(n_clusters, len(X))
        
        if n_clusters == 1:
            return [0] * len(X), 1, 1.0
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # 计算轮廓系数
        try:
            score = silhouette_score(X, labels)
        except:
            score = 0.0
        
        logger.info(f"[K-Means] 聚类数={n_clusters}, 轮廓系数={score:.3f}")
        return labels.tolist(), n_clusters, float(score)
    
    def _dbscan_cluster(self, X: np.ndarray) -> Tuple[List[int], int, float]:
        """
        使用DBSCAN进行基于密度的聚类
        
        Args:
            X: 特征矩阵
        
        Returns:
            labels, n_clusters, score
        """
        # DBSCAN参数（可调优）
        eps = 0.5  # 邻域半径
        min_samples = 2  # 最小样本数
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(X)
        
        # 计算说话人数量（排除噪声点-1）
        n_speakers = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 计算轮廓系数
        try:
            if n_speakers > 1:
                score = silhouette_score(X, labels)
            else:
                score = 0.0
        except:
            score = 0.0
        
        logger.info(f"[DBSCAN] 识别出{n_speakers}个说话人, 轮廓系数={score:.3f}")
        return labels.tolist(), n_speakers, float(score)
    
    def _auto_cluster(
        self,
        X: np.ndarray,
        min_clusters: int,
        max_clusters: int
    ) -> Tuple[List[int], int, float]:
        """
        自动选择最优聚类数
        
        尝试不同的聚类数（min_clusters到max_clusters），
        选择轮廓系数最高的那个。
        
        Args:
            X: 特征矩阵
            min_clusters: 最小聚类数
            max_clusters: 最大聚类数
        
        Returns:
            labels, n_clusters, score
        """
        # 确保范围合理
        max_clusters = min(max_clusters, len(X))
        min_clusters = max(2, min_clusters)
        
        if max_clusters < min_clusters:
            max_clusters = min_clusters
        
        logger.info(f"[自动聚类] 尝试聚类数范围: {min_clusters}-{max_clusters}")
        
        best_score = -1
        best_labels = None
        best_n_clusters = min_clusters
        
        # 尝试不同的聚类数
        for n in range(min_clusters, max_clusters + 1):
            try:
                labels, _, score = self._kmeans_cluster(X, n_clusters=n)
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_n_clusters = n
            except Exception as e:
                logger.warning(f"[警告] 聚类数{n}失败: {e}")
                continue
        
        if best_labels is None:
            # 如果所有尝试都失败，返回单一说话人
            logger.warning("[警告] 所有聚类尝试失败，默认返回单一说话人")
            return [0] * len(X), 1, 0.0
        
        logger.info(f"[自动聚类] 最优聚类数={best_n_clusters}, 轮廓系数={best_score:.3f}")
        return best_labels, best_n_clusters, best_score
    
    def set_similarity_threshold(self, threshold: float):
        """
        设置相似度阈值
        
        Args:
            threshold: 新的阈值，范围[0, 1]
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"阈值必须在[0, 1]范围内，当前值: {threshold}")
        
        self.similarity_threshold = threshold
        logger.info(f"[SpeakerClusteringProcessor] 相似度阈值已更新为: {threshold}")


# ==================== 3. SpeakerMatchingProcessor ====================

class SpeakerMatchingProcessor:
    """
    说话人匹配处理器
    
    功能：
    - 将embedding与已注册说话人进行匹配
    - 将聚类结果与已注册说话人对齐
    """
    
    def __init__(self, model_manager):
        """
        初始化说话人匹配处理器
        
        Args:
            model_manager: ModelManager实例
        """
        self.model_manager = model_manager
        self.embedding_processor = SpeakerEmbeddingProcessor(model_manager)
        
        # 相似度阈值（可调整）
        self.similarity_threshold = 0.5
        
        logger.info("[SpeakerMatchingProcessor] 初始化完成")
    
    def match_with_registered_speakers(
        self,
        embeddings: List[np.ndarray],
        registered_embeddings: Dict[str, np.ndarray],
        threshold: Optional[float] = None
    ) -> List[Tuple[int, Optional[str], float]]:
        """
        将聚类结果与已注册的说话人进行匹配
        
        Args:
            embeddings: 待匹配的特征向量列表
            registered_embeddings: 已注册说话人的特征向量字典 {speaker_name: embedding}
            threshold: 相似度阈值，超过该阈值才认为匹配成功（默认使用self.similarity_threshold）
        
        Returns:
            匹配结果列表，每个元素为 (segment_idx, speaker_name, similarity)
            - segment_idx: 片段索引
            - speaker_name: 匹配到的说话人名称（如果未匹配则为None）
            - similarity: 最高相似度分数
        
        示例：
            [(0, 'doctor', 0.85), (1, None, 0.42), (2, 'doctor', 0.78)]
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        results = []
        
        for idx, emb in enumerate(embeddings):
            best_match = None
            best_similarity = 0.0
            
            # 与每个已注册说话人比较
            for speaker_name, registered_emb in registered_embeddings.items():
                similarity = self.embedding_processor.compute_similarity(emb, registered_emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= threshold:
                        best_match = speaker_name
            
            results.append((idx, best_match, best_similarity))
            
            if best_match:
                logger.info(f"[匹配] 片段{idx} -> {best_match} (相似度: {best_similarity:.3f})")
            else:
                logger.info(f"[匹配] 片段{idx} -> 未知说话人 (最高相似度: {best_similarity:.3f})")
        
        return results
    
    def align_clusters_with_registered(
        self,
        cluster_labels: List[int],
        embeddings: List[np.ndarray],
        registered_embeddings: Dict[str, np.ndarray],
        threshold: Optional[float] = None
    ) -> Dict[int, Optional[str]]:
        """
        将聚类结果与已注册说话人对齐
        
        对每个聚类（cluster），计算其中心点与已注册说话人的相似度，
        如果超过阈值，则将该聚类标记为该说话人。
        
        Args:
            cluster_labels: 聚类标签列表 [0, 1, 0, 2, 1, ...]
            embeddings: 对应的特征向量列表
            registered_embeddings: 已注册说话人的特征向量字典
            threshold: 相似度阈值
        
        Returns:
            聚类到说话人的映射字典 {cluster_id: speaker_name}
            例如: {0: 'doctor', 1: 'patient', 2: None}
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        # 计算每个聚类的中心点
        cluster_centers = {}
        for label in set(cluster_labels):
            if label < 0:  # 跳过噪声点
                continue
            
            # 找到该聚类的所有样本
            indices = [i for i, l in enumerate(cluster_labels) if l == label]
            cluster_embs = [embeddings[i] for i in indices]
            
            # 计算中心点（平均）
            center = np.mean(cluster_embs, axis=0)
            cluster_centers[label] = center
        
        # 将每个聚类中心与已注册说话人匹配
        cluster_to_speaker = {}
        
        for cluster_id, center in cluster_centers.items():
            best_match = None
            best_similarity = 0.0
            
            for speaker_name, registered_emb in registered_embeddings.items():
                similarity = self.embedding_processor.compute_similarity(center, registered_emb)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= threshold:
                        best_match = speaker_name
            
            cluster_to_speaker[cluster_id] = best_match
            
            if best_match:
                logger.info(f"[对齐] 聚类{cluster_id} -> {best_match} (相似度: {best_similarity:.3f})")
            else:
                logger.info(f"[对齐] 聚类{cluster_id} -> 未知说话人 (最高相似度: {best_similarity:.3f})")
        
        return cluster_to_speaker


# ==================== 4. SpeakerDiarizationProcessor ====================

class SpeakerDiarizationProcessor:
    """
    说话人分离处理器
    
    功能：
    - 提取speaker embedding
    - 执行说话人聚类
    - 匹配已注册说话人
    - 为片段添加说话人标签
    """
    
    def __init__(self, model_manager):
        """
        初始化说话人分离处理器
        
        Args:
            model_manager: ModelManager实例
        """
        self.model_manager = model_manager
        
        # 初始化子处理器
        self.embedding_processor = SpeakerEmbeddingProcessor(model_manager)
        self.clustering_processor = SpeakerClusteringProcessor(model_manager)
        self.matching_processor = SpeakerMatchingProcessor(model_manager)
        
        # 加载说话人注册服务（用于加载已注册说话人）
        from ..services.speaker_service import SpeakerEnrollmentService
        self.enrollment_service = SpeakerEnrollmentService(model_manager)
        
        logger.info("[SpeakerDiarizationProcessor] 初始化完成")
    
    def process_segments(
        self,
        audio: np.ndarray,
        segments: List[Dict[str, Any]],
        mode: str = "cluster",
        registered_speakers: Optional[List[str]] = None,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        为片段添加说话人信息
        
        Args:
            audio: 完整音频数据
            segments: 片段列表，每个片段包含 start_sample, end_sample, text
            mode: 识别模式
                - "cluster": 仅聚类识别
                - "cluster_match": 聚类后与注册说话人匹配
                - "direct_match": 直接匹配每个片段
            registered_speakers: 已注册说话人名称列表
            similarity_threshold: 相似度阈值
        
        Returns:
            添加了speaker和similarity字段的片段列表
        """
        logger.info(f"[Speaker] 开始说话人识别，模式={mode}，共{len(segments)}个片段")
        
        try:
            # Step 1: 提取embeddings
            embeddings, valid_segments = self._extract_embeddings(audio, segments)
            if not embeddings:
                logger.warning(f"[Speaker] 警告：未能提取有效embedding")
                return segments
            
            logger.info(f"[Speaker] 成功提取{len(embeddings)}个embedding")
            
            # Step 2: 加载已注册说话人
            registered_embeddings = self._load_registered_speakers(
                registered_speakers, mode
            )
            
            # Step 3: 根据模式执行识别
            if mode == "cluster":
                self._cluster_mode(valid_segments, embeddings)
            elif mode == "cluster_match":
                self._cluster_match_mode(
                    valid_segments, embeddings, registered_embeddings, similarity_threshold
                )
            elif mode == "direct_match":
                self._direct_match_mode(
                    valid_segments, embeddings, registered_embeddings, similarity_threshold
                )
            else:
                raise ValueError(f"无效的说话人识别模式: {mode}")
            
            # Step 4: 合并结果
            result = self._merge_results(segments, valid_segments)
            
            logger.info(f"[Speaker] 说话人识别完成")
            return result
            
        except Exception as e:
            logger.error(f"[Speaker] 说话人识别失败: {e}")
            import traceback
            traceback.print_exc()
            return segments
    
    def _extract_embeddings(
        self, 
        audio: np.ndarray, 
        segments: List[Dict]
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """提取每个片段的speaker embedding"""
        embeddings = []
        valid_segments = []
        
        for seg in segments:
            start_sample = seg["start_sample"]
            end_sample = seg["end_sample"]
            seg_audio = audio[start_sample:end_sample]
            
            # 至少0.1秒才能提取embedding
            if len(seg_audio) > 1600:
                try:
                    emb = self.embedding_processor.extract_embedding(seg_audio)
                    embeddings.append(emb)
                    valid_segments.append(seg)
                except Exception as e:
                    logger.warning(f"[Speaker] 警告：片段embedding提取失败: {e}")
        
        return embeddings, valid_segments
    
    def _load_registered_speakers(
        self, 
        speaker_names: Optional[List[str]], 
        mode: str
    ) -> Dict[str, np.ndarray]:
        """加载已注册说话人的embeddings"""
        if not speaker_names or mode not in ["cluster_match", "direct_match"]:
            return {}
        
        registered_embeddings = {}
        for name in speaker_names:
            emb = self.enrollment_service.load_speaker(name)
            if emb is not None:
                registered_embeddings[name] = emb
        
        logger.info(f"[Speaker] 加载了{len(registered_embeddings)}个已注册说话人")
        return registered_embeddings
    
    def _cluster_mode(self, segments: List[Dict], embeddings: List[np.ndarray]):
        """
        聚类模式：仅识别说话人数量
        
        可选聚类方式（修改method参数）：
        - method='auto': K-Means + 轮廓系数自动选择（推荐，更准确）
        - method='spectral': 谱聚类（FunASR官方，需要高质量embedding）
        - method='kmeans': 固定K-Means
        """
        # ⭐ 使用auto模式：K-Means + 轮廓系数自动选择（更准确）
        labels, n_speakers, score = self.clustering_processor.cluster_speakers(embeddings, method='auto')
        
        if score is not None:
            logger.info(f"[Speaker] 聚类完成：{n_speakers}个说话人，质量分数{score:.3f}")
        else:
            logger.info(f"[Speaker] 聚类完成：{n_speakers}个说话人")
        
        for i, seg in enumerate(segments):
            seg["speaker"] = f"说话人{labels[i]}"
            seg["similarity"] = None
    
    def _cluster_match_mode(
        self,
        segments: List[Dict],
        embeddings: List[np.ndarray],
        registered_embeddings: Dict[str, np.ndarray],
        threshold: float
    ):
        """聚类匹配模式：聚类后与已注册说话人对齐"""
        # 聚类
        labels, n_speakers, score = self.clustering_processor.cluster_speakers(embeddings)
        logger.info(f"[Speaker] 聚类完成：{n_speakers}个说话人")
        
        # 对齐聚类与已注册说话人
        cluster_to_speaker = self.matching_processor.align_clusters_with_registered(
            labels, embeddings, registered_embeddings, threshold
        )
        
        # 分配标签
        for i, seg in enumerate(segments):
            cluster_id = labels[i]
            matched_speaker = cluster_to_speaker.get(cluster_id)
            
            if matched_speaker:
                seg["speaker"] = matched_speaker
                sim = self.embedding_processor.compute_similarity(
                    embeddings[i], registered_embeddings[matched_speaker]
                )
                seg["similarity"] = float(sim)
            else:
                seg["speaker"] = f"未知说话人{cluster_id}"
                seg["similarity"] = None
    
    def _direct_match_mode(
        self,
        segments: List[Dict],
        embeddings: List[np.ndarray],
        registered_embeddings: Dict[str, np.ndarray],
        threshold: float
    ):
        """直接匹配模式：每个片段独立匹配最相似的说话人"""
        match_results = self.matching_processor.match_with_registered_speakers(
            embeddings, registered_embeddings, threshold
        )
        
        for seg, (_, speaker_name, similarity) in zip(segments, match_results):
            seg["speaker"] = speaker_name if speaker_name else "未知说话人"
            seg["similarity"] = float(similarity)
    
    def _merge_results(
        self, 
        all_segments: List[Dict], 
        valid_segments: List[Dict]
    ) -> List[Dict]:
        """
        合并有效片段和原始片段列表
        
        使用start_sample作为唯一标识进行匹配，避免对象相等性判断问题
        """
        # 为valid_segments建立索引（使用start_sample作为唯一标识）
        valid_map = {seg["start_sample"]: seg for seg in valid_segments}
        
        result = []
        for seg in all_segments:
            if seg["start_sample"] in valid_map:
                # 使用已处理的segment（包含speaker信息）
                result.append(valid_map[seg["start_sample"]])
            else:
                # segment未能提取embedding，标记为"未识别"
                seg["speaker"] = "未识别"
                seg["similarity"] = None
                result.append(seg)
        
        return result

