"""
说话人分离工具函数

从FunASR官方speaker_utils.py提取的核心函数，避免依赖问题。

原始来源：
https://github.com/alibaba-damo-academy/FunASR
Copyright (c) Alibaba, Inc. and its affiliates.

提取的函数：
- sv_chunk: VAD片段切分成固定长度chunk
- postprocess: 说话人分离后处理（合并、平滑、处理重叠）
- correct_labels: 标签连续化
- merge_seque: 合并相同说话人的连续片段
- smooth: 平滑处理，移除短片段
"""

import numpy as np
from typing import List


def sv_chunk(vad_segments: list, fs: int = 16000) -> list:
    """
    将VAD分段切分成固定长度的chunk（用于说话人embedding提取）
    
    Args:
        vad_segments: VAD片段列表，格式 [[start_sec, end_sec, audio_data], ...]
        fs: 采样率，默认16000Hz
    
    Returns:
        chunk列表，格式 [[chunk_start, chunk_end, chunk_audio], ...]
    
    配置：
        - seg_dur: chunk时长 1.5秒
        - seg_shift: chunk滑动步长 0.75秒（50%重叠）
    """
    config = {
        "seg_dur": 1.5,
        "seg_shift": 0.75,
    }

    def seg_chunk(seg_data):
        """对单个segment进行chunk切分"""
        seg_st = seg_data[0]  # segment起始时间（秒）
        data = seg_data[2]    # segment音频数据
        chunk_len = int(config["seg_dur"] * fs)
        chunk_shift = int(config["seg_shift"] * fs)
        last_chunk_ed = 0
        seg_res = []
        
        for chunk_st in range(0, data.shape[0], chunk_shift):
            chunk_ed = min(chunk_st + chunk_len, data.shape[0])
            if chunk_ed <= last_chunk_ed:
                break
            last_chunk_ed = chunk_ed
            chunk_st = max(0, chunk_ed - chunk_len)
            chunk_data = data[chunk_st:chunk_ed]
            
            # 如果chunk长度不足，进行padding
            if chunk_data.shape[0] < chunk_len:
                chunk_data = np.pad(
                    chunk_data, 
                    (0, chunk_len - chunk_data.shape[0]), 
                    "constant"
                )
            
            seg_res.append([
                chunk_st / fs + seg_st,  # chunk起始时间
                chunk_ed / fs + seg_st,  # chunk结束时间
                chunk_data               # chunk音频数据
            ])
        
        return seg_res

    # 对所有segment进行chunk切分
    segs = []
    for i, s in enumerate(vad_segments):
        segs.extend(seg_chunk(s))

    return segs


def postprocess(
    segments: list, 
    vad_segments: list, 
    labels: np.ndarray, 
    embeddings: np.ndarray
) -> list:
    """
    说话人分离后处理（FunASR官方实现）
    
    处理步骤：
    1. 标签连续化：将聚类标签转换为连续的ID
    2. 合并相同说话人的连续片段
    3. 计算每个说话人的中心embedding
    4. 处理重叠区域：找中点平均分配
    5. 平滑处理：移除<1秒的短片段，分配给相邻说话人
    
    Args:
        segments: 时间片段列表，格式 [[start, end], ...]
        vad_segments: VAD片段（未使用，保持接口兼容）
        labels: 聚类标签数组，shape=(n_segments,)
        embeddings: embedding矩阵，shape=(n_segments, embedding_dim)
    
    Returns:
        处理后的结果列表，格式 [[start, end, speaker_id], ...]
    """
    assert len(segments) == len(labels), "segments和labels长度必须一致"
    
    # Step 1: 标签连续化
    labels = correct_labels(labels)
    
    # Step 2: 构建初始结果
    distribute_res = []
    for i in range(len(segments)):
        distribute_res.append([segments[i][0], segments[i][1], labels[i]])
    
    # Step 3: 合并相同说话人的连续片段
    distribute_res = merge_seque(distribute_res)

    # Step 4: 计算每个说话人的中心embedding（用于后续可能的操作）
    spk_embs = []
    for i in range(labels.max() + 1):
        spk_emb = embeddings[labels == i].mean(0)
        spk_embs.append(spk_emb)
    spk_embs = np.stack(spk_embs)

    def is_overlapped(t1, t2):
        """判断两个时间点是否重叠（t1是前一个片段的结束时间，t2是当前片段的开始时间）"""
        if t1 > t2 + 1e-4:
            return True
        return False

    # Step 5: 处理重叠区域
    # 如果前一个片段的结束时间 > 当前片段的开始时间，则在中点切分
    for i in range(1, len(distribute_res)):
        if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
            p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
            distribute_res[i][0] = p
            distribute_res[i - 1][1] = p

    # Step 6: 平滑处理（移除短片段）
    distribute_res = smooth(distribute_res)

    return distribute_res


def correct_labels(labels: np.ndarray) -> np.ndarray:
    """
    标签连续化
    
    将聚类标签转换为从0开始的连续整数。
    例如：[5, 5, 2, 2, 0, 0] -> [0, 0, 1, 1, 2, 2]
    
    Args:
        labels: 原始聚类标签数组
    
    Returns:
        连续化后的标签数组
    """
    labels_id = 0
    id2id = {}
    new_labels = []
    
    for i in labels:
        if i not in id2id:
            id2id[i] = labels_id
            labels_id += 1
        new_labels.append(id2id[i])
    
    return np.array(new_labels)


def merge_seque(distribute_res: list) -> list:
    """
    合并相同说话人的连续片段
    
    如果相邻两个片段的说话人相同且时间连续，则合并为一个片段。
    
    Args:
        distribute_res: 片段列表，格式 [[start, end, speaker_id], ...]
    
    Returns:
        合并后的片段列表
    """
    if not distribute_res:
        return []
    
    res = [distribute_res[0]]
    
    for i in range(1, len(distribute_res)):
        # 如果当前片段的说话人与上一个不同，或者时间不连续，则添加新片段
        if distribute_res[i][2] != res[-1][2] or distribute_res[i][0] > res[-1][1]:
            res.append(distribute_res[i])
        else:
            # 否则，扩展上一个片段的结束时间
            res[-1][1] = distribute_res[i][1]
    
    return res


def smooth(res: list, mindur: float = 1.0) -> list:
    """
    平滑处理：移除过短的片段
    
    对于时长小于mindur的片段，将其分配给最近的说话人：
    - 如果是第一个片段，分配给下一个片段的说话人
    - 如果是最后一个片段，分配给前一个片段的说话人
    - 否则，分配给距离更近的片段的说话人
    
    Args:
        res: 片段列表，格式 [[start, end, speaker_id], ...]
        mindur: 最小时长阈值（秒），默认1.0秒
    
    Returns:
        平滑后的片段列表
    """
    # Step 1: 对时间进行四舍五入
    for i in range(len(res)):
        res[i][0] = round(res[i][0], 2)
        res[i][1] = round(res[i][1], 2)
        
        # Step 2: 对于过短的片段，重新分配说话人
        if res[i][1] - res[i][0] < mindur:
            if i == 0:
                # 第一个片段，分配给下一个片段的说话人
                res[i][2] = res[i + 1][2]
            elif i == len(res) - 1:
                # 最后一个片段，分配给前一个片段的说话人
                res[i][2] = res[i - 1][2]
            else:
                # 中间片段，分配给距离更近的片段的说话人
                dist_to_prev = res[i][0] - res[i - 1][1]
                dist_to_next = res[i + 1][0] - res[i][1]
                
                if dist_to_prev <= dist_to_next:
                    res[i][2] = res[i - 1][2]
                else:
                    res[i][2] = res[i + 1][2]
    
    # Step 3: 再次合并相同说话人的连续片段
    res = merge_seque(res)

    return res
