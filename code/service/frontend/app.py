"""
Gradio 前端应用
"""
import gradio as gr
import requests
import websockets
import asyncio
import json
import base64
import numpy as np
import librosa
import threading
import queue
import time
import os
import random
from typing import Optional, List
from config import BACKEND_URL, FRONTEND_PORT, BACKEND_PORT
from backend.utils.logger_manager import LoggerManager

# 后端 API 地址
API_BASE = BACKEND_URL

# 初始化前端logger
logger = LoggerManager.get_frontend_logger()


reference_text_list = [
    "春天到了，万物复苏，花开鸟鸣。阳光洒在大地上，小草从土里探出头来，柳树抽出嫩芽。孩子们在公园里奔跑玩耍，欢声笑语回荡在空中。",
    "清晨的空气格外清新，远处传来了小鸟的啁啾声。路上的行人步履匆匆，开启了新一天的生活。一阵微风吹过，树叶在阳光下沙沙作响，让人心情愉悦。",
    "每当夜晚降临，城市的灯火逐渐亮起。街道上车水马龙，热闹非凡。人们结束一天的工作，回到温馨的家，与家人共享美好时光。"
]



def get_random_reference_text():
    # 随机选择一段文本作为录音参考
    logger.info("[前端点击] 刷新参考文本")
    return random.choice(reference_text_list)

def get_premade_audio_files() -> list:
    """
    获取预制音频文件列表
    """
    try:
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(path, 'audio/short_audio')
        return [f for f in os.listdir(path) if f.endswith('.wav')]
    except Exception as e:
        return ["error",path]

def get_long_audio_files() -> list:
    """
    获取本地长音频文件列表（audio/long_audio目录）
    """
    try:
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        path = os.path.join(path, 'audio/long_audio')
        if os.path.exists(path):
            return [f for f in os.listdir(path) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        return []
    except Exception as e:
        return []

def get_long_audio_file_path(filename: str) -> Optional[str]:
    """
    根据文件名获取本地长音频文件的完整路径
    """
    if not filename or not filename.strip():
        return None
    try:
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        file_path = os.path.join(path, 'audio/long_audio', filename.strip())
        if os.path.exists(file_path):
            return file_path
        return None
    except Exception as e:
        return None

def enroll_speaker(
    speaker_name: str,
    premade_audio: Optional[str] = None,
    upload_audio = None,
    record_audio = None
) -> str:
    """
    注册说话人
    
    Args:
        speaker_name: 说话人名称
        premade_audio: 选择的预制音频文件名（可能为 None）
        upload_audio: 上传的音频文件（可能为 None）
        record_audio: 录制的音频文件（可能为 None）
    
    Returns:
        注册结果消息
    """
    logger.info(f"[前端点击] 注册说话人 - 说话人名称: {speaker_name}, 预制音频: {premade_audio}, "
                f"上传音频: {upload_audio is not None}, 录制音频: {record_audio is not None}")
    
    if not speaker_name or not speaker_name.strip():
        return "❌ 请输入说话人名称"
    
    speaker_name = speaker_name.strip()
    
    # 按优先级选择音频文件：预制音频 > 上传音频 > 录制音频
    audio_file_path = None
    
    # 检查预制音频（优先级最高）
    if premade_audio and isinstance(premade_audio, str) and premade_audio.strip():
        try:
            path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            path = os.path.join(path, 'audio/short_audio', premade_audio)
            if os.path.exists(path):
                audio_file_path = path
        except Exception as e:
            return f"❌ 无法找到预制音频文件: {str(e)}"
    
    # 检查上传音频（优先级第二）
    if not audio_file_path and upload_audio:
        if isinstance(upload_audio, tuple) and len(upload_audio) >= 2:
            audio_path = upload_audio[1]
        elif hasattr(upload_audio, 'name'):
            audio_path = upload_audio.name
        else:
            audio_path = str(upload_audio)
        
        if audio_path and os.path.exists(audio_path):
            audio_file_path = audio_path
    
    # 检查录制音频（优先级最低）
    if not audio_file_path and record_audio:
        if isinstance(record_audio, tuple) and len(record_audio) >= 2:
            audio_path = record_audio[1]
        elif hasattr(record_audio, 'name'):
            audio_path = record_audio.name
        else:
            audio_path = str(record_audio)
        
        if audio_path and os.path.exists(audio_path):
            audio_file_path = audio_path
    
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "❌ 请提供音频文件（选择预制音频、上传音频或录制音频）"
    
    try:
        # 调用后端 API
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            data = {'speaker_name': speaker_name}
            response = requests.post(f"{API_BASE}/api/speaker/enroll", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            # 适配后端返回格式：{success: true, message: "...", data: {speaker_name, embedding_path, ...}}
            if result.get('success'):
                data = result.get('data', {})
                return f"✅ {result['message']}\n说话人: {data.get('speaker_name', speaker_name)}\n保存路径: {data.get('embedding_path', 'N/A')}"
            else:
                return f"❌ {result.get('message', '注册失败')}"
        else:
            error_detail = response.json().get('detail', '未知错误')
            return f"❌ 注册失败: {error_detail}"
    except Exception as e:
        return f"❌ 注册时发生错误: {str(e)}"


def get_registered_speakers() -> List[str]:
    """获取已注册说话人列表"""
    try:
        response = requests.get(f"{API_BASE}/api/speaker/speakers")
        if response.status_code == 200:
            result = response.json()
            # 适配后端返回格式：{success: true, data: {speakers: [...], count: N, ...}}
            if result.get('success'):
                data = result.get('data', {})
                return data.get('speakers', [])
            return []
        return []
    except:
        return []


def get_registered_speakers_list() -> List[str]:
    """
    获取已注册说话人列表（用于下拉框）
    
    Returns:
        说话人名称列表
    """
    return get_registered_speakers()


def get_registered_speakers_display() -> str:
    """获取已注册说话人显示文本"""
    speakers = get_registered_speakers()
    if not speakers:
        return "暂无已注册的说话人"
    
    result = f"已注册说话人 ({len(speakers)} 个):\n\n"
    for i, speaker in enumerate(speakers, 1):
        result += f"{i}. {speaker}\n"
    return result


def delete_speaker(speaker_name: str) -> tuple:
    """
    删除已注册的说话人
    
    Args:
        speaker_name: 要删除的说话人名称
    
    Returns:
        (结果消息, 更新后的列表文本, 更新后的下拉框选项)
    """
    logger.info(f"[前端点击] 删除说话人 - 说话人名称: {speaker_name}")
    
    if not speaker_name or not speaker_name.strip():
        return "❌ 请选择要删除的说话人", get_registered_speakers_display(), gr.update(choices=get_registered_speakers_list())
    
    try:
        response = requests.delete(
            f"{API_BASE}/api/speaker/speakers/{speaker_name.strip()}",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            # 适配后端返回格式：{success: true, message: "...", data: {speaker_name}}
            if result.get('success'):
                message = f"✅ {result.get('message', '删除成功')}"
            else:
                message = f"❌ {result.get('message', '删除失败')}"
            return message, get_registered_speakers_display(), gr.update(choices=get_registered_speakers_list())
        else:
            error_detail = response.json().get('detail', '未知错误')
            return f"❌ 删除失败: {error_detail}", get_registered_speakers_display(), gr.update(choices=get_registered_speakers_list())
    except requests.exceptions.ConnectionError:
        return f"❌ 无法连接到后端服务", get_registered_speakers_display(), gr.update(choices=get_registered_speakers_list())
    except Exception as e:
        return f"❌ 删除时发生错误: {str(e)}", get_registered_speakers_display(), gr.update(choices=get_registered_speakers_list())


def refresh_speakers() -> tuple:
    """
    刷新说话人列表
    
    Returns:
        (更新后的列表文本, 更新后的下拉框选项)
    """
    logger.info("[前端点击] 刷新说话人列表")
    updated_list = get_registered_speakers_display()
    updated_choices = get_registered_speakers_list()
    return updated_list, gr.update(choices=updated_choices)


def recognize_speaker(audio_file, speaker_name: Optional[str], mode: str, threshold: float):
    """识别说话人"""
    logger.info(f"[前端点击] 开始识别 - 音频文件: {audio_file is not None}, "
                f"说话人: {speaker_name}, 模式: {mode}, 阈值: {threshold}")
    
    if not audio_file:
        return "❌ 请先上传或录制音频文件", [], ""
    
    # 处理 Gradio Audio 返回值
    if isinstance(audio_file, tuple) and len(audio_file) >= 2:
        audio_path = audio_file[1]
    elif hasattr(audio_file, 'name'):
        audio_path = audio_file.name
    else:
        audio_path = str(audio_file)
    
    try:
        # 调用后端 API
        with open(audio_path, 'rb') as f:
            files = {'audio_file': f}
            # 适配后端参数格式
            data = {
                'enable_speaker_diarization': 'true',  # 默认启用说话人识别
                'speaker_mode': mode,  # mode -> speaker_mode
                'similarity_threshold': threshold
            }
            # 如果指定了说话人，转换为 registered_speakers 格式（逗号分隔）
            if speaker_name and speaker_name.strip():
                data['registered_speakers'] = speaker_name.strip()
            
            response = requests.post(f"{API_BASE}/api/offline_asr/offline_asr", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            # 适配后端返回格式：{success: true, text: "...", duration: N, segments: [...]}
            if not result.get('success', True):
                error_msg = result.get('error', '识别失败')
                return f"❌ {error_msg}", [], ""
            
            segments = result.get('segments', [])
            if not segments:
                return "✅ 识别完成，但未检测到语音片段", [], result.get('text', '')
            
            # 转换格式：segments -> results，适配前端期望的格式
            results = []
            for seg in segments:
                start_sec = seg.get('start', 0)
                end_sec = seg.get('end', 0)
                duration_sec = end_sec - start_sec
                
                # 处理说话人名称：后端可能返回 "speaker_0", "speaker_1" 或已注册的说话人名称
                speaker = seg.get('speaker', 'unknown')
                
                results.append({
                    'speaker_name': speaker,  # 保持后端返回的speaker值
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'duration_sec': duration_sec,
                    'similarity': seg.get('similarity'),
                    'text': seg.get('text', '').strip()
                })
            
            # 生成摘要
            summary = f"✅ 识别完成！共识别 {len(results)} 个片段\n\n"
            speaker_groups = {}
            for r in results:
                name = r['speaker_name']
                if name not in speaker_groups:
                    speaker_groups[name] = []
                speaker_groups[name].append(r)
            
            summary += f"说话人数量: {len(speaker_groups)}\n\n"
            for name in sorted(speaker_groups.keys()):
                segs = speaker_groups[name]
                total_duration = sum(r['duration_sec'] for r in segs)
                summary += f"{name}: {len(segs)} 个片段, 总时长 {total_duration:.2f}s\n"
            
            # 格式化详细结果
            details_data = []
            for r in results:
                details_data.append([
                    r['speaker_name'],
                    f"{r['start_sec']:.2f}",
                    f"{r['end_sec']:.2f}",
                    f"{r['duration_sec']:.2f}",
                    f"{r['similarity']:.4f}" if r.get('similarity') is not None else "N/A",
                    r.get('text', '').strip() or "(无文本)"
                ])
            
            # 生成合并文本
            merged_text = merge_results_by_speaker(results)
            
            return summary, details_data, merged_text
        else:
            error_detail = response.json().get('detail', response.json().get('error', '未知错误'))
            error_msg = f"❌ 识别失败: {error_detail}"
            return error_msg, [], ""
    except Exception as e:
        error_msg = f"❌ 识别时发生错误: {str(e)}"
        return error_msg, [], ""


def merge_results_by_speaker(results):
    """按说话人合并结果"""
    if not results:
        return ""
    
    merged_results = []
    current_speaker = None
    current_text = ""
    
    for result in results:
        speaker_name = result['speaker_name']
        text = result.get('text', '').strip()
        
        if current_speaker is None:
            current_speaker = speaker_name
            current_text = text
        elif current_speaker == speaker_name:
            if text:
                current_text = current_text + " " + text if current_text else text
        else:
            merged_results.append(f"{current_speaker}：{current_text or '(无文本)'}")
            current_speaker = speaker_name
            current_text = text
    
    if current_speaker is not None:
        merged_results.append(f"{current_speaker}：{current_text or '(无文本)'}")
    
    return "\n".join(merged_results)


# ==================== 流式ASR相关函数 ====================

# WebSocket服务器地址 - 使用后端的pipeline接口以获取完整功能（流式ASR + 离线ASR）
WS_URL_BASE = f"ws://localhost:{BACKEND_PORT}/api/stream_asr/pipeline"


def determine_speaker_mode(speaker_name: Optional[str], selected_mode: str) -> str:
    """
    确定最终使用的说话人识别模式
    
    规则：
    1. 当没有选择说话人的时候，无论选什么模式，都默认cluster
    2. 当选择说话人的时候，默认direct_match，如果有选择，则按选择的处理
    
    Args:
        speaker_name: 选择的说话人名称（可选）
        selected_mode: 用户选择的模式
    
    Returns:
        最终使用的模式
    """
    if not speaker_name or not speaker_name.strip():
        # 没有选择说话人，强制使用cluster
        return "cluster"
    else:
        # 选择了说话人，使用用户选择的模式（默认direct_match）
        if selected_mode in ("cluster", "direct_match", "cluster_match"):
            return selected_mode
        else:
            # 如果用户选择无效，默认使用direct_match
            return "direct_match"

class StreamingASRClient:
    """流式ASR客户端，处理与后端的WebSocket通信"""
    
    def __init__(self):
        self.websocket = None
        self.result_queue = queue.Queue()
        self.is_connected = False
        self.loop = None
        self.thread = None
        self.speaker_name = None
        self.speaker_mode = "cluster"
        self.similarity_threshold = 0.5
    
    async def connect(self, speaker_name: Optional[str] = None, speaker_mode: str = "cluster", similarity_threshold: float = 0.5):
        """连接到WebSocket服务器"""
        try:
            self.speaker_name = speaker_name
            self.speaker_mode = speaker_mode
            self.similarity_threshold = similarity_threshold
            
            # 构建WebSocket URL，使用查询参数配置说话人识别
            ws_url = WS_URL_BASE
            params = []
            params.append("enable_speaker_diarization=true")
            params.append(f"speaker_mode={speaker_mode}")
            params.append(f"similarity_threshold={similarity_threshold}")
            if speaker_name and speaker_name.strip():
                params.append(f"registered_speakers={speaker_name.strip()}")
            
            if params:
                ws_url += "?" + "&".join(params)
            
            # 设置较长的超时时间，避免因为处理时间长而断开
            # ping_interval=20 表示每20秒发送一次ping，保持连接活跃
            # ping_timeout=10 表示等待pong响应的超时时间为10秒
            # close_timeout=10 表示关闭连接的超时时间
            self.websocket = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_connected = True
            
            # 启动接收消息的协程
            asyncio.create_task(self.receive_messages())
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.is_connected = False
    
    async def receive_messages(self):
        """接收WebSocket消息"""
        try:
            while self.is_connected and self.websocket is not None:
                try:
                    # 设置接收超时，避免长时间阻塞
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=300.0)
                    data = json.loads(message)
                    
                    # 后端返回格式：{status: "success", aligned_text: "...", offline_segments: [...], streaming_history: [...]}
                    # 后端每次返回的是累积的完整结果，直接放入队列供后续处理
                    # 注意：不要拆分消息，保持后端返回的完整格式
                    self.result_queue.put(data)
                except asyncio.TimeoutError:
                    # 接收超时，但不一定是连接断开，继续等待
                    logger.debug("接收消息超时，继续等待...")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket连接已关闭（接收消息时）")
                    self.is_connected = False
                    break
                except websockets.exceptions.WebSocketException as e:
                    logger.error(f"WebSocket异常（接收消息时）: {e}")
                    self.is_connected = False
                    break
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败（接收消息时）: {e}")
                    # JSON解析失败不应该断开连接，继续接收下一条消息
                    continue
                except Exception as e:
                    logger.error(f"接收消息时发生未知错误: {e}")
                    # 其他错误也不应该立即断开，继续尝试接收
                    import traceback
                    traceback.print_exc()
                    continue
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket连接已关闭（外层异常）")
            self.is_connected = False
        except Exception as e:
            logger.error(f"接收消息协程出错: {e}")
            self.is_connected = False
            import traceback
            traceback.print_exc()
    
    async def send_audio_chunk(self, audio_data: np.ndarray, sample_rate: int):
        """发送音频chunk - 适配后端格式：发送float32数组的JSON或二进制"""
        if not self.is_connected or self.websocket is None:
            return False
        
        try:
            # 后端期望：{"audio": [floats], "sample_rate": 16000} 或二进制float32
            # 确保是float32格式
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 转换为列表（JSON格式）
            audio_list = audio_data.tolist()
            
            message = {
                "audio": audio_list,
                "sample_rate": sample_rate
            }
            
            await self.websocket.send(json.dumps(message))
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket连接已关闭，无法发送音频chunk")
            self.is_connected = False
            return False
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket错误: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logger.error(f"发送音频chunk时出错: {e}")
            # 检查是否是连接相关的错误
            if "keepalive" in str(e).lower() or "timeout" in str(e).lower() or "closed" in str(e).lower():
                self.is_connected = False
            return False
    
    async def send_audio_end(self):
        """发送音频结束信号"""
        if not self.is_connected or self.websocket is None:
            return False
        
        try:            # 发送一个特殊的结束消息
            message = {
                "audio_end": True
            }
            await self.websocket.send(json.dumps(message))
            return True
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket连接已关闭，无法发送结束信号")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """断开连接"""
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()
    
    def get_result(self, timeout=0.1):
        """获取结果（非阻塞）"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


# 全局客户端实例
streaming_client = StreamingASRClient()

def start_streaming_websocket_loop(speaker_name: Optional[str] = None, speaker_mode: str = "cluster", similarity_threshold: float = 0.5):
    """在新的事件循环中运行WebSocket"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    streaming_client.loop = loop
    loop.run_until_complete(streaming_client.connect(speaker_name, speaker_mode, similarity_threshold))
    loop.run_forever()


def load_local_audio_file(filename: str):
    """加载本地音频文件到Audio组件"""
    logger.info(f"[前端点击] 加载本地音频文件 - 文件名: {filename}")
    if not filename or not filename.strip():
        return None
    
    file_path = get_long_audio_file_path(filename)
    if file_path and os.path.exists(file_path):
        # 返回文件路径，Audio组件可以识别
        return file_path
    return None

def process_streaming_uploaded_audio(audio_file, speaker_name: Optional[str] = None, speaker_mode: str = "direct_match", similarity_threshold: float = 0.5):
    """处理上传的音频文件（流式ASR）"""
    logger.info(f"[前端点击] 处理上传音频（流式ASR） - 音频文件: {audio_file is not None}, "
                f"说话人: {speaker_name}, 模式: {speaker_mode}, 阈值: {similarity_threshold}")
    if audio_file is None:
        return "", "", ""
    
    # 确定最终使用的模式
    final_mode = determine_speaker_mode(speaker_name, speaker_mode)
    
    try:
        # 处理文件路径
        if isinstance(audio_file, tuple) and len(audio_file) >= 2:
            file_path = audio_file[1]
        elif isinstance(audio_file, str):
            file_path = audio_file
        else:
            file_path = str(audio_file)
        
        # 读取音频文件
        audio_data, sr = librosa.load(file_path, sr=None, mono=True)
        
        # 重采样到16kHz
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # 连接到WebSocket
        if not streaming_client.is_connected:
            if streaming_client.thread is None or not streaming_client.thread.is_alive():
                streaming_client.thread = threading.Thread(
                    target=start_streaming_websocket_loop, 
                    args=(speaker_name, final_mode, similarity_threshold),
                    daemon=True
                )
                streaming_client.thread.start()
                # 等待连接建立
                for _ in range(10):
                    if streaming_client.is_connected:
                        break
                    time.sleep(0.5)
        
        if not streaming_client.is_connected:
            return "错误: 无法连接到ASR服务器", "", ""
        
        # 清空结果队列
        while not streaming_client.result_queue.empty():
            try:
                streaming_client.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # 模拟流式传输：将音频分成chunk发送
        chunk_size = 3200  # 200ms at 16kHz
        streaming_results = []
        offline_results = []
        vad_results = []
        final_result = None
        
        async def send_audio():
            try:
                for i in range(0, len(audio_data), chunk_size):
                    # 检查连接状态，如果断开则停止发送
                    if not streaming_client.is_connected:
                        logger.warning(f"⚠️ 连接已断开，停止发送音频（已发送 {i}/{len(audio_data)} 采样点）")
                        return
                    
                    chunk = audio_data[i:i+chunk_size]
                    if len(chunk) > 0:
                        success = await streaming_client.send_audio_chunk(chunk, sr)
                        if not success:
                            logger.warning(f"⚠️ 发送音频chunk失败，停止发送（已发送 {i}/{len(audio_data)} 采样点）")
                            return
                        await asyncio.sleep(0.05)  # 减少延迟
                
                # 检查连接状态
                if not streaming_client.is_connected:
                    logger.warning("⚠️ 连接已断开，无法发送最后剩余的音频")
                    return
                
                # 发送最后剩余的音频
                remaining = len(audio_data) % chunk_size
                if remaining > 0:
                    last_chunk = audio_data[-remaining:]
                    success = await streaming_client.send_audio_chunk(last_chunk, sr)
                    if not success:
                        logger.warning("⚠️ 发送最后音频chunk失败")
                        return
                    await asyncio.sleep(0.05)
                
                # 等待处理完成（给后端一些时间处理最后的chunk）
                await asyncio.sleep(2)
                
                # 再次检查连接状态，只有在连接正常时才发送结束信号
                if streaming_client.is_connected:
                    await streaming_client.send_audio_end()
                else:
                    logger.warning("⚠️ 连接已断开，无法发送结束信号")
            except Exception as e:
                logger.error(f"发送音频时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 在事件循环中发送音频
        if streaming_client.loop and streaming_client.loop.is_running():
            asyncio.run_coroutine_threadsafe(send_audio(), streaming_client.loop)
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(send_audio())
            finally:
                loop.close()
        
        # 收集结果 - 后端每次返回累积的完整结果
        start_time = time.time()
        timeout = 120
        last_result = None
        
        while time.time() - start_time < timeout:
            result = streaming_client.get_result()
            if result:
                # 后端返回的是完整的pipeline结果：{status: "success", streaming_history: [...], offline_segments: [...]}
                if "status" in result and result.get("status") == "success":
                    last_result = result  # 保存最新的完整结果
                    # start_time = time.time()  # 重置超时计时器
            time.sleep(0.1)
        
        # # 等待最后的结果（确保获取到最终状态）
        # for _ in range(50):
        #     result = streaming_client.get_result()
        #     if result and "status" in result and result.get("status") == "success":
        #         last_result = result
        #     time.sleep(0.1)
        
        # 使用最新的完整结果
        if last_result:
            # 处理VAD片段
            if "vad_segments" in last_result and last_result["vad_segments"]:
                vad_results = last_result["vad_segments"]
            
            # 处理流式历史
            if "streaming_history" in last_result and last_result["streaming_history"]:
                for item in last_result["streaming_history"]:
                    streaming_results.append({
                        "start_time": item.get("timestamp", 0),
                        "end_time": item.get("timestamp", 0) + 0.2,
                        "text": item.get("text", ""),
                        "speaker": "",
                        "similarity": None
                    })
            
            # 处理离线片段
            if "offline_segments" in last_result and last_result["offline_segments"]:
                for seg in last_result["offline_segments"]:
                    offline_results.append({
                        "start_time": seg.get("start", 0),
                        "end_time": seg.get("end", 0),
                        "text": seg.get("text", ""),
                        "speaker": seg.get("speaker", ""),
                        "similarity": seg.get("similarity")
                    })
        
        # 格式化结果显示
        streaming_text = format_streaming_results(streaming_results, is_streaming=True)
        offline_text = format_streaming_results(offline_results, is_streaming=False)
        vad_text = format_vad_results(vad_results)
        
        return streaming_text, offline_text, vad_text
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"错误: {str(e)}", "", ""


def format_streaming_results(results, is_streaming=True):
    """格式化流式ASR结果显示"""
    if not results:
        return "（暂无结果）"
    
    lines = []
    for idx, result in enumerate(results, 1):
        start_time = result.get("start_time", 0)
        end_time = result.get("end_time", 0)
        text = result.get("text", "")
        speaker = result.get("speaker", "")
        similarity = result.get("similarity")
        
        if is_streaming:
            lines.append(f"{idx}. [{start_time:.2f}s-{end_time:.2f}s] [流式] {text}")
        else:
            speaker_tag = f"[{speaker}]" if speaker else ""
            sim_tag = f"(相似度:{similarity:.3f})" if similarity is not None else ""
            lines.append(f"{idx}. [{start_time:.2f}s-{end_time:.2f}s] {speaker_tag} {sim_tag} {text} ✓")
    
    return "\n".join(lines)


def format_vad_results(vad_segments):
    """格式化VAD结果显示"""
    if not vad_segments:
        return "（暂无VAD结果）"
    
    lines = []
    for idx, seg in enumerate(vad_segments, 1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start
        lines.append(f"{idx}. [{start:.2f}s-{end:.2f}s] 时长: {duration:.2f}s")
    
    return "\n".join(lines)


# 实时音频处理的状态
realtime_streaming_results = []
realtime_offline_results = []
realtime_vad_results = []
realtime_is_recording = False
realtime_audio_buffer = np.array([], dtype=np.float32)
realtime_target_sample_rate = 16000
realtime_target_chunk_duration_ms = 200
realtime_target_chunk_samples = int(realtime_target_sample_rate * realtime_target_chunk_duration_ms / 1000)


def start_realtime_streaming_recording(speaker_name: Optional[str] = None, speaker_mode: str = "direct_match", similarity_threshold: float = 0.5):
    """开始实时录音（流式ASR）"""
    logger.info(f"[前端点击] 开始实时录音（流式ASR） - 说话人: {speaker_name}, 模式: {speaker_mode}, 阈值: {similarity_threshold}")
    global realtime_is_recording, realtime_streaming_results, realtime_offline_results, realtime_vad_results, realtime_audio_buffer
    
    # 确定最终使用的模式
    final_mode = determine_speaker_mode(speaker_name, speaker_mode)
    
    realtime_is_recording = True
    realtime_streaming_results = []
    realtime_offline_results = []
    realtime_vad_results = []
    realtime_audio_buffer = np.array([], dtype=np.float32)
    
    # 连接到WebSocket
    if not streaming_client.is_connected:
        if streaming_client.thread is None or not streaming_client.thread.is_alive():
            streaming_client.thread = threading.Thread(
                target=start_streaming_websocket_loop,
                args=(speaker_name, final_mode, similarity_threshold),
                daemon=True
            )
            streaming_client.thread.start()
            for _ in range(10):
                if streaming_client.is_connected:
                    break
                time.sleep(0.5)
    
    return "开始录音..."


def stop_realtime_streaming_recording():
    """停止实时录音（流式ASR）"""
    logger.info("[前端点击] 停止实时录音（流式ASR）")
    global realtime_is_recording, realtime_audio_buffer
    
    realtime_is_recording = False
    
    # 发送缓冲区中剩余的音频数据
    if len(realtime_audio_buffer) > 0 and streaming_client.is_connected and streaming_client.loop and streaming_client.loop.is_running():
        asyncio.run_coroutine_threadsafe(
            streaming_client.send_audio_chunk(realtime_audio_buffer, realtime_target_sample_rate),
            streaming_client.loop
        )
        realtime_audio_buffer = np.array([], dtype=np.float32)
    
    # 发送结束信号
    if streaming_client.is_connected and streaming_client.loop and streaming_client.loop.is_running():
        asyncio.run_coroutine_threadsafe(streaming_client.send_audio_end(), streaming_client.loop)
    
    return "停止录音"


def process_realtime_streaming_audio(audio, speaker_name: Optional[str] = None, speaker_mode: str = "direct_match", similarity_threshold: float = 0.5):
    """处理实时音频输入（流式ASR），实现200ms窗口"""
    global realtime_streaming_results, realtime_offline_results, realtime_vad_results, realtime_audio_buffer
    
    # 确定最终使用的模式（仅在首次连接时使用，后续使用已连接的模式）
    # 注意：这里不重新连接，因为已经在start_realtime_streaming_recording中连接了
    
    if audio is None or not realtime_is_recording:
        return format_streaming_results(realtime_streaming_results, is_streaming=True), format_streaming_results(realtime_offline_results, is_streaming=False), format_vad_results(realtime_vad_results)
    
    try:
        sample_rate, audio_data = audio
        
        # 确保是单通道
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 转换为浮点类型
        if audio_data.dtype != np.float32:
            if audio_data.dtype in [np.int16, np.int32]:
                max_val = np.iinfo(audio_data.dtype).max
                audio_data = audio_data.astype(np.float32) / max_val
            else:
                audio_data = audio_data.astype(np.float32)
        
        # 重采样到16kHz
        if sample_rate != realtime_target_sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=realtime_target_sample_rate)
            sample_rate = realtime_target_sample_rate
        
        # 将新音频数据添加到缓冲区
        realtime_audio_buffer = np.concatenate([realtime_audio_buffer, audio_data])
        
        # 当缓冲区达到或超过200ms时，发送200ms的chunk
        while len(realtime_audio_buffer) >= realtime_target_chunk_samples:
            chunk_200ms = realtime_audio_buffer[:realtime_target_chunk_samples]
            realtime_audio_buffer = realtime_audio_buffer[realtime_target_chunk_samples:]
            
            # 发送200ms的音频chunk
            if streaming_client.is_connected and streaming_client.loop and streaming_client.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    streaming_client.send_audio_chunk(chunk_200ms, realtime_target_sample_rate), 
                    streaming_client.loop
                )
        
        # 收集结果 - 后端每次返回累积的完整结果
        while True:
            result = streaming_client.get_result()
            if result is None:
                break
            
            # 处理完整的pipeline结果：{status: "success", streaming_history: [...], offline_segments: [...], vad_segments: [...]}
            if "status" in result and result.get("status") == "success":
                # 处理VAD片段（累积结果，需要更新）
                if "vad_segments" in result and result["vad_segments"]:
                    # 更新VAD结果（使用最新的完整列表）
                    realtime_vad_results = result["vad_segments"]
                
                # 处理流式历史（累积结果，需要去重）
                if "streaming_history" in result and result["streaming_history"]:
                    for item in result["streaming_history"]:
                        data = {
                            "start_time": item.get("timestamp", 0),
                            "end_time": item.get("timestamp", 0) + 0.2,
                            "text": item.get("text", ""),
                            "speaker": "",
                            "similarity": None
                        }
                        # 避免重复添加（基于时间戳判断）
                        found = False
                        for existing in realtime_streaming_results:
                            if abs(existing.get("start_time", 0) - data.get("start_time", 0)) < 0.1:
                                found = True
                                break
                        if not found:
                            realtime_streaming_results.append(data)
                
                # 处理离线片段（累积结果，需要更新）
                if "offline_segments" in result and result["offline_segments"]:
                    for seg in result["offline_segments"]:
                        data = {
                            "start_time": seg.get("start", 0),
                            "end_time": seg.get("end", 0),
                            "text": seg.get("text", ""),
                            "speaker": seg.get("speaker", ""),
                            "similarity": seg.get("similarity")
                        }
                        # 更新或添加离线结果（基于开始时间匹配）
                        found = False
                        for idx, existing in enumerate(realtime_offline_results):
                            if abs(existing.get("start_time", 0) - data.get("start_time", 0)) < 0.1:
                                realtime_offline_results[idx] = data
                                found = True
                                break
                        if not found:
                            realtime_offline_results.append(data)
        
        return format_streaming_results(realtime_streaming_results, is_streaming=True), format_streaming_results(realtime_offline_results, is_streaming=False), format_vad_results(realtime_vad_results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"错误: {str(e)}", "", ""


def update_realtime_streaming_results():
    """定期更新实时结果显示（流式ASR）"""
    global realtime_streaming_results, realtime_offline_results, realtime_vad_results
    
    # 收集新结果 - 后端每次返回累积的完整结果
    while True:
        result = streaming_client.get_result()
        # logger.info(f"获取实时结果: {result}")
        if result is None:
            break
        
        # 处理完整的pipeline结果：{status: "success", streaming_history: [...], offline_segments: [...], vad_segments: [...]}
        if "status" in result and result.get("status") == "success":
            # 处理VAD片段（累积结果，需要更新）
            if "vad_segments" in result and result["vad_segments"]:
                # 更新VAD结果（使用最新的完整列表）
                realtime_vad_results = result["vad_segments"]
            
            # 处理流式历史（累积结果，需要去重）
            if "streaming_history" in result and result["streaming_history"]:
                for item in result["streaming_history"]:
                    data = {
                        "start_time": item.get("timestamp", 0),
                        "end_time": item.get("timestamp", 0) + 0.2,
                        "text": item.get("text", ""),
                        "speaker": "",
                        "similarity": None
                    }
                    # 避免重复添加（基于时间戳判断）
                    found = False
                    for existing in realtime_streaming_results:
                        if abs(existing.get("start_time", 0) - data.get("start_time", 0)) < 0.1:
                            found = True
                            break
                    if not found:
                        realtime_streaming_results.append(data)
            
            # 处理离线片段（累积结果，需要更新）
            if "offline_segments" in result and result["offline_segments"]:
                logger.info(f"处理离线片段: {result['offline_segments']}")
                for seg in result["offline_segments"]:
                    data = {
                        "start_time": seg.get("start", 0),
                        "end_time": seg.get("end", 0),
                        "text": seg.get("text", ""),
                        "speaker": seg.get("speaker", ""),
                        "similarity": seg.get("similarity")
                    }
                    # 更新或添加离线结果（基于开始时间匹配）
                    found = False
                    for idx, existing in enumerate(realtime_offline_results):
                        if abs(existing.get("start_time", 0) - data.get("start_time", 0)) < 0.1:
                            realtime_offline_results[idx] = data
                            found = True
                            break
                    if not found:
                        realtime_offline_results.append(data)
    
    return format_streaming_results(realtime_streaming_results, is_streaming=True), format_streaming_results(realtime_offline_results, is_streaming=False), format_vad_results(realtime_vad_results)


# ==================== 创建 Gradio 界面 ====================

# 创建 Gradio 界面
with gr.Blocks(title="Local Clinical Scribe", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Local Clinical Scribe")
    gr.Markdown("### 本地优先的临床对话转写与结构化病历草稿工具")
    
    with gr.Tabs():
        # 语音注册页面
        with gr.Tab("📝 语音注册"):
            gr.Markdown("### 对说话人进行声纹注册")
            
            with gr.Group():
                with gr.Row():
                    enroll_speaker_name = gr.Textbox(
                        label="说话人名称",
                        placeholder="请输入说话人名称"
                    )
                with gr.Row():
                    with gr.Column():
                        premade_audio_dropdown = gr.Dropdown(
                            label="选择预制音频",
                            choices=get_premade_audio_files(),
                            value=None,
                        )
                        upload_audio = gr.Audio(
                            label="上传音频",
                            type="filepath",
                            sources=["upload"]
                        )
                    with gr.Column():
                        reference_textbox = gr.Textbox(
                            label="录音参考文本",
                            value=get_random_reference_text(),
                            lines=4,
                            max_lines=6,
                            interactive=False,
                            show_copy_button=True
                        )
                        refresh_text_btn = gr.Button("🔄 更换参考文本", variant="secondary")
                        record_audio = gr.Audio(
                            label="录制音频",
                            type="filepath",
                            sources=["microphone"]
                        )
                with gr.Row():
                    enroll_btn = gr.Button("🚀 开始注册", variant="primary", size="lg")
                with gr.Row():
                    enroll_result = gr.Textbox(label="注册结果", lines=8, interactive=False)
            
            with gr.Group():
                gr.Markdown("---")
                gr.Markdown("### 📋 已注册说话人列表")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        speakers_display = gr.Textbox(
                            label="",
                            value=get_registered_speakers_display(),
                            lines=10,
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("🔄 刷新列表", variant="secondary")
                
                # 删除操作区域
                with gr.Row():
                    delete_speaker_dropdown = gr.Dropdown(
                        label="选择要删除的说话人",
                        choices=get_registered_speakers_list(),
                        value=None,
                        allow_custom_value=False,
                        scale=3,
                        info="请谨慎操作，删除后无法恢复"
                    )
                    delete_btn = gr.Button("🗑️ 删除选中说话人", variant="stop", scale=1)
                
                delete_result = gr.Textbox(
                    label="删除结果",
                    lines=2,
                    interactive=False
                )
            
            # 刷新列表按钮事件
            refresh_btn.click(
                fn=refresh_speakers,
                inputs=[],
                outputs=[speakers_display, delete_speaker_dropdown]
            )
            
            # 更换参考文本按钮事件
            refresh_text_btn.click(
                fn=get_random_reference_text,
                inputs=[],
                outputs=[reference_textbox]
            )
            
            # 注册按钮事件
            enroll_btn.click(
                fn=enroll_speaker,
                inputs=[enroll_speaker_name, premade_audio_dropdown, upload_audio, record_audio],
                outputs=[enroll_result]
            ).then(
                fn=refresh_speakers,
                inputs=[],
                outputs=[speakers_display, delete_speaker_dropdown]
            )
            
            # 删除说话人按钮事件
            delete_btn.click(
                fn=delete_speaker,
                inputs=[delete_speaker_dropdown],
                outputs=[delete_result, speakers_display, delete_speaker_dropdown]
            )
        
        # 语音识别页面
        with gr.Tab("🎯 语音识别"):
            gr.Markdown("### 对音频进行说话人识别和 ASR 转录")
            
            with gr.Row():
                with gr.Column():
                    recog_audio = gr.Audio(
                        label="上传音频或录制音频",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )
                    recog_speaker = gr.Dropdown(
                        label="选择已注册的说话人（可选）",
                        choices=get_registered_speakers(),
                        value=None,
                        allow_custom_value=False
                    )
                    recog_refresh_btn = gr.Button("🔄 刷新说话人列表", variant="secondary", size="sm")
                    recog_mode = gr.Radio(
                        label="说话人识别模式",
                        choices=[
                            ("自动聚类", "cluster"),
                            ("直接匹配", "direct_match"),
                            ("聚类+匹配", "cluster_match")
                        ],
                        value="cluster"
                    )
                    recog_threshold = gr.Slider(
                        label="相似度阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05
                    )
                    recog_btn = gr.Button("🚀 开始识别", variant="primary", size="lg")
                
                with gr.Column():
                    recog_result_summary = gr.Textbox(
                        label="识别结果摘要",
                        lines=10,
                        interactive=False
                    )
            
            recog_result_details = gr.Dataframe(
                label="详细识别结果",
                headers=["说话人", "开始时间(秒)", "结束时间(秒)", "时长(秒)", "相似度", "文本"],
                interactive=False,
                wrap=True
            )
            
            recog_result_text = gr.Textbox(
                label="识别文本（按说话人合并）",
                lines=15,
                interactive=False
            )
            
            # 更新说话人列表
            def update_speaker_list():
                logger.info("[前端点击] 刷新识别页面说话人列表")
                return gr.update(choices=get_registered_speakers())
            
            recog_btn.click(
                fn=recognize_speaker,
                inputs=[recog_audio, recog_speaker, recog_mode, recog_threshold],
                outputs=[recog_result_summary, recog_result_details, recog_result_text]
            )
            
            recog_refresh_btn.click(
                fn=update_speaker_list,
                inputs=[],
                outputs=[recog_speaker]
            )
        
        # 流式ASR页面
        with gr.Tab("🎙️ 流式ASR"):
            gr.Markdown("### 实时流式语音识别（支持麦克风和上传音频）")
            
            with gr.Row():
                with gr.Column():
                    streaming_speaker_dropdown = gr.Dropdown(
                        label="选择已注册的说话人（可选）",
                        choices=get_registered_speakers(),
                        value=None,
                        allow_custom_value=False
                    )
                    streaming_refresh_btn = gr.Button("🔄 刷新说话人列表", variant="secondary", size="sm")
                    streaming_mode = gr.Radio(
                        label="说话人识别模式",
                        choices=[
                            ("自动聚类", "cluster"),
                            ("直接匹配", "direct_match"),
                            ("聚类+匹配", "cluster_match")
                        ],
                        value="direct_match",
                        info="未选择说话人时自动使用'自动聚类'模式"
                    )
                    streaming_threshold = gr.Slider(
                        label="相似度阈值",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        info="仅在匹配模式（直接匹配/聚类+匹配）时有效"
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("📁 上传音频文件"):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    streaming_local_audio_dropdown = gr.Dropdown(
                                        label="选择本地音频文件（audio/long_audio）",
                                        choices=get_long_audio_files(),
                                        value=None,
                                        info="从本地目录选择音频文件"
                                    )
                                    streaming_load_local_btn = gr.Button("加载本地音频", variant="secondary")
                                with gr.Column(scale=1):
                                    streaming_upload_audio = gr.Audio(
                                        label="上传音频文件",
                                        type="filepath",
                                        sources=["upload"]
                                    )
                            streaming_upload_btn = gr.Button("开始处理", variant="primary")
                            streaming_upload_status = gr.Textbox(label="状态", value="等待上传文件", interactive=False)
                        
                        with gr.Tab("🎤 实时音频输入"):
                            streaming_realtime_audio = gr.Audio(
                                label="实时音频输入（16kHz单通道）",
                                type="numpy",
                                sources=["microphone"],
                                streaming=True,
                                waveform_options={"sample_rate": 16000}
                            )
                            with gr.Row():
                                streaming_start_btn = gr.Button("开始录音", variant="primary")
                                streaming_stop_btn = gr.Button("停止录音")
                            streaming_status_text = gr.Textbox(label="状态", value="未开始", interactive=False)
                
                with gr.Column():
                    streaming_vad_output = gr.Textbox(
                        label="VAD检测结果（语音片段）",
                        lines=8,
                        max_lines=12,
                        interactive=False,
                        info="显示VAD检测到的语音片段时间范围"
                    )
                    streaming_streaming_output = gr.Textbox(
                        label="流式ASR结果（实时）",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
                    streaming_offline_output = gr.Textbox(
                        label="离线ASR结果（修正后）",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
            
            # 加载本地音频文件事件
            def process_local_audio_file(filename, speaker_name, speaker_mode, similarity_threshold):
                """处理本地音频文件的包装函数"""
                logger.info(f"[前端点击] 处理本地音频文件（流式ASR） - 文件名: {filename}, "
                            f"说话人: {speaker_name}, 模式: {speaker_mode}, 阈值: {similarity_threshold}")
                if not filename:
                    return "", "", ""
                file_path = get_long_audio_file_path(filename)
                if file_path:
                    return process_streaming_uploaded_audio(file_path, speaker_name, speaker_mode, similarity_threshold)
                return "错误: 无法找到本地音频文件", "", ""
            
            streaming_load_local_btn.click(
                fn=load_local_audio_file,
                inputs=[streaming_local_audio_dropdown],
                outputs=[streaming_upload_audio]
            ).then(
                fn=process_local_audio_file,
                inputs=[streaming_local_audio_dropdown, streaming_speaker_dropdown, streaming_mode, streaming_threshold],
                outputs=[streaming_streaming_output, streaming_offline_output, streaming_vad_output]
            )
            
            # 上传文件处理事件
            streaming_upload_btn.click(
                fn=process_streaming_uploaded_audio,
                inputs=[streaming_upload_audio, streaming_speaker_dropdown, streaming_mode, streaming_threshold],
                outputs=[streaming_streaming_output, streaming_offline_output, streaming_vad_output]
            )
            
            # 实时录音按钮事件
            streaming_start_btn.click(
                fn=start_realtime_streaming_recording,
                inputs=[streaming_speaker_dropdown, streaming_mode, streaming_threshold],
                outputs=[streaming_status_text]
            )
            
            streaming_stop_btn.click(
                fn=stop_realtime_streaming_recording,
                inputs=None,
                outputs=[streaming_status_text]
            )
            
            # 实时音频流处理
            streaming_realtime_audio.stream(
                fn=process_realtime_streaming_audio,
                inputs=[streaming_realtime_audio, streaming_speaker_dropdown, streaming_mode, streaming_threshold],
                outputs=[streaming_streaming_output, streaming_offline_output, streaming_vad_output]
            )
            
            # 定期更新结果（每500ms）
            demo.load(
                fn=update_realtime_streaming_results,
                inputs=None,
                outputs=[streaming_streaming_output, streaming_offline_output, streaming_vad_output],
                every=0.5
            )
            
            # 更新说话人列表
            def update_streaming_speaker_list():
                logger.info("[前端点击] 刷新流式ASR页面说话人列表")
                return gr.update(choices=get_registered_speakers())
            
            streaming_refresh_btn.click(
                fn=update_streaming_speaker_list,
                inputs=[],
                outputs=[streaming_speaker_dropdown]
            )


if __name__ == "__main__":
    logger.info(f"\n{'='*70}")
    logger.info("🚀 启动 Gradio 前端服务")
    logger.info(f"{'='*70}")
    logger.info(f"前端地址: http://0.0.0.0:{FRONTEND_PORT}")
    logger.info(f"后端 API: {API_BASE}")
    logger.info(f"{'='*70}\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=FRONTEND_PORT,
        share=False,
        inbrowser=True
    )

