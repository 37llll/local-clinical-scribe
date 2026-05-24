# Local Clinical Scribe / 本地临床转写结构化助手

Local-first clinical conversation documentation tool.

本地优先的临床对话转写与结构化病历草稿工具。

- [English](#english)
- [中文](#中文)

---

## English

Local Clinical Scribe turns recorded or streamed Chinese medical conversations
into:

- diarized transcript segments
- timestamped ASR output
- reviewable structured note drafts
- evidence links back to source transcript spans

Version `0.4.0` adds the first complete product workbench at `/app`: draft from
transcript JSON, edit structured sections, finalize the note, export
Markdown/JSON, inspect saved encounters, and view runtime capability status.

Version `0.3.0` was the first public-ready release. It keeps the v0.2 product
loop and adds explicit repository safety guardrails: synthetic demo data only,
public safety scanning, security policy, third-party notices, and release
checklists.

Version `0.2.0` turned the first draft API into a small product loop:
transcription, deterministic clinical note drafting, local encounter storage,
clinician finalization, and Markdown/JSON export. The note draft is never a
diagnosis or final medical record. It is clinician-reviewed documentation
support.

### Product Boundary

This project is designed for documentation support:

1. Capture or upload audio.
2. Run local ASR, VAD, punctuation, and optional speaker diarization.
3. Generate a structured draft from transcript evidence.
4. Let a clinician review, edit, and approve the final note.

It should not be used to automate diagnosis, prescribe medication, or make
clinical decisions without licensed clinician review.

### Current Capabilities

- FastAPI backend with REST and WebSocket endpoints.
- First-party browser workbench at `/app`.
- Offline ASR endpoint for uploaded audio.
- Streaming ASR/VAD WebSocket endpoints.
- Speaker enrollment, listing, deletion, and verification.
- Clinical note draft API:
  - `POST /api/clinical_note/from_transcript`
  - `POST /api/clinical_note/from_audio`
- Encounter review and export API:
  - `GET /api/clinical_note/encounters`
  - `GET /api/clinical_note/encounters/{encounter_id}`
  - `POST /api/clinical_note/finalize`
  - `GET /api/clinical_note/encounters/{encounter_id}/export?format=markdown`
- Lazy model loading, so the API can start before models are downloaded.
- Local-only runtime data directories for audio and speaker embeddings.

### Repository Hygiene

This repository intentionally does not commit:

- patient or sample audio
- speaker embeddings
- logs
- downloaded model artifacts
- historical internal Git history

Runtime data lives under `data/` by default and is ignored by Git.

### Quick Start

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements_cpu.txt
```

Download models when you want to run ASR:

```powershell
cd pretrained_models
python download_models.py
```

Start the API:

```powershell
cd code/service
python start_backend.py
```

Open:

- Product workbench: `http://localhost:63100/app`
- API docs: `http://localhost:63100/docs`
- Health check: `http://localhost:63100/health`
- Runtime capabilities: `http://localhost:63100/capabilities`

Before public release or external sharing:

```powershell
python scripts/public_safety_scan.py
```

### Configuration

Useful environment variables:

```powershell
$env:PROJECT_ROOT="C:\path\to\local-clinical-scribe"
$env:LOCAL_CLINICAL_SCRIBE_DEVICE="cpu"
$env:PRELOAD_MODELS="false"
$env:BACKEND_PORT="63100"
$env:FRONTEND_PORT="63101"
$env:ENCOUNTER_DIR="C:\path\to\local-clinical-scribe\data\encounters"
```

Use `LOCAL_CLINICAL_SCRIBE_DEVICE=cuda:0` when you have a compatible GPU and
local models.

The core clinical note APIs can start without audio dependencies installed.
Audio endpoints appear only when optional runtime dependencies such as FunASR
and librosa are available; check `/capabilities` for the current state.

### Review And Export

Draft endpoints save a local encounter record under `data/encounters`.
Review clients can submit edited sections to:

```text
POST /api/clinical_note/finalize
```

Then export the reviewed note:

```text
GET /api/clinical_note/encounters/{encounter_id}/export?format=markdown
GET /api/clinical_note/encounters/{encounter_id}/export?format=json
```

### Example: Draft From Transcript

The example below is synthetic demo data and does not describe a real patient
or clinician.

```json
{
  "encounter_id": "demo-001",
  "template": "soap",
  "segments": [
    {
      "id": "seg_0001",
      "start": 0.0,
      "end": 3.2,
      "speaker": "patient",
      "text": "我这两天咳嗽，晚上有点发热。"
    },
    {
      "id": "seg_0002",
      "start": 3.5,
      "end": 8.1,
      "speaker": "doctor",
      "text": "建议先做血常规和胸片检查，注意休息，多喝水。"
    }
  ]
}
```

Send it to:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:63100/api/clinical_note/from_transcript `
  -Method Post `
  -ContentType "application/json" `
  -Body (Get-Content examples/clinical_note_request.json -Raw)
```

### Version Plan

- `0.1.x`: local transcription, speaker handling, deterministic note draft.
- `0.2.x`: local encounter persistence, clinician review finalization,
  Markdown/JSON export.
- `0.3.x`: public release readiness, safety scan, security docs, demo data
  policy.
- `0.4.x`: first-party browser workbench for draft, review, finalize, export.
- `0.5.x`: richer export templates and DOCX.
- `0.6.x`: evidence viewer, audit log, quality metrics.
- `0.7.x`: optional LLM structuring with schema validation and citations.
- `0.8.x`: FHIR-compatible export and integration hooks.

See [docs/PRODUCT_PLAN.md](docs/PRODUCT_PLAN.md) for the working roadmap.

---

## 中文

Local Clinical Scribe 是一个本地优先的临床对话文档工具，用于将录音或实时
流式中文医疗对话转换为：

- 区分说话人的转写片段
- 带时间戳的 ASR 输出
- 可审核、可编辑的结构化病历草稿
- 可回溯到原始转写片段的证据链接

`0.4.0` 版本新增了第一个完整产品工作台 `/app`：可以从 transcript JSON
生成草稿，编辑结构化段落，完成审核定稿，导出 Markdown/JSON，查看已保存
encounter，并查看当前运行能力状态。

`0.3.0` 是第一个面向公开仓库的版本。它保留了 v0.2 的产品闭环，并增加了
明确的公开安全约束：只保留合成 demo 数据、公开安全扫描、Security Policy、
第三方声明和 release checklist。

`0.2.0` 将最初的草稿 API 演进成了一个小型产品闭环：转写、确定性病历草稿
生成、本地 encounter 存储、医生审核定稿，以及 Markdown/JSON 导出。生成
结果不是诊断结论，也不是最终病历，只能作为经临床人员审核后的文档辅助。

### 产品边界

本项目定位于文档辅助：

1. 采集或上传音频。
2. 在本地运行 ASR、VAD、标点恢复，以及可选的说话人区分。
3. 基于 transcript 证据生成结构化草稿。
4. 由临床人员审核、编辑并确认最终记录。

本项目不应用于自动诊断、自动开药，或在没有持证临床人员审核的情况下做出
临床决策。

### 当前能力

- 基于 FastAPI 的后端，提供 REST 与 WebSocket 接口。
- 第一方浏览器工作台：`/app`。
- 上传音频的离线 ASR 接口。
- 流式 ASR/VAD WebSocket 接口。
- 说话人注册、列表、删除和验证。
- 临床病历草稿 API：
  - `POST /api/clinical_note/from_transcript`
  - `POST /api/clinical_note/from_audio`
- Encounter 审核与导出 API：
  - `GET /api/clinical_note/encounters`
  - `GET /api/clinical_note/encounters/{encounter_id}`
  - `POST /api/clinical_note/finalize`
  - `GET /api/clinical_note/encounters/{encounter_id}/export?format=markdown`
- 懒加载模型：模型未下载时 API 也可以先启动。
- 音频与说话人 embedding 等运行时数据只保存在本地目录。

### 仓库卫生

本仓库有意不提交以下内容：

- 患者音频或样例音频
- 说话人 embedding
- 日志
- 下载后的模型文件
- 历史内部 Git 记录

默认运行数据位于 `data/`，并被 Git 忽略。

### 快速开始

创建并激活 Python 环境，然后安装依赖：

```powershell
pip install -r requirements_cpu.txt
```

如果需要运行 ASR，下载本地模型：

```powershell
cd pretrained_models
python download_models.py
```

启动 API：

```powershell
cd code/service
python start_backend.py
```

打开：

- 产品工作台：`http://localhost:63100/app`
- API 文档：`http://localhost:63100/docs`
- 健康检查：`http://localhost:63100/health`
- 运行能力状态：`http://localhost:63100/capabilities`

公开发布或对外分享前，运行安全扫描：

```powershell
python scripts/public_safety_scan.py
```

### 配置

常用环境变量：

```powershell
$env:PROJECT_ROOT="C:\path\to\local-clinical-scribe"
$env:LOCAL_CLINICAL_SCRIBE_DEVICE="cpu"
$env:PRELOAD_MODELS="false"
$env:BACKEND_PORT="63100"
$env:FRONTEND_PORT="63101"
$env:ENCOUNTER_DIR="C:\path\to\local-clinical-scribe\data\encounters"
```

如果有兼容 GPU 和本地模型，可以使用 `LOCAL_CLINICAL_SCRIBE_DEVICE=cuda:0`。

核心临床病历 API 不依赖音频相关依赖即可启动。只有当 FunASR、librosa 等
可选运行依赖可用时，音频接口才会启用；可以通过 `/capabilities` 查看当前
状态。

### 审核与导出

草稿接口会在 `data/encounters` 下保存本地 encounter 记录。审核客户端可以
将编辑后的 sections 提交到：

```text
POST /api/clinical_note/finalize
```

然后导出已审核记录：

```text
GET /api/clinical_note/encounters/{encounter_id}/export?format=markdown
GET /api/clinical_note/encounters/{encounter_id}/export?format=json
```

### 示例：从 Transcript 生成草稿

下面是合成 demo 数据，不代表任何真实患者或临床人员。

```json
{
  "encounter_id": "demo-001",
  "template": "soap",
  "segments": [
    {
      "id": "seg_0001",
      "start": 0.0,
      "end": 3.2,
      "speaker": "patient",
      "text": "我这两天咳嗽，晚上有点发热。"
    },
    {
      "id": "seg_0002",
      "start": 3.5,
      "end": 8.1,
      "speaker": "doctor",
      "text": "建议先做血常规和胸片检查，注意休息，多喝水。"
    }
  ]
}
```

发送到接口：

```powershell
Invoke-RestMethod `
  -Uri http://localhost:63100/api/clinical_note/from_transcript `
  -Method Post `
  -ContentType "application/json" `
  -Body (Get-Content examples/clinical_note_request.json -Raw)
```

### 版本规划

- `0.1.x`：本地转写、说话人处理、确定性病历草稿。
- `0.2.x`：本地 encounter 持久化、临床人员审核定稿、Markdown/JSON 导出。
- `0.3.x`：公开发布准备、安全扫描、安全文档、demo 数据策略。
- `0.4.x`：第一方浏览器工作台，支持草稿、审核、定稿、导出。
- `0.5.x`：更丰富的导出模板和 DOCX。
- `0.6.x`：证据查看器、审计日志、质量指标。
- `0.7.x`：可选 LLM 结构化，带 schema 校验和引用证据。
- `0.8.x`：FHIR 兼容导出和集成 hook。

工作路线图见 [docs/PRODUCT_PLAN.md](docs/PRODUCT_PLAN.md)。
