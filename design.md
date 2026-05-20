# 音频切片 Web 工具设计

## 1. 背景

本工具用于处理一组固定结构的输入文件：

- 一个 `.wav` 文件：包含多段人声朗读音频，按顺序拼接。
- 一个 `.txt` 文件：每行是一句英文文本，行顺序与 `.wav` 中的朗读片段顺序一致。

目标是将原始 `.wav` 按 `.txt` 的每一行切分成独立音频文件，并按句子内容命名。例如：

- `hello nissan.wav`
- `hello nissan-1.wav`
- `hello nissan-2.wav`

同一句文本出现多次时，所有对应音频都应保留，并按出现顺序生成后缀。

当前样例目录：

```text
data/20260518/
  H19-英式005.wav
  英式005.txt
```

样例特征：

- 文本共 240 个有效文本行。注意：该文件最后一行没有 trailing newline，因此 `wc -l` 会显示 239，但按 `splitlines()`/实际任务行数应处理为 240。
- 音频约 631.73 秒。
- 音频格式为 16 kHz、mono、PCM WAV。
- 文本中存在大量重复短句，例如 `Hey Bleeker`、`Bleeker`、`Start Recording`。

由于存在重复句、近音句和短句，本项目不应依赖纯静音检测或纯 ASR 文本结果来决定切片顺序。正确策略是：**txt 控制顺序和命名，模型只负责寻找每行文本在音频中的时间边界，独立 ASR 只用于质检。**

## 2. 目标

### 2.1 产品目标

用户在 Web 页面中上传一个 `.wav` 文件和一个 `.txt` 文件，系统自动完成：

1. 校验输入文件。
2. 使用模型对齐 `.wav` 和 `.txt`。
3. 按 `.txt` 每一行切出独立 `.wav` 文件。
4. 生成切片压缩包。
5. 对每个切片做 ASR 质检。
6. 展示整体进度、每条切片的置信度/风险状态、预览播放器和报告。
7. 用户下载包含所有切片和报告的 `.zip` 文件。

### 2.2 工程目标

系统应满足：

- 内部使用优先，部署和维护成本低。
- 支持单机处理，后续可扩展为多 worker。
- 对长音频任务提供可靠进度反馈。
- 处理结果可复现、可追踪、可人工复核。
- 模型失败、边界异常、ASR 质检失败时有明确错误信息。
- 后端 pipeline 可脱离 Web UI 单独运行，方便命令行批处理和测试。

## 3. 非目标

第一版不做以下功能：

- 不做用户登录、权限系统和多人协作。
- 不做云存储上传。
- 不做在线人工拖拽修正切片边界。
- 不做数据库级永久归档。
- 不做复杂的说话人识别。
- 不做纯静音检测切分作为主流程。
- 不让 ASR 自动决定最终文本、排序或文件名。

这些可以作为后续版本扩展。

## 4. 推荐技术栈

### 4.1 后端

推荐：

- Python 3.11+
- FastAPI：HTTP API、SSE 进度流、静态产物访问。
- RQ + Redis：后台任务队列。
- ffmpeg：音频探测、裁剪、导出。
- ctc-forced-aligner：主对齐引擎。
- faster-whisper 或 WhisperX：后处理 ASR 质检。
- pandas / pydantic：报告和结构化数据。
- pytest：pipeline 单元测试和集成测试。

选择理由：

- 音频和模型生态主要在 Python。
- FastAPI 与长任务状态查询、文件上传集成简单。
- RQ 比 Celery 轻，适合内部单机工具。
- Redis/RQ 使模型处理和 Web 请求解耦，避免请求超时。
- ffmpeg 稳定、跨平台、适合批量裁剪 WAV。

### 4.2 前端

推荐：

- Vite + React + TypeScript。
- TanStack Query：任务状态轮询/缓存。
- SSE EventSource：实时进度事件。
- 原生 `<audio>`：切片预览。
- CSS Modules 或 Tailwind CSS：快速实现稳定 UI。

选择理由：

- React/Vite 轻量，适合内部工具。
- TypeScript 能约束任务状态、报告字段、API 响应。
- UI 重点是上传、进度、表格、预览，不需要复杂应用框架。

### 4.3 部署

推荐单机部署：

```text
docker compose
  api: FastAPI
  worker: RQ worker
  redis: queue/state backend
  web: Vite build static files, or served by FastAPI
  storage: local volume ./storage
```

第一版也可以本地开发运行：

```text
uvicorn app.main:app --reload
rq worker wavesplit
npm run dev
redis-server
```

## 5. 系统架构

```text
Browser
  |
  | upload wav + txt
  v
FastAPI API Server
  |
  | create job, persist files, enqueue
  v
Redis Queue
  |
  v
Worker Process
  |
  | validate -> align -> cut -> QA -> zip
  v
Local Job Storage
  |
  | progress/status/report/download
  v
Browser
```

### 5.1 模块划分

#### Web UI

负责：

- 文件上传。
- 任务创建。
- 任务进度展示。
- 结果表格展示。
- 切片音频预览。
- 质检报告查看。
- zip 下载。

不负责：

- 直接处理音频。
- 直接运行模型。
- 推断对齐边界。

#### API Server

负责：

- 接收上传文件。
- 创建 job id。
- 保存原始文件。
- 校验基础文件格式。
- 将 job 投递到后台队列。
- 暴露 job 状态、进度、报告、下载接口。
- 提供切片预览文件访问。

不负责：

- 在请求生命周期内执行模型。
- 在内存中长期保存大文件。

#### Worker

负责：

- 读取 job 输入。
- 执行完整 pipeline。
- 持续写入进度。
- 生成切片、报告和 zip。
- 记录错误和诊断信息。

#### Pipeline Core

负责：

- 文本读取和规范化。
- 音频探测。
- CTC forced alignment。
- word-level timestamp 到 line-level timestamp 的聚合。
- ffmpeg 裁剪。
- ASR 质检。
- 报告生成。

该模块应设计为纯 Python 服务层，既能被 worker 调用，也能被 CLI 和测试直接调用。

## 6. 数据目录设计

所有任务保存在本地 `storage/jobs/{job_id}`：

```text
storage/
  jobs/
    20260518-193000-8f3a/
      input/
        original.wav
        transcript.txt
      normalized/
        transcript.normalized.json
      alignment/
        raw_alignment.json
        line_alignment.json
      clips/
        Hey Bleeker.wav
        Hey Bleeker-1.wav
        Bleeker.wav
      qa/
        asr_results.json
        qa_report.csv
        qa_report.json
        qa_report.html
      logs/
        worker.log
        ffmpeg.log
      output/
        clips.zip
        manifest.csv
        manifest.json
      status.json
```

### 6.1 status.json

`status.json` 是任务状态的单一事实来源，API 和前端都从这里读取任务进展。

示例：

```json
{
  "job_id": "20260518-193000-8f3a",
  "state": "running",
  "stage": "qa",
  "progress": 0.78,
  "message": "Running ASR QA for clip 162/240",
  "created_at": "2026-05-18T19:30:00+08:00",
  "updated_at": "2026-05-18T19:38:12+08:00",
  "input": {
    "audio_filename": "H19-英式005.wav",
    "text_filename": "英式005.txt",
    "line_count": 240,
    "audio_duration_sec": 631.731
  },
  "counts": {
    "total": 240,
    "aligned": 240,
    "cut": 240,
    "qa_pass": 222,
    "qa_review": 14,
    "qa_fail": 4
  },
  "error": null
}
```

状态枚举：

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

阶段枚举：

- `upload_saved`
- `validating`
- `normalizing_text`
- `aligning`
- `building_segments`
- `cutting`
- `qa_asr`
- `qa_scoring`
- `packaging`
- `done`
- `error`

## 7. 输入校验

### 7.1 WAV 校验

上传后立即使用 `ffprobe` 检查：

- 文件可读取。
- 至少包含一个音频流。
- duration > 0。
- 格式为 WAV 或可由 ffmpeg 解码。
- 建议采样率为 16 kHz 或 44.1/48 kHz。
- 如果不是 16 kHz mono，pipeline 内部转为临时 16 kHz mono 对齐文件，但最终切片可从原始 WAV 裁剪。

若音频无法解码，任务失败并返回明确错误：

```text
Audio file cannot be decoded by ffmpeg.
```

### 7.2 TXT 校验

读取规则：

- 默认 UTF-8。
- 如果 UTF-8 失败，可尝试 UTF-8 with BOM。
- 按行读取。
- 去掉行尾换行。
- 保留原始句子用于命名。
- 规范化文本用于模型对齐。

校验规则：

- 至少 1 个非空行。
- 每个非空行规范化后至少包含一个英文 token。
- 空行默认视为错误，而不是自动跳过，避免音频和文本行错位。
- 总行数和预计时长不做硬性匹配，只做提示。

## 8. 文本规范化与行映射

系统必须同时保留两套文本：

### 8.1 original_text

来自 txt 原文，负责：

- UI 展示。
- 输出文件名。
- 报告中给人工复核。

### 8.2 normalized_text

用于模型对齐和 ASR 比较。

建议规则：

- 转小写。
- Unicode NFKC 规范化。
- 将连续空白折叠为单个空格。
- 去掉首尾空白。
- 去掉对齐模型无法稳定处理的标点。
- 保留英文单词、数字、常见缩写中的字母数字。

示例：

```text
Original: "Take a Photo!"
Normalized: "take a photo"
```

### 8.3 line manifest

生成 `transcript.normalized.json`：

```json
[
  {
    "line_index": 1,
    "original_text": "Hey Bleeker",
    "normalized_text": "hey bleeker",
    "tokens": ["hey", "bleeker"],
    "token_start_index": 0,
    "token_end_index": 2
  },
  {
    "line_index": 2,
    "original_text": "Hey Bleeker",
    "normalized_text": "hey bleeker",
    "tokens": ["hey", "bleeker"],
    "token_start_index": 2,
    "token_end_index": 4
  }
]
```

`token_start_index` 和 `token_end_index` 用于将 word-level alignment 聚合回每一行。即使句子重复，也能按出现顺序稳定映射。

## 9. 主对齐方案

### 9.1 推荐引擎

第一版使用 `ctc-forced-aligner`：

- 输入：音频 + 完整 transcript。
- 输出：word-level 或 sentence-level timestamps。
- 模型：默认 MMS forced aligner 或 Hugging Face CTC 模型。
- 粒度：优先 word-level。

选择 word-level 的原因：

- 短句和重复句较多，word-level 更容易映射和质检。
- 可以按 txt 行的 token 范围聚合为 line-level。
- 可以发现单行内部漏词、错词、异常长停顿。

### 9.2 对齐输入构造

将所有 `normalized_text` 按行顺序拼接为完整 transcript：

```text
hey bleeker
hey bleeker
bleeker
take a photo
...
```

同时保存每行 token 范围。模型输出的 word timestamps 按全局 token index 映射回行。

### 9.3 line-level 时间聚合

对于第 `i` 行：

```text
line_start = first_token.start
line_end = last_token.end
```

然后应用 padding：

```text
padded_start = max(0, line_start - pre_padding_ms)
padded_end = min(audio_duration, line_end + post_padding_ms)
```

推荐默认：

- `pre_padding_ms = 80`
- `post_padding_ms = 120`

### 9.4 相邻片段重叠处理

padding 可能导致相邻片段重叠。处理规则：

1. 如果 `segment[i].end <= segment[i+1].start`，保持不变。
2. 如果存在重叠，但原始边界之间有 gap，则按 gap 分配 padding。
3. 如果原始 speech 时间本身重叠或模型边界倒置，标记为 `boundary_overlap`，并使用 midpoint 修正：

```text
mid = (raw_end_i + raw_start_next) / 2
segment[i].end = mid
segment[i+1].start = mid
```

该片段 QA 状态至少为 `review`。

### 9.5 对齐失败处理

对齐可能失败的情况：

- 模型无法加载。
- 音频过长或内存不足。
- transcript 与音频严重不匹配。
- word 数量对不上。
- 某些 token 没有 timestamp。

失败策略：

- 不生成切片 zip。
- 保存 `raw_alignment.json` 和错误日志。
- UI 显示失败阶段和错误摘要。
- 支持用户下载诊断包。

## 10. 切片导出

### 10.1 ffmpeg 裁剪

使用 ffmpeg 从原始 WAV 裁剪，而不是从对齐用的重采样临时文件裁剪。

推荐命令形态：

```bash
ffmpeg -y \
  -ss {start} \
  -to {end} \
  -i input.wav \
  -map 0:a:0 \
  -acodec pcm_s16le \
  output.wav
```

说明：

- 输出统一为 WAV PCM，便于后续使用。
- 如需速度优先，可将 `-ss` 放在 `-i` 前；如需更精确，可放在 `-i` 后。第一版建议保守使用精确裁剪。
- 每个片段记录 ffmpeg 返回码和 stderr 摘要。

### 10.2 文件命名

命名以 `original_text` 为基础：

1. 去掉首尾空格。
2. 折叠连续空白为单个空格。
3. 移除或替换文件系统非法字符：
   - `/`
   - `\`
   - `:`
   - `*`
   - `?`
   - `"`
   - `<`
   - `>`
   - `|`
   - null byte
4. 限制文件名长度，例如 base name 最多 120 个字符。
5. 如果规范化后为空，使用 `line-{line_index}`。

重复命名规则：

```text
hello nissan.wav
hello nissan-1.wav
hello nissan-2.wav
```

注意：第一个出现的重复句不加后缀，第二个开始加 `-1`。这与用户示例一致。

建议同时在 manifest 中保存：

- `line_index`
- `original_text`
- `normalized_text`
- `output_file`
- `duplicate_index`

## 11. ASR 质检设计

### 11.1 质检原则

ASR 质检只用于判断切片是否可能有问题，不参与最终切片排序、文件名或文本替换。

原因：

- 输入 txt 是标准答案。
- ASR 对短句和品牌词可能误识别。
- 重复句场景下，ASR 无法可靠判断第几次出现。

### 11.2 推荐 ASR 引擎

第一版推荐 `faster-whisper`：

- 部署简单。
- CPU/GPU 都能跑。
- 对短英语句子表现稳定。
- 输出文本和置信相关信息。

如果需要 word-level ASR 时间戳或更强对齐，可切换到 WhisperX。

推荐默认模型：

- CPU 环境：`small.en` 或 `medium.en`。
- GPU 环境：`medium.en` 或 `large-v3`。

内部使用场景 license 不作为主要约束。

### 11.3 QA 指标

每个切片生成以下指标：

#### 文本相似度

对 ASR 文本和 `normalized_text` 做比较：

- `asr_text`
- `asr_normalized_text`
- `exact_match`
- `similarity`
- `wer`

推荐实现：

- `rapidfuzz.fuzz.ratio` 计算 similarity。
- `jiwer` 计算 WER。

阈值：

```text
pass:
  exact_match == true
  or similarity >= 92 and wer <= 0.25

review:
  similarity >= 80 and wer <= 0.50

fail:
  similarity < 80
  or wer > 0.50
```

短句特殊规则：

- 1-2 个词的短句，WER 对单词错误过于敏感，应结合 exact match 和 similarity。
- 对 `Bleeker` 这类专有词，ASR 可能误写，需要允许人工配置 alias。

Alias 示例：

```json
{
  "bleeker": ["bleaker", "bleeker.", "bleaker."]
}
```

#### 边界质量

分析切片音频波形能量：

- `leading_silence_ms`
- `trailing_silence_ms`
- `peak_dbfs`
- `rms_dbfs`
- `start_energy_flag`
- `end_energy_flag`

建议规则：

- 开头 `50ms` 内能量很高：可能切掉开头，标记 `start_maybe_clipped`。
- 结尾最后 `50ms` 能量很高：可能切掉结尾，标记 `end_maybe_clipped`。
- 首部静音 > `800ms`：标记 `leading_silence_long`。
- 尾部静音 > `1000ms`：标记 `trailing_silence_long`。

这些规则不应直接判定失败，只提升为 `review`。

#### 时长异常

记录：

- `duration_sec`
- `duration_zscore_by_text`
- `duration_ratio_to_text_median`

对重复句尤其有用。例如同一句出现 20 次时，若某个片段时长是同句 median 的 2.5 倍，应标记 review。

推荐规则：

```text
duration < 0.25s -> fail
duration > 8.0s -> review
same_text_duration_ratio > 2.5 -> review
same_text_duration_ratio < 0.4 -> review
```

#### 对齐置信度

如果 CTC 对齐引擎提供 token confidence 或 emission score，记录并聚合：

- `alignment_score_min`
- `alignment_score_mean`
- `alignment_score_p10`

如果主对齐工具不直接提供置信度，则用以下代理指标：

- token 是否全部有时间戳。
- token 时间是否单调递增。
- line duration 是否合理。
- 与前后片段是否重叠。
- ASR 相似度。

UI 中展示的 `confidence` 可以是综合分，而不是模型原始概率。

建议综合分：

```text
confidence = 100
  - text_mismatch_penalty
  - boundary_penalty
  - duration_penalty
  - overlap_penalty
```

范围限制在 0-100。

### 11.4 QA 状态

每个切片最终状态：

- `pass`：可直接使用。
- `review`：建议人工听一下。
- `fail`：高度可疑，默认在 UI 中突出显示。

状态计算优先级：

```text
if hard_error:
  fail
else if severe_text_mismatch:
  fail
else if any_review_flag:
  review
else:
  pass
```

## 12. Web UI 设计

### 12.1 页面结构

第一版使用单页应用：

```text
Header
  "Audio Splitter"

Main
  Upload Panel
  Job Progress Panel
  Results Summary
  Clip Table
  Clip Preview Drawer/Panel

Footer
  Version / model info / local storage note
```

不做营销型首页，打开即是工具界面。

### 12.2 上传面板

内容：

- WAV 文件选择/拖拽区域。
- TXT 文件选择/拖拽区域。
- 上传按钮。
- 基础说明：
  - wav 是拼接朗读音频。
  - txt 每行对应一个片段。
  - 行顺序必须与音频顺序一致。

上传前校验：

- 两个文件都已选择。
- wav 后缀正确。
- txt 后缀正确。
- 文件大小显示。

上传后：

- 禁用上传按钮。
- 显示 job id。
- 自动跳转到进度区域。

### 12.3 进度面板

进度应显示两类信息：

#### 总体进度

```text
Running QA
[####################------] 78%
Running ASR QA for clip 162/240
```

#### 阶段进度

阶段列表：

```text
1. Validate files       done
2. Normalize text       done
3. Align transcript     done
4. Cut clips            done
5. Run QA               running
6. Package zip          pending
```

每个阶段展示：

- 状态：pending/running/done/error。
- 耗时。
- 当前处理数，例如 `162/240`。
- 错误摘要。

### 12.4 结果摘要

成功后展示：

```text
240 clips generated
222 pass · 14 review · 4 fail
Duration: 631.73s
Model: ctc-forced-aligner / faster-whisper small.en
```

操作按钮：

- Download ZIP
- Download QA CSV
- Download Manifest
- Show only review/fail

### 12.5 切片表格

字段：

- 行号。
- 原句。
- 输出文件名。
- 开始时间。
- 结束时间。
- 时长。
- 综合置信度。
- QA 状态。
- ASR 文本。
- 相似度。
- WER。
- flags。
- 预览按钮。

表格功能：

- 按状态筛选：all/pass/review/fail。
- 搜索原句或文件名。
- 按置信度排序。
- 按行号排序。
- 点击行打开预览。

状态视觉：

- `pass`：低强调。
- `review`：黄色/橙色标记。
- `fail`：红色标记。

### 12.6 切片预览

用户点击某一行后显示预览面板：

```text
Line 42
Original: Take a Photo
ASR: take a photo
Status: pass
Confidence: 96

[audio player]

Start: 102.314s
End: 103.827s
Duration: 1.513s

Flags:
  none
```

预览能力：

- 播放切片音频。
- 显示对齐时间。
- 显示 ASR 文本对比。
- 显示 QA flags。
- 下载单个切片。

第一版不要求波形图。如果实现成本允许，可加静态波形或能量条。

### 12.7 失败页面

任务失败时展示：

- 失败阶段。
- 错误摘要。
- 可下载诊断日志。
- 可重新上传。

示例：

```text
Alignment failed
The aligner produced 475 word timestamps, but transcript contains 478 tokens.
Download diagnostics
```

## 13. API 设计

### 13.1 创建任务

```http
POST /api/jobs
Content-Type: multipart/form-data

audio: file
transcript: file
```

响应：

```json
{
  "job_id": "20260518-193000-8f3a",
  "state": "queued"
}
```

### 13.2 查询状态

```http
GET /api/jobs/{job_id}
```

响应：`status.json`。

### 13.3 进度事件流

```http
GET /api/jobs/{job_id}/events
```

SSE event 示例：

```text
event: progress
data: {"stage":"qa_asr","progress":0.78,"message":"Running ASR QA for clip 162/240"}
```

前端应在 SSE 断开时回退到轮询 `GET /api/jobs/{job_id}`。

### 13.4 获取结果报告

```http
GET /api/jobs/{job_id}/report
```

返回：

```json
{
  "job_id": "20260518-193000-8f3a",
  "summary": {
    "total": 240,
    "pass": 221,
    "review": 14,
    "fail": 4
  },
  "clips": [
    {
      "line_index": 1,
      "original_text": "Hey Bleeker",
      "output_file": "Hey Bleeker.wav",
      "start": 1.497,
      "end": 2.413,
      "duration": 0.916,
      "confidence": 94,
      "status": "pass",
      "asr_text": "hey bleeker",
      "similarity": 100,
      "wer": 0,
      "flags": []
    }
  ]
}
```

### 13.5 下载 zip

```http
GET /api/jobs/{job_id}/download
```

返回 `clips.zip`。

### 13.6 下载单个切片

```http
GET /api/jobs/{job_id}/clips/{clip_id}
```

`clip_id` 建议使用 manifest 中的稳定 id，而不是直接暴露文件路径。

### 13.7 下载报告文件

```http
GET /api/jobs/{job_id}/manifest.csv
GET /api/jobs/{job_id}/qa_report.csv
GET /api/jobs/{job_id}/diagnostics.zip
```

## 14. 输出 zip 内容

下载的 zip 应包含：

```text
clips/
  Hey Bleeker.wav
  Hey Bleeker-1.wav
  Bleeker.wav
manifest.csv
manifest.json
qa_report.csv
qa_report.json
README.txt
```

`README.txt` 说明：

- 输入文件名。
- 处理时间。
- 模型名称。
- padding 配置。
- QA 状态含义。

## 15. Manifest 字段

`manifest.csv`：

```text
clip_id,
line_index,
original_text,
normalized_text,
output_file,
start_sec,
end_sec,
duration_sec,
duplicate_index,
alignment_score_mean,
confidence,
status,
asr_text,
asr_normalized_text,
similarity,
wer,
leading_silence_ms,
trailing_silence_ms,
flags
```

`clip_id` 示例：

```text
clip-000001
clip-000002
```

## 16. 配置设计

使用 `config.yaml` 或环境变量配置：

```yaml
storage_dir: ./storage
max_upload_mb: 1024

alignment:
  engine: ctc_forced_aligner
  model: default
  language: en
  pre_padding_ms: 80
  post_padding_ms: 120

qa:
  enabled: true
  asr_engine: faster_whisper
  asr_model: small.en
  similarity_pass: 92
  similarity_review: 80
  wer_pass: 0.25
  wer_review: 0.50
  leading_silence_review_ms: 800
  trailing_silence_review_ms: 1000

worker:
  queue_name: wavesplit
  concurrency: 1
```

第一版建议 worker concurrency 默认 1，避免多个模型任务同时抢 CPU/GPU 内存。

## 17. 任务进度计算

总进度建议按阶段加权：

```text
validating:        5%
normalizing_text:  5%
aligning:         35%
building_segments: 5%
cutting:          20%
qa_asr:           20%
packaging:        10%
```

如果某阶段内部有明确计数，例如切片和 QA，则按 `current / total` 更新阶段内进度。

示例：

```text
progress = base_progress + stage_weight * current / total
```

## 18. 错误处理

### 18.1 用户可修复错误

- 未上传 wav。
- 未上传 txt。
- txt 为空。
- txt 含空行。
- wav 无法解码。
- 文件超过大小限制。

UI 应直接提示如何修复。

### 18.2 数据不匹配错误

- transcript token 数与模型输出不一致。
- 对齐边界非单调。
- 大量行无法生成时间戳。
- 音频明显短于文本预期。

UI 显示：

- 失败阶段。
- 错误摘要。
- 下载诊断包。

### 18.3 系统错误

- 模型加载失败。
- ffmpeg 不存在。
- Redis 不可用。
- 磁盘空间不足。

启动时应做健康检查：

- `ffmpeg -version`
- `ffprobe -version`
- Redis ping。
- 模型可加载或模型路径存在。

## 19. 性能与资源

### 19.1 预期规模

第一版目标：

- 单个 WAV：最长 30-60 分钟。
- 单个 txt：最多 2000 行。
- 单任务处理。
- 内部少量用户。

### 19.2 优化策略

- 对齐阶段使用完整音频一次处理，避免逐句 ASR。
- 切片阶段可并发 ffmpeg，但限制并发数。
- QA 阶段可按切片逐个 ASR，便于进度显示和失败恢复。
- 模型加载在 worker 生命周期内复用，避免每个 job 重复加载。
- 生成 zip 时流式写入，避免一次性读入所有音频。

### 19.3 GPU 支持

配置项：

```yaml
device: auto
compute_type: int8
```

CPU 模式优先可运行，GPU 模式提升速度。

## 20. 安全与清理

虽然是内部工具，仍需做基础防护：

- 不信任上传文件名，服务端生成内部文件名。
- API 不直接拼接用户传入路径。
- 下载单个切片使用 `clip_id` 查 manifest。
- 限制上传大小。
- 限制 job 目录访问范围。
- 定期清理旧 job，例如保留 7 天。
- zip 内文件名必须经过安全清洗。

清理策略：

```text
每天清理 updated_at 超过 7 天的 succeeded/failed job。
running/queued job 不自动删除。
```

## 21. 测试计划

### 21.1 单元测试

覆盖：

- 文本读取。
- 文本规范化。
- 空行检测。
- token range 映射。
- 文件名安全清洗。
- 重复文件名后缀。
- line-level 时间聚合。
- padding 和相邻重叠处理。
- QA 状态计算。
- manifest 生成。

### 21.2 集成测试

使用小型 fixture：

```text
tests/fixtures/
  sample.wav
  sample.txt
```

覆盖：

- 上传任务。
- worker 执行。
- 生成 clips。
- 生成 zip。
- 生成 manifest。
- API 返回报告。

### 21.3 模型冒烟测试

至少保留一个 10-30 秒的小样例，CI 或本地测试可选运行：

```text
pytest -m model
```

普通单元测试不应强依赖大模型下载。

### 21.4 人工验收

使用 `data/20260518` 样例验收：

- 成功生成 240 个切片。
- zip 可下载并解压。
- 文件名重复后缀符合规则。
- QA report 有 pass/review/fail 分类。
- Web 页面可预览任意切片。
- review/fail 过滤可用。

## 22. 实施里程碑

### Milestone 1: Pipeline CLI

目标：

- 命令行输入 wav/txt。
- 生成切片、manifest 和 zip。
- 完成基本文本规范化和命名规则。

验收：

```bash
wavesplit process data/20260518/H19-英式005.wav data/20260518/英式005.txt --out output/20260518
```

### Milestone 2: 模型对齐接入

目标：

- 接入 `ctc-forced-aligner`。
- 生成 word-level raw alignment。
- 聚合 line-level alignment。
- 处理 padding 和重叠。

验收：

- 样例音频可生成 240 条 line alignment。
- manifest 中每条都有 start/end/duration。

### Milestone 3: ASR QA

目标：

- 接入 faster-whisper 或 WhisperX。
- 对每个切片识别。
- 生成 similarity、WER、flags、confidence。

验收：

- QA report 可读。
- 明显错误片段进入 review/fail。

### Milestone 4: Web API + Worker

目标：

- FastAPI 上传。
- RQ worker 执行 pipeline。
- job 状态和进度可查询。
- 支持 zip 下载。

验收：

- 浏览器上传后任务后台运行。
- 页面刷新不丢任务状态。

### Milestone 5: Web UI

目标：

- 上传界面。
- 进度条。
- 结果摘要。
- 切片表格。
- 音频预览。
- 下载按钮。

验收：

- 非开发用户可以完成完整流程。

## 23. 默认参数建议

```text
alignment.pre_padding_ms = 80
alignment.post_padding_ms = 120
qa.asr_model = small.en
qa.similarity_pass = 92
qa.similarity_review = 80
qa.wer_pass = 0.25
qa.wer_review = 0.50
worker.concurrency = 1
job_retention_days = 7
```

## 24. 后续可选增强

可在第一版稳定后增加：

- 波形图和边界可视化。
- 人工调整 start/end 后重新导出单个切片。
- 批量上传多个 wav/txt 文件对。
- 断点恢复。
- GPU worker 池。
- 任务历史列表。
- alias 管理界面。
- 对 review/fail 片段生成单独压缩包。
- 多种输出命名模板。
- 支持 CSV/Excel transcript。

## 25. 关键设计决策

1. **使用模型 forced alignment，而不是静音切分。**
   重复短句场景下，静音切分无法稳定保证文本行和音频片段一一对应。

2. **使用 txt 作为标准答案。**
   文件名、顺序、切片数量全部由 txt 决定。

3. **ASR 只做质检。**
   Whisper/faster-whisper 结果不覆盖原始文本，避免短句误识别破坏输出。

4. **word-level 对齐后聚合到 line-level。**
   这样比直接 sentence-level 更容易诊断和修正异常。

5. **Web 请求和模型任务解耦。**
   长时间模型处理必须在 worker 中运行，API 只负责状态和产物访问。

6. **所有产物落盘。**
   内部工具优先保证可追踪、可复核、可下载诊断信息。

## 26. 待确认问题

以下问题不阻塞第一版，但实现前建议确认：

1. 是否固定只处理英语，还是未来需要多语言。
2. 内部部署机器是否有 NVIDIA GPU。
3. 单个上传文件最大可能多大。
4. QA 阈值是否需要按不同数据集配置。
5. review/fail 的切片是否仍包含在最终 zip 中。当前设计默认全部包含，并在报告中标记状态。
6. 是否需要在 zip 中额外包含 `review/` 或 `fail/` 分类副本。当前设计默认不复制，避免重复文件。

## 27. GPU 机器继续实现与验收备注

当前 CPU 机器已经完成基础实现和本地验证，但 ASR QA 阶段受硬件限制明显：

- 已确认 `ctc-forced-aligner` ONNX 模型可加载并能为 `data/20260518` 样例生成 240 条 line-level timestamp。
- 已确认样例可以生成 240 个 clip、manifest、QA report 和 zip。
- CPU 上使用 `faster-whisper tiny.en` 做 QA 时，短句和专有词容易误识别，例如 `Bleeker` 被识别为空、`Pleeker`、`Please` 等；这类结果不能作为最终质量结论。
- CPU 上尝试更大的 Whisper 模型会非常慢，因此下一步应迁移到 GPU 机器继续跑完整验收。

### 27.1 迁移到 GPU 机器后先做的事

在新机器上进入 repo 后执行：

```bash
python3 -m pip install -r requirements.txt
cd web && npm install && npm run build && cd ..
python3 -m pytest -q
```

确认外部依赖：

```bash
ffmpeg -version
ffprobe -version
python3 - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
```

ONNX forced alignment 模型应放在：

```text
storage/models/ctc_forced_aligner.onnx
```

如果 transfer repo 时没有带上 `storage/models/`，需要在新机器重新放置该文件，或修改 `config.yaml` 中的 `alignment.model` 指向实际路径。

### 27.2 GPU 上建议的配置

优先使用模型对齐和更强 ASR QA：

```yaml
alignment:
  engine: auto
  model: storage/models/ctc_forced_aligner.onnx
  ctc_batch_size: 4

qa:
  enabled: true
  asr_engine: faster_whisper
  asr_model: medium.en
  device: cuda
  compute_type: float16
```

如果显存足够，可把 `qa.asr_model` 改为 `large-v3`；如果显存不足，退回 `small.en` 或 `medium.en` + `compute_type: int8_float16`。

### 27.3 GPU 机器上的下一步命令

先用样例重新跑一份新 job，不要复用 CPU 机器上已有的 QA 结果：

```bash
rm -rf storage/jobs/20260518-gpu-acceptance
python3 -m wavesplit process \
  data/20260518/H19-英式005.wav \
  data/20260518/英式005.txt \
  --out storage/jobs/20260518-gpu-acceptance \
  --asr-model medium.en
```

如果通过 Web 路径验收：

```bash
uvicorn wavesplit.api:app --host 0.0.0.0 --port 8000
```

然后在浏览器上传同一组 `wav/txt`，确认后台任务完成后可下载 zip、manifest 和 QA CSV。

### 27.4 GPU 验收必须检查

不要只看 `status.json` 是否 `succeeded`，必须检查实际产物：

1. `storage/jobs/20260518-gpu-acceptance/status.json`
   - `state == "succeeded"`
   - `counts.total == 240`
   - `counts.aligned == 240`
   - `counts.cut == 240`
2. `alignment/raw_alignment.json`
   - `engine == "ctc_forced_aligner"`
   - `line_timestamp_count == 240`
   - `word_timestamp_count` 应等于 transcript token 数，当前样例为 578。
3. `clips/`
   - 实际 `.wav` 文件数为 240。
   - 重复命名符合规则，例如 `Hey Bleeker.wav`、`Hey Bleeker-1.wav`、`Hey Bleeker-19.wav`。
4. `output/clips.zip`
   - zip 中包含 `clips/` 下 240 个 wav。
   - zip 中包含 `manifest.csv`、`manifest.json`、`qa_report.csv`、`qa_report.json`、`README.txt`。
5. `qa/qa_report.json`
   - 有 pass/review/fail 分类。
   - 对 `fail` 和高风险 `review` 行抽样试听，区分是真边界问题还是 ASR 对短句/专有词的误识别。
6. Web UI
   - 可上传任务。
   - 进度会更新。
   - 表格筛选 all/pass/review/fail 可用。
   - 点击行可播放 preview。
   - zip、manifest、QA CSV 下载可用。

### 27.5 如果 GPU QA 仍有大量 fail

处理顺序：

1. 先抽样试听 `fail` 行对应的 clip，不要让 ASR 自动改文本、顺序或文件名。
2. 如果音频切片正确但 ASR 文本是专有词或短句误识别，补充 `qa.aliases`，并将 1-2 个词短句的严重 mismatch 降级为 `review`，不要直接判定切片失败。
3. 如果同一文本段出现异常长 clip，例如一个 `Take a Photo` clip 包含多次朗读，优先检查 `alignment/line_alignment.json` 的 raw start/end，再增加 CTC span 修复逻辑或局部 energy refinement。
4. 修复后重新跑完整样例 job，直到 240 个切片、zip、manifest、QA report 和 Web preview 都通过验收。
