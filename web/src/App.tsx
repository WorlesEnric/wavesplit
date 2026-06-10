import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileAudio,
  FileText,
  ListFilter,
  LockKeyhole,
  Loader2,
  LogOut,
  Pause,
  Play,
  Search,
  UploadCloud,
  XCircle
} from "lucide-react";
import { artifactUrl, clipUrl, createBatchTimestampZip, createJob, createTimestampTxt, getAuthSession, getJob, getReport, login, logout } from "./api";
import type { AuthSession, ClipRecord, JobReport, JobStatus, QAStatus } from "./types";

const stages = [
  ["validating", "校验"],
  ["normalizing_text", "规范文本"],
  ["aligning", "对齐"],
  ["building_segments", "分段"],
  ["cutting", "切分"],
  ["qa_asr", "ASR"],
  ["qa_scoring", "评分"],
  ["packaging", "打包"]
];

const qaStatusLabels: Record<QAStatus, string> = {
  pass: "通过",
  review: "复核",
  fail: "失败",
  missing_audio: "无音频"
};

const filterLabels: Record<"all" | QAStatus, string> = {
  all: "全部",
  ...qaStatusLabels
};

const statusMessages: Record<string, string> = {
  "Upload saved": "上传已保存",
  "Validating input files": "正在校验输入文件",
  "Input files validated": "输入文件校验完成",
  "Normalizing transcript": "正在规范化文本",
  "Transcript normalized": "文本规范化完成",
  "Aligning transcript to audio": "正在将文本对齐到音频",
  "Building output segment manifest": "正在生成输出片段清单",
  "Output segment manifest built": "输出片段清单已生成",
  "Starting ASR QA": "正在启动 ASR 质检",
  "QA scoring complete": "质检评分完成",
  "Packaging zip and reports": "正在打包 ZIP 和报告",
  "Packaging complete": "打包完成",
  Done: "完成"
};

function formatSeconds(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${value.toFixed(3)} 秒`;
}

function formatStatusMessage(message: string | null | undefined) {
  if (!message) return "暂无任务";
  const aligned = message.match(/^Aligned (\d+)\/(\d+) lines$/);
  if (aligned) return `已对齐 ${aligned[1]}/${aligned[2]} 行`;
  const cutting = message.match(/^Cutting clip (\d+)\/(\d+)$/);
  if (cutting) return `正在切分片段 ${cutting[1]}/${cutting[2]}`;
  const asr = message.match(/^Running ASR QA for clip (\d+)\/(\d+)$/);
  if (asr) return `正在质检片段 ${asr[1]}/${asr[2]}`;
  const generated = message.match(/^Generated (\d+) clips$/);
  if (generated) return `已生成 ${generated[1]} 个片段`;
  const unexpected = message.match(/^Unexpected error: (.*)$/);
  if (unexpected) return `意外错误：${unexpected[1]}`;
  return statusMessages[message] ?? message;
}

interface TimestampSegment {
  index: number;
  line: string;
  start_sec: number;
  end_sec: number;
}

function parseTimestampSegments(text: string): TimestampSegment[] {
  return text
    .split(/\r?\n/)
    .map((line, index) => {
      const [start, end] = line.trim().split(/\s+/).map(Number);
      if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
      return {
        index: index + 1,
        line: `${start.toFixed(3)} ${end.toFixed(3)}`,
        start_sec: start,
        end_sec: end
      };
    })
    .filter((segment): segment is TimestampSegment => Boolean(segment));
}

function referenceFromFilename(filename: string) {
  return filename.replace(/\.[^/.]+$/, "");
}

function statusIcon(status: string) {
  if (status === "pass" || status === "succeeded") return <CheckCircle2 aria-hidden="true" />;
  if (status === "fail" || status === "failed") return <XCircle aria-hidden="true" />;
  return <AlertTriangle aria-hidden="true" />;
}

function SignInPage({ onSignIn }: { onSignIn: (session: AuthSession) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      onSignIn(await login(username, password));
    } catch (err) {
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="signInShell">
      <form className="signInPanel" onSubmit={submit}>
        <div className="signInMark">
          <LockKeyhole aria-hidden="true" />
          <div>
            <h1>WaveSplit</h1>
            <p>登录</p>
          </div>
        </div>
        <label className="fieldStack">
          <span>用户名</span>
          <input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} />
        </label>
        <label className="fieldStack">
          <span>密码</span>
          <input autoComplete="current-password" type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
        </label>
        {error && <p className="errorText">{error}</p>}
        <button className="primaryButton" type="submit" disabled={!username || !password || submitting}>
          {submitting ? <Loader2 aria-hidden="true" /> : <LockKeyhole aria-hidden="true" />}
          <span>{submitting ? "正在登录" : "登录"}</span>
        </button>
      </form>
    </div>
  );
}

export default function App() {
  const [auth, setAuth] = useState<AuthSession | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string>(() => localStorage.getItem("wavesplit:lastJob") ?? "");
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [report, setReport] = useState<JobReport | null>(null);
  const [selected, setSelected] = useState<ClipRecord | null>(null);
  const [filter, setFilter] = useState<"all" | QAStatus>("all");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<"line" | "confidence">("line");
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [timestampAudio, setTimestampAudio] = useState<File | null>(null);
  const [timestampReference, setTimestampReference] = useState("");
  const [timestamping, setTimestamping] = useState(false);
  const [timestampError, setTimestampError] = useState("");
  const [timestampDownloadUrl, setTimestampDownloadUrl] = useState("");
  const [timestampFilename, setTimestampFilename] = useState("timestamps.txt");
  const [timestampSegments, setTimestampSegments] = useState<TimestampSegment[]>([]);
  const [timestampCount, setTimestampCount] = useState<number | null>(null);
  const [timestampAudioUrl, setTimestampAudioUrl] = useState("");
  const [selectedTimestampIndex, setSelectedTimestampIndex] = useState<number | null>(null);
  const [timestampPlaying, setTimestampPlaying] = useState(false);
  const [batchTimestampFiles, setBatchTimestampFiles] = useState<File[]>([]);
  const [batchTimestamping, setBatchTimestamping] = useState(false);
  const [batchTimestampError, setBatchTimestampError] = useState("");
  const [batchTimestampDownloadUrl, setBatchTimestampDownloadUrl] = useState("");
  const [batchTimestampFilename, setBatchTimestampFilename] = useState("timestamps.zip");
  const [batchTimestampCount, setBatchTimestampCount] = useState<number | null>(null);
  const timestampAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    getAuthSession().then(setAuth).catch(() => setAuth({ authenticated: false, username: null }));
  }, []);

  useEffect(() => {
    return () => {
      if (timestampDownloadUrl) URL.revokeObjectURL(timestampDownloadUrl);
      if (timestampAudioUrl) URL.revokeObjectURL(timestampAudioUrl);
      if (batchTimestampDownloadUrl) URL.revokeObjectURL(batchTimestampDownloadUrl);
    };
  }, [timestampDownloadUrl, timestampAudioUrl, batchTimestampDownloadUrl]);

  useEffect(() => {
    if (!jobId || !auth?.authenticated) return;
    localStorage.setItem("wavesplit:lastJob", jobId);
    let closed = false;
    const eventSource = new EventSource(`/api/jobs/${jobId}/events`);
    const poll = window.setInterval(async () => {
      if (closed) return;
      try {
        const next = await getJob(jobId);
        setStatus(next);
        if (next.state === "succeeded") setReport(await getReport(jobId));
      } catch {
        // SSE is the primary path. Polling is only a quiet fallback.
      }
    }, 3000);

    eventSource.addEventListener("progress", async (event) => {
      const next = JSON.parse((event as MessageEvent).data) as JobStatus;
      setStatus(next);
      if (next.state === "succeeded") {
        setReport(await getReport(jobId));
        eventSource.close();
      }
    });
    eventSource.onerror = () => eventSource.close();
    getJob(jobId)
      .then(async (next) => {
        setStatus(next);
        if (next.state === "succeeded") setReport(await getReport(jobId));
      })
      .catch(() => undefined);
    return () => {
      closed = true;
      eventSource.close();
      window.clearInterval(poll);
    };
  }, [auth?.authenticated, jobId]);

  const filteredClips = useMemo(() => {
    const clips = report?.clips ?? [];
    const needle = query.trim().toLowerCase();
    return clips
      .filter((clip) => filter === "all" || clip.status === filter)
      .filter((clip) => {
        if (!needle) return true;
        return (
          clip.original_text.toLowerCase().includes(needle) ||
          (clip.output_file ?? "").toLowerCase().includes(needle) ||
          clip.asr_text.toLowerCase().includes(needle)
        );
      })
      .sort((a, b) => (sort === "line" ? a.line_index - b.line_index : a.confidence - b.confidence));
  }, [filter, query, report?.clips, sort]);

  const selectedTimestampSegment =
    selectedTimestampIndex === null ? null : timestampSegments[selectedTimestampIndex] ?? null;

  function clearTimestampPreview() {
    setTimestampDownloadUrl("");
    setTimestampAudioUrl("");
    setTimestampSegments([]);
    setTimestampCount(null);
    setSelectedTimestampIndex(null);
    setTimestampPlaying(false);
  }

  function clearBatchTimestampResult() {
    setBatchTimestampDownloadUrl("");
    setBatchTimestampCount(null);
  }

  async function playTimestampSegment(segment: TimestampSegment, index: number) {
    const player = timestampAudioRef.current;
    if (!player || !timestampAudioUrl) return;
    setSelectedTimestampIndex(index);
    player.currentTime = segment.start_sec;
    try {
      await player.play();
      setTimestampPlaying(true);
    } catch {
      setTimestampPlaying(false);
    }
  }

  function pauseTimestampSegment() {
    timestampAudioRef.current?.pause();
    setTimestampPlaying(false);
  }

  function handleTimestampTimeUpdate() {
    const player = timestampAudioRef.current;
    if (!player || !selectedTimestampSegment) return;
    if (player.currentTime >= selectedTimestampSegment.end_sec) {
      player.pause();
      player.currentTime = selectedTimestampSegment.end_sec;
      setTimestampPlaying(false);
    }
  }

  function handleTimestampPlay() {
    const player = timestampAudioRef.current;
    if (!player || !selectedTimestampSegment) return;
    if (player.currentTime < selectedTimestampSegment.start_sec || player.currentTime >= selectedTimestampSegment.end_sec) {
      player.currentTime = selectedTimestampSegment.start_sec;
    }
    setTimestampPlaying(true);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!audio || !transcript) return;
    setUploading(true);
    setUploadProgress(0);
    setError("");
    setReport(null);
    setSelected(null);
    try {
      const created = await createJob(audio, transcript, setUploadProgress);
      setJobId(created.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }

  async function submitTimestamps(event: FormEvent) {
    event.preventDefault();
    if (!timestampAudio || !timestampReference.trim()) return;
    setTimestamping(true);
    setTimestampError("");
    clearTimestampPreview();
    try {
      const result = await createTimestampTxt(timestampAudio, timestampReference);
      const segments = parseTimestampSegments(result.text);
      setTimestampFilename(result.filename);
      setTimestampSegments(segments);
      setTimestampCount(result.segmentCount || segments.length);
      setSelectedTimestampIndex(segments.length ? 0 : null);
      setTimestampDownloadUrl(URL.createObjectURL(result.blob));
      setTimestampAudioUrl(URL.createObjectURL(timestampAudio));
    } catch (err) {
      setTimestampError(err instanceof Error ? err.message : "生成时间戳失败");
    } finally {
      setTimestamping(false);
    }
  }

  async function submitBatchTimestamps(event: FormEvent) {
    event.preventDefault();
    if (!batchTimestampFiles.length) return;
    setBatchTimestamping(true);
    setBatchTimestampError("");
    clearBatchTimestampResult();
    try {
      const result = await createBatchTimestampZip(batchTimestampFiles);
      setBatchTimestampFilename(result.filename);
      setBatchTimestampCount(result.fileCount || batchTimestampFiles.length);
      setBatchTimestampDownloadUrl(URL.createObjectURL(result.blob));
    } catch (err) {
      setBatchTimestampError(err instanceof Error ? err.message : "批量生成时间戳失败");
    } finally {
      setBatchTimestamping(false);
    }
  }

  function chooseFile(kind: "audio" | "transcript", event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (kind === "audio") setAudio(file);
    else setTranscript(file);
  }

  function chooseTimestampFile(event: ChangeEvent<HTMLInputElement>) {
    setTimestampAudio(event.target.files?.[0] ?? null);
    setTimestampError("");
    clearTimestampPreview();
  }

  function chooseBatchTimestampFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).filter((file) => file.name.toLowerCase().endsWith(".wav"));
    setBatchTimestampFiles(files);
    setBatchTimestampError("");
    clearBatchTimestampResult();
  }

  async function signOut() {
    await logout().catch(() => undefined);
    setAuth({ authenticated: false, username: null });
    setStatus(null);
    setReport(null);
    setSelected(null);
  }

  if (auth === null) {
    return (
      <div className="signInShell">
        <div className="signInPanel">
          <div className="signInMark">
            <Loader2 aria-hidden="true" />
            <div>
              <h1>WaveSplit</h1>
              <p>正在加载</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!auth.authenticated) {
    return <SignInPage onSignIn={setAuth} />;
  }

  return (
    <div className="appShell">
      <header className="topbar">
        <div>
          <h1>WaveSplit</h1>
          <p>按文本顺序切分 WAV 音频</p>
        </div>
        <div className="topbarActions">
          {status && (
            <div className={`jobPill state-${status.state}`}>
              {status.state === "running" || status.state === "queued" ? <Loader2 aria-hidden="true" /> : statusIcon(status.state)}
              <span>{status.job_id}</span>
            </div>
          )}
          <button className="iconButton" onClick={signOut} title="退出登录">
            <LogOut aria-hidden="true" />
            <span>{auth.username}</span>
          </button>
          </div>
      </header>

      <main className="workspace">
        <section className="panel uploadPanel">
          <form onSubmit={submit}>
            <div className="uploadGrid">
              <label className="fileDrop">
                <FileAudio aria-hidden="true" />
                <span>WAV 音频</span>
                <strong>{audio?.name ?? "选择文件"}</strong>
                <input type="file" accept=".wav,audio/wav" onChange={(event) => chooseFile("audio", event)} />
              </label>
              <label className="fileDrop">
                <FileText aria-hidden="true" />
                <span>TXT 文本</span>
                <strong>{transcript?.name ?? "选择文件"}</strong>
                <input type="file" accept=".txt,text/plain" onChange={(event) => chooseFile("transcript", event)} />
              </label>
            </div>
            <div className="uploadActions">
              <button className="primaryButton" type="submit" disabled={!audio || !transcript || uploading}>
                {uploading ? <Loader2 aria-hidden="true" /> : <UploadCloud aria-hidden="true" />}
                <span>{uploading ? `上传中 ${Math.round(uploadProgress * 100)}%` : "开始任务"}</span>
              </button>
              {error && <p className="errorText">{error}</p>}
            </div>
          </form>
        </section>

        <section className="panel timestampPanel">
          <form onSubmit={submitTimestamps}>
            <div className="panelHeader">
              <div>
                <h2>语音时间戳</h2>
                <p>WAV 到 TXT</p>
              </div>
              {timestampCount !== null && <strong>{timestampCount} 段</strong>}
            </div>
            <div className="timestampGrid">
              <label className="fileDrop">
                <FileAudio aria-hidden="true" />
                <span>WAV 音频</span>
                <strong>{timestampAudio?.name ?? "选择文件"}</strong>
                <input type="file" accept=".wav,audio/wav" onChange={chooseTimestampFile} />
              </label>
              <label className="timestampTextBox">
                <span>参考文本</span>
                <textarea value={timestampReference} onChange={(event) => setTimestampReference(event.target.value)} />
              </label>
              <div className="timestampPreviewBox">
                <span>TXT 预览</span>
                {timestampSegments.length ? (
                  <div className="timestampLineList">
                    {timestampSegments.map((segment, index) => (
                      <button
                        className={`timestampLine ${selectedTimestampIndex === index ? "selected" : ""}`}
                        key={`${segment.line}-${segment.index}`}
                        onClick={() => playTimestampSegment(segment, index)}
                        title={`试听第 ${segment.index} 段`}
                        type="button"
                      >
                        {timestampPlaying && selectedTimestampIndex === index ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
                        <code>{segment.line}</code>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="emptyTimestamp">暂无结果</div>
                )}
              </div>
            </div>
            {timestampAudioUrl && selectedTimestampSegment && (
              <div className="timestampPlayerWindow">
                <div className="previewTitle">
                  <strong>第 {selectedTimestampSegment.index} 段</strong>
                  <code>{selectedTimestampSegment.line}</code>
                </div>
                <audio
                  controls
                  onPause={() => setTimestampPlaying(false)}
                  onPlay={handleTimestampPlay}
                  onTimeUpdate={handleTimestampTimeUpdate}
                  ref={timestampAudioRef}
                  src={timestampAudioUrl}
                />
                <div className="timestampPlayerActions">
                  <button className="iconButton" onClick={() => playTimestampSegment(selectedTimestampSegment, selectedTimestampIndex ?? 0)} type="button">
                    <Play aria-hidden="true" />
                    <span>试听</span>
                  </button>
                  <button className="iconButton" onClick={pauseTimestampSegment} type="button">
                    <Pause aria-hidden="true" />
                    <span>暂停</span>
                  </button>
                </div>
              </div>
            )}
            <div className="uploadActions">
              <button className="primaryButton" type="submit" disabled={!timestampAudio || !timestampReference.trim() || timestamping}>
                {timestamping ? <Loader2 aria-hidden="true" /> : <UploadCloud aria-hidden="true" />}
                <span>{timestamping ? "处理中" : "生成 TXT"}</span>
              </button>
              {timestampDownloadUrl && (
                <a className="iconButton" href={timestampDownloadUrl} download={timestampFilename}>
                  <Download aria-hidden="true" />
                  <span>下载 TXT</span>
                </a>
              )}
              {timestampError && <p className="errorText">{timestampError}</p>}
            </div>
          </form>
        </section>

        <section className="panel batchTimestampPanel">
          <form onSubmit={submitBatchTimestamps}>
            <div className="panelHeader">
              <div>
                <h2>批量时间戳</h2>
                <p>文件名作为参考文本</p>
              </div>
              {batchTimestampCount !== null && <strong>{batchTimestampCount} 个</strong>}
            </div>
            <div className="batchTimestampGrid">
              <label className="fileDrop">
                <FileAudio aria-hidden="true" />
                <span>WAV 音频</span>
                <strong>{batchTimestampFiles.length ? `${batchTimestampFiles.length} 个文件` : "选择文件"}</strong>
                <input multiple type="file" accept=".wav,audio/wav" onChange={chooseBatchTimestampFiles} />
              </label>
              <div className="batchFileList">
                <span>待处理</span>
                {batchTimestampFiles.length ? (
                  <div>
                    {batchTimestampFiles.map((file) => (
                      <div className="batchFileItem" key={`${file.name}-${file.size}`}>
                        <FileText aria-hidden="true" />
                        <div>
                          <strong>{file.name}</strong>
                          <small>{referenceFromFilename(file.name)}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="emptyTimestamp">暂无文件</div>
                )}
              </div>
            </div>
            <div className="uploadActions">
              <button className="primaryButton" type="submit" disabled={!batchTimestampFiles.length || batchTimestamping}>
                {batchTimestamping ? <Loader2 aria-hidden="true" /> : <UploadCloud aria-hidden="true" />}
                <span>{batchTimestamping ? "处理中" : "生成 ZIP"}</span>
              </button>
              {batchTimestampDownloadUrl && (
                <a className="iconButton" href={batchTimestampDownloadUrl} download={batchTimestampFilename}>
                  <Download aria-hidden="true" />
                  <span>下载 ZIP</span>
                </a>
              )}
              {batchTimestampError && <p className="errorText">{batchTimestampError}</p>}
            </div>
          </form>
        </section>

        <section className="panel progressPanel">
          <div className="panelHeader">
            <div>
              <h2>进度</h2>
              <p>{formatStatusMessage(status?.message)}</p>
            </div>
            <strong>{Math.round((status?.progress ?? 0) * 100)}%</strong>
          </div>
          <div className="progressTrack">
            <div style={{ width: `${Math.round((status?.progress ?? 0) * 100)}%` }} />
          </div>
          <div className="stageGrid">
            {stages.map(([key, label]) => {
              const active = status?.stage === key;
              const done = status ? stages.findIndex(([name]) => name === key) < stages.findIndex(([name]) => name === status.stage) : false;
              return (
                <div className={`stageItem ${active ? "active" : ""} ${done ? "done" : ""}`} key={key}>
                  <span>{label}</span>
                </div>
              );
            })}
          </div>
          {status?.error && (
            <div className="failureActions">
              <p className="errorText">{formatStatusMessage(status.error)}</p>
              <a className="iconButton" href={artifactUrl(status.job_id, "diagnostics.zip")}>
                <Download aria-hidden="true" />
                <span>诊断包</span>
              </a>
            </div>
          )}
        </section>

        {report && status && (
          <>
            <section className="summaryGrid">
              <div className="summaryItem">
                <span>总数</span>
                <strong>{report.summary.total}</strong>
              </div>
              <div className="summaryItem pass">
                <span>通过</span>
                <strong>{report.summary.pass}</strong>
              </div>
              <div className="summaryItem review">
                <span>复核</span>
                <strong>{report.summary.review}</strong>
              </div>
              <div className="summaryItem fail">
                <span>失败</span>
                <strong>{report.summary.fail}</strong>
              </div>
              <div className="summaryItem missing_audio">
                <span>无音频</span>
                <strong>{report.summary.missing_audio ?? 0}</strong>
              </div>
              <div className="summaryItem wide">
                <span>时长</span>
                <strong>{formatSeconds(report.summary.duration_sec)}</strong>
              </div>
            </section>

            <section className="toolbar">
              <div className="searchBox">
                <Search aria-hidden="true" />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索文本、文件名或 ASR" />
              </div>
              <div className="segmented" aria-label="筛选片段">
                {(["all", "pass", "review", "fail", "missing_audio"] as const).map((item) => (
                  <button key={item} className={filter === item ? "selected" : ""} onClick={() => setFilter(item)}>
                    {filterLabels[item]}
                  </button>
                ))}
              </div>
              <button className="iconButton" onClick={() => setSort(sort === "line" ? "confidence" : "line")} title="切换排序">
                <ListFilter aria-hidden="true" />
                <span>{sort === "line" ? "行号" : "置信度"}</span>
              </button>
              <a className="iconButton" href={artifactUrl(status.job_id, "download")}>
                <Download aria-hidden="true" />
                <span>ZIP</span>
              </a>
              <a className="iconButton" href={artifactUrl(status.job_id, "qa_report.csv")}>
                <FileText aria-hidden="true" />
                <span>质检 CSV</span>
              </a>
              <a className="iconButton" href={artifactUrl(status.job_id, "manifest.csv")}>
                <FileText aria-hidden="true" />
                <span>清单</span>
              </a>
            </section>

            <section className="resultsLayout">
              <div className="tableShell">
                <table>
                  <thead>
                    <tr>
                      <th>行号</th>
                      <th>原文</th>
                      <th>文件</th>
                      <th>开始</th>
                      <th>结束</th>
                      <th>置信度</th>
                      <th>状态</th>
                      <th>ASR</th>
                      <th>标记</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredClips.map((clip) => (
                      <tr key={clip.clip_id} onClick={() => setSelected(clip)} className={selected?.clip_id === clip.clip_id ? "selectedRow" : ""}>
                        <td>{clip.line_index}</td>
                        <td>{clip.original_text}</td>
                        <td>{clip.output_file ?? "-"}</td>
                        <td>{formatSeconds(clip.start_sec)}</td>
                        <td>{formatSeconds(clip.end_sec)}</td>
                        <td>{clip.confidence}</td>
                        <td>
                          <span className={`statusBadge ${clip.status}`}>{qaStatusLabels[clip.status]}</span>
                        </td>
                        <td>{clip.asr_text || "-"}</td>
                        <td>{clip.flags.length ? clip.flags.join(", ") : "无"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <aside className="previewPanel">
                {selected ? (
                  <>
                    <div className="previewTitle">
                      <span className={`statusBadge ${selected.status}`}>{statusIcon(selected.status)} {qaStatusLabels[selected.status]}</span>
                      <strong>第 {selected.line_index} 行</strong>
                    </div>
                    <h3>{selected.original_text}</h3>
                    <p>{selected.asr_text || "暂无 ASR 文本"}</p>
                    {selected.output_file ? (
                      <audio controls src={clipUrl(status.job_id, selected.clip_id)} />
                    ) : (
                      <div className="emptyPreview">该行没有匹配到音频片段</div>
                    )}
                    <dl>
                      <div><dt>文件</dt><dd>{selected.output_file ?? "-"}</dd></div>
                      <div><dt>开始</dt><dd>{formatSeconds(selected.start_sec)}</dd></div>
                      <div><dt>结束</dt><dd>{formatSeconds(selected.end_sec)}</dd></div>
                      <div><dt>时长</dt><dd>{formatSeconds(selected.duration_sec)}</dd></div>
                      <div><dt>相似度</dt><dd>{selected.similarity?.toFixed(1) ?? "-"}</dd></div>
                      <div><dt>WER</dt><dd>{selected.wer?.toFixed(2) ?? "-"}</dd></div>
                      <div><dt>标记</dt><dd>{selected.flags.length ? selected.flags.join(", ") : "无"}</dd></div>
                    </dl>
                    {selected.output_file && (
                      <a className="primaryButton" href={clipUrl(status.job_id, selected.clip_id)}>
                        <Download aria-hidden="true" />
                        <span>下载片段</span>
                      </a>
                    )}
                  </>
                ) : (
                  <div className="emptyPreview">选择一行预览音频片段</div>
                )}
              </aside>
            </section>
          </>
        )}
      </main>
    </div>
  );
}
