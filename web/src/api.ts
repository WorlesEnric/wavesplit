import type { AuthSession, JobReport, JobStatus } from "./types";

const CHUNK_SIZE = 16 * 1024 * 1024;
const CHUNK_TIMEOUT_MS = 300000;
const CHUNK_RETRIES = 3;

const ERROR_MESSAGES: Record<string, string> = {
  "Authentication required.": "请先登录。",
  "Invalid username or password.": "用户名或密码错误。",
  "Please upload a .wav audio file.": "请上传 .wav 音频文件。",
  "Please upload a .txt transcript file.": "请上传 .txt 文本文件。",
  "Upload files must not be empty.": "上传文件不能为空。",
  "Invalid upload chunk count.": "上传分片数量无效。",
  "Uploaded file exceeds the configured size limit.": "上传文件超过配置的大小限制。",
  "Invalid upload file kind.": "上传文件类型无效。",
  "Invalid upload chunk index.": "上传分片序号无效。",
  "Upload session not found.": "上传会话不存在。",
  "Invalid upload session metadata.": "上传会话元数据无效。",
  "Upload is missing one or more chunks.": "上传缺少一个或多个分片。",
  "Job status not found.": "任务状态不存在。",
  "Report is not ready.": "报告尚未生成。",
  "Zip is not ready.": "ZIP 文件尚未生成。",
  "Manifest is not ready.": "清单尚未生成。",
  "QA report is not ready.": "质检报告尚未生成。",
  "Clip id not found.": "音频片段不存在。",
  "Invalid clip path.": "音频片段路径无效。",
  "Clip file is missing.": "音频片段文件缺失。"
};

function errorMessage(payload: { detail?: unknown } | null, fallback: string) {
  const detail = typeof payload?.detail === "string" ? payload.detail : "";
  return detail ? ERROR_MESSAGES[detail] ?? detail : fallback;
}

export async function getAuthSession(): Promise<AuthSession> {
  const response = await fetch("/api/auth/session");
  if (!response.ok) return { authenticated: false, username: null };
  return response.json();
}

export async function login(username: string, password: string): Promise<AuthSession> {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password })
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(errorMessage(payload, "登录失败"));
  }
  return response.json();
}

export async function logout(): Promise<AuthSession> {
  const response = await fetch("/api/auth/logout", { method: "POST" });
  if (!response.ok) throw new Error("退出登录失败");
  return response.json();
}

export async function createJob(audio: File, transcript: File, onProgress?: (progress: number) => void): Promise<{ job_id: string; state: string }> {
  if (audio.size + transcript.size > CHUNK_SIZE) return createChunkedJob(audio, transcript, onProgress);

  const body = new FormData();
  body.append("audio", audio);
  body.append("transcript", transcript);
  const response = await fetch("/api/jobs", { method: "POST", body });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(errorMessage(payload, "上传失败"));
  }
  return response.json();
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timeout);
  }
}

async function uploadChunkWithRetry(url: string, body: FormData) {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= CHUNK_RETRIES; attempt += 1) {
    try {
      const response = await fetchWithTimeout(url, { method: "POST", body }, CHUNK_TIMEOUT_MS);
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(errorMessage(payload, "上传失败"));
      }
      return;
    } catch (error) {
      lastError = error;
      if (attempt === CHUNK_RETRIES) break;
      await delay(attempt * 1000);
    }
  }
  const message = lastError instanceof Error ? lastError.message : "上传失败";
  throw new Error(message === "The operation was aborted." ? "上传分片超时，请重试。" : message);
}

async function createChunkedJob(audio: File, transcript: File, onProgress?: (progress: number) => void): Promise<{ job_id: string; state: string }> {
  const audioChunks = Math.ceil(audio.size / CHUNK_SIZE);
  const transcriptChunks = Math.ceil(transcript.size / CHUNK_SIZE);
  const init = await fetch("/api/uploads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio_filename: audio.name,
      transcript_filename: transcript.name,
      audio_size: audio.size,
      transcript_size: transcript.size,
      audio_chunks: audioChunks,
      transcript_chunks: transcriptChunks
    })
  });
  if (!init.ok) {
    const payload = await init.json().catch(() => null);
    throw new Error(errorMessage(payload, "上传失败"));
  }

  const { upload_id } = (await init.json()) as { upload_id: string };
  const totalBytes = audio.size + transcript.size;
  let uploadedBytes = 0;

  async function sendChunks(fileKind: "audio" | "transcript", file: File, chunkCount: number) {
    for (let index = 0; index < chunkCount; index += 1) {
      const start = index * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const body = new FormData();
      body.append("file_kind", fileKind);
      body.append("chunk_index", String(index));
      body.append("chunk", file.slice(start, end), `${file.name}.part${index}`);
      await uploadChunkWithRetry(`/api/uploads/${upload_id}/chunks`, body);
      uploadedBytes += end - start;
      onProgress?.(Math.max(0, Math.min(1, uploadedBytes / totalBytes)));
    }
  }

  await sendChunks("audio", audio, audioChunks);
  await sendChunks("transcript", transcript, transcriptChunks);

  const complete = await fetch(`/api/uploads/${upload_id}/complete`, { method: "POST" });
  if (!complete.ok) {
    const payload = await complete.json().catch(() => null);
    throw new Error(errorMessage(payload, "上传失败"));
  }
  onProgress?.(1);
  return complete.json();
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) throw new Error("任务状态不可用");
  return response.json();
}

export async function getReport(jobId: string): Promise<JobReport> {
  const response = await fetch(`/api/jobs/${jobId}/report`);
  if (!response.ok) throw new Error("报告尚未生成");
  return response.json();
}

export function artifactUrl(jobId: string, artifact: "download" | "manifest.csv" | "qa_report.csv" | "diagnostics.zip") {
  return `/api/jobs/${jobId}/${artifact}`;
}

export function clipUrl(jobId: string, clipId: string) {
  return `/api/jobs/${jobId}/clips/${clipId}`;
}
