export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type QAStatus = "pass" | "review" | "fail" | "missing_audio";

export interface JobStatus {
  job_id: string;
  state: JobState;
  stage: string;
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
  input: {
    audio_filename: string;
    text_filename: string;
    line_count: number | null;
    audio_duration_sec: number | null;
  };
  counts: {
    total: number;
    aligned: number;
    cut: number;
    qa_pass: number;
    qa_review: number;
    qa_fail: number;
    qa_missing_audio?: number;
  };
  error: string | null;
}

export interface ClipRecord {
  clip_id: string;
  line_index: number;
  original_text: string;
  normalized_text: string;
  output_file: string | null;
  start_sec: number | null;
  end_sec: number | null;
  duration_sec: number | null;
  duplicate_index: number;
  alignment_score_mean: number | null;
  confidence: number;
  status: QAStatus;
  asr_text: string;
  asr_normalized_text: string;
  similarity: number | null;
  wer: number | null;
  leading_silence_ms: number | null;
  trailing_silence_ms: number | null;
  peak_dbfs: number | null;
  rms_dbfs: number | null;
  flags: string[];
}

export interface JobReport {
  job_id: string;
  summary: {
    total: number;
    pass: number;
    review: number;
    fail: number;
    missing_audio: number;
    duration_sec: number;
    alignment_engine: string;
    asr_engine: string;
    asr_model: string | null;
  };
  clips: ClipRecord[];
}

export interface AuthSession {
  authenticated: boolean;
  username: string | null;
}
