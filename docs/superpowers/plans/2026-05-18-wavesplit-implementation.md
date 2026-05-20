# WaveSplit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete wav + txt audio slicing system described in `design.md`, then verify it against `data/20260518`.

**Architecture:** Implement a Python pipeline core that writes every job artifact to `storage/jobs/{job_id}` and can be called by CLI, RQ worker, or FastAPI. Add a Vite React UI that uploads files, streams progress, displays reports, previews clips, and downloads artifacts.

**Tech Stack:** Python 3.11+, FastAPI, RQ/Redis, ffmpeg/ffprobe, faster-whisper, numpy/pandas, pytest, Vite React TypeScript.

---

## Chunk 1: Pipeline Core

- [ ] Create config, data models, storage, status, transcript normalization, audio probing/cutting, alignment, segment padding, QA, packaging, and CLI modules.
- [ ] Add focused tests for text, naming, overlap handling, QA scoring, and a synthetic end-to-end pipeline job.
- [ ] Run tests and fix defects.

## Chunk 2: API And Worker

- [ ] Add FastAPI upload, status, SSE, report, clip preview, manifest, QA, diagnostics, and zip download routes.
- [ ] Add RQ worker entrypoint and local background fallback when Redis is unavailable.
- [ ] Verify API routes with a local job.

## Chunk 3: Web UI

- [ ] Add Vite React TypeScript app with upload, progress stages, summary, filterable table, preview panel, and download buttons.
- [ ] Build the frontend and serve the built app from FastAPI.
- [ ] Verify in browser across the main workflow.

## Chunk 4: Sample Acceptance

- [ ] Copy `data/20260518` inputs into a new job directory.
- [ ] Run the full pipeline on `H19-英式005.wav` and `英式005.txt`.
- [ ] Confirm 239 clips, manifests, QA reports, zip contents, duplicate suffix naming, and preview/download paths.
