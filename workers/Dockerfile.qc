# QC worker (queue: auto-qc). Build from the wavesplit repo root:
#   docker build -f workers/Dockerfile.qc -t wavesplit-qc-worker .
# Run (own host):
#   docker run --rm \
#     -e GATEWAY_URL=http://gateway:4500 \
#     -e WORKER_BEARER=<auto-qc service-account token> \
#     -e QUEUE=auto-qc \
#     wavesplit-qc-worker
FROM python:3.11-slim

# ffmpeg/ffprobe are required at runtime (wavesplit.audio shells out to them).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ONLY the QC deps (faster-whisper; no torch/onnxruntime).
COPY workers/requirements-qc.txt /app/workers/requirements-qc.txt
RUN pip install --no-cache-dir -r workers/requirements-qc.txt

# Bring in the wavesplit library + the workers package.
COPY wavesplit /app/wavesplit
COPY workers /app/workers

ENV PYTHONUNBUFFERED=1 \
    QUEUE=auto-qc \
    GATEWAY_URL=http://localhost:4500 \
    POLL_INTERVAL=1.0

ENTRYPOINT ["python", "-m", "workers.qc_worker"]
