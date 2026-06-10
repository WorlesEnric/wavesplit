"""Standalone HTTP workers that connect to the EchoThink gateway over HTTP.

Two pull-workers live here:

* ``qc_worker``        — serves queue ``auto-qc``  (single-clip automated QC).
* ``alignment_worker`` — serves queue ``align-audio`` (long WAV + TXT → clips).

Both speak the same gateway HTTP protocol implemented by
:mod:`wavesplit.workers.gateway_client`, mirroring the TypeScript reference
worker (``echothink-worker/src/gateway-transport.ts`` + ``run.ts``).
"""
