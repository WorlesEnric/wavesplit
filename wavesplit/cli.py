from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import process_inputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wavesplit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process a wav + transcript pair.")
    process_parser.add_argument("audio")
    process_parser.add_argument("transcript")
    process_parser.add_argument("--out", required=True, help="Output job directory.")
    process_parser.add_argument("--config", default=None)
    process_parser.add_argument("--no-qa", action="store_true", help="Disable ASR QA for this run.")
    process_parser.add_argument("--asr-model", default=None)
    process_parser.add_argument("--asr-engine", default=None)

    args = parser.parse_args(argv)
    if args.command == "process":
        config = load_config(args.config)
        if args.no_qa:
            config.qa.enabled = False
        if args.asr_model:
            config.qa.asr_model = args.asr_model
        if args.asr_engine:
            config.qa.asr_engine = args.asr_engine
        payload = process_inputs(
            audio_path=args.audio,
            transcript_path=args.transcript,
            out_dir=args.out,
            config=config,
            job_id=Path(args.out).name,
        )
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
