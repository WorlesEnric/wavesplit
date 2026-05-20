from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline
from .storage import job_dir_for


def run_job(job_id: str, config_path: str | None = None) -> dict[str, object]:
    config = load_config(config_path)
    return run_pipeline(job_dir_for(config, job_id), config=config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a WaveSplit worker job directly.")
    parser.add_argument("job_id")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    run_job(args.job_id, args.config)


if __name__ == "__main__":
    main()
