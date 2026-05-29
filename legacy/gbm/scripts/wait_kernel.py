#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time


TERMINAL_OK = {"COMPLETE", "KernelWorkerStatus.COMPLETE"}
TERMINAL_BAD = {
    "ERROR",
    "CANCELLED",
    "FAILED",
    "KernelWorkerStatus.ERROR",
    "KernelWorkerStatus.CANCELLED",
    "KernelWorkerStatus.FAILED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-cmd", default="kaggle")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--poll-interval", type=float, default=120)
    return parser.parse_args()


def read_status(kaggle_cmd: str, kernel: str) -> str:
    proc = subprocess.run(
        [*kaggle_cmd.split(), "kernels", "status", kernel],
        check=True,
        text=True,
        capture_output=True,
    )
    line = proc.stdout.strip()
    print(line, flush=True)
    if '"' in line:
        return line.rsplit('"', 2)[1]
    return line.rsplit(maxsplit=1)[-1]


def main() -> None:
    args = parse_args()
    deadline = time.monotonic() + args.timeout
    while True:
        status = read_status(args.kaggle_cmd, args.kernel)
        if status in TERMINAL_OK or status.endswith(".COMPLETE"):
            return
        if status in TERMINAL_BAD or any(status.endswith(f".{name}") for name in TERMINAL_BAD):
            raise SystemExit(f"Kernel ended with {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {args.kernel}; last status={status}")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
