"""Docker entrypoint with useful diagnostics for remote Portainer runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EntrypointArgs:
    config: str
    rebuild_cache: bool = False
    keep_alive_on_fail: int = 0
    preflight_sleep: int = 0
    sleep_only: bool = False


def parse_entrypoint_args(argv: list[str] | None = None) -> EntrypointArgs:
    parser = argparse.ArgumentParser(description="BPHWT Docker entrypoint")
    parser.add_argument("--config", default="configs/bphwt_server.yml")
    parser.add_argument("--rebuild-cache", default=os.getenv("BPHWT_REBUILD_CACHE", "false"))
    parser.add_argument("--keep-alive-on-fail", type=int, default=0)
    parser.add_argument("--preflight-sleep", type=int, default=0)
    parser.add_argument("--sleep-only", default="false")
    ns, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[entrypoint] ignoring unknown args: {unknown}", flush=True)
    return EntrypointArgs(
        config=ns.config,
        rebuild_cache=_parse_bool(str(ns.rebuild_cache)),
        keep_alive_on_fail=max(0, int(ns.keep_alive_on_fail)),
        preflight_sleep=max(0, int(ns.preflight_sleep)),
        sleep_only=_parse_bool(str(ns.sleep_only)),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_entrypoint_args(argv)
    _print_preflight(args)
    if args.preflight_sleep > 0:
        print(f"[entrypoint] sleeping before train for {args.preflight_sleep}s", flush=True)
        time.sleep(args.preflight_sleep)
    if args.sleep_only:
        print("[entrypoint] sleep-only requested; exiting without training", flush=True)
        return 0

    try:
        cmd = [sys.executable, "-m", "bphwt.train", args.config]
        if args.rebuild_cache:
            cmd.append("--rebuild-cache")
        print(f"[entrypoint] running: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0 and args.keep_alive_on_fail > 0:
            print(
                f"[entrypoint] train exited with code {result.returncode}; "
                f"keeping container alive for {args.keep_alive_on_fail}s",
                flush=True,
            )
            time.sleep(args.keep_alive_on_fail)
        return result.returncode
    except BaseException:
        traceback.print_exc()
        if args.keep_alive_on_fail > 0:
            print(f"[entrypoint] failure: keeping container alive for {args.keep_alive_on_fail}s", flush=True)
            time.sleep(args.keep_alive_on_fail)
        raise


def _print_preflight(args: EntrypointArgs) -> None:
    print("[entrypoint] BPHWT container preflight", flush=True)
    print(f"[entrypoint] cwd={Path.cwd()}", flush=True)
    print(f"[entrypoint] argv_config={args.config}", flush=True)
    print(f"[entrypoint] python={sys.version.split()[0]} exe={sys.executable}", flush=True)
    print(f"[entrypoint] files={sorted(p.name for p in Path.cwd().iterdir())}", flush=True)
    print(f"[entrypoint] configs={sorted(p.name for p in Path('configs').glob('*.yml'))}", flush=True)
    print(f"[entrypoint] config_exists={Path(args.config).exists()} path={Path(args.config).resolve()}", flush=True)
    for key in ["CMD_ARGS", "CLEARML_CONFIG_FILE", "CLEARML_CACHE_DIR", "NVIDIA_VISIBLE_DEVICES"]:
        value = os.getenv(key)
        if value:
            print(f"[entrypoint] env {key}={value}", flush=True)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
