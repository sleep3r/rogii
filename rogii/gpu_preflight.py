from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail fast when the selected config needs GPU runtime support.",
        add_help=False,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stack.yml"))
    args, _unknown = parser.parse_known_args(argv)
    return args


def base_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config.get("model", {}).get("base_models") or [])


def lightgbm_gpu_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = []
    for model in base_models(config):
        if str(model.get("name", "")).lower() != "lightgbm":
            continue
        params = model.get("params") or {}
        device_type = str(params.get("device_type", "cpu")).lower()
        if device_type in {"gpu", "cuda"}:
            models.append(model)
    return models


def catboost_gpu_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = []
    for model in base_models(config):
        if str(model.get("name", "")).lower() != "catboost":
            continue
        params = model.get("params") or {}
        if str(params.get("task_type", "CPU")).upper() == "GPU":
            models.append(model)
    return models


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"Required command not found in container: {command[0]}")
    return subprocess.run(command, text=True, capture_output=True, check=False)


def require_command(command: list[str], label: str) -> str:
    result = run_command(command)
    output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}: {output}"
        )
    return output


def run_lightgbm_opencl_smoke(model: dict[str, Any]) -> None:
    from lightgbm import LGBMRegressor

    params = dict(model.get("params") or {})
    smoke_params = {
        "device_type": params.get("device_type", "gpu"),
        "gpu_device_id": int(params.get("gpu_device_id", 0)),
        "gpu_use_dp": bool(params.get("gpu_use_dp", False)),
        "max_bin": int(params.get("max_bin", 63)),
        "n_estimators": 1,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 1,
        "n_jobs": 1,
        "verbosity": -1,
        "random_state": 42,
    }
    X = np.array(
        [
            [0.0, 1.0, 0.2],
            [1.0, 0.0, 0.3],
            [2.0, 1.0, 0.4],
            [3.0, 0.0, 0.5],
            [4.0, 1.0, 0.6],
            [5.0, 0.0, 0.7],
            [6.0, 1.0, 0.8],
            [7.0, 0.0, 0.9],
        ],
        dtype=np.float32,
    )
    y = np.array([0.0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.8], dtype=np.float32)
    LGBMRegressor(**smoke_params).fit(X, y)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    mode = os.getenv("ROGII_GPU_PREFLIGHT", "auto").lower()
    if mode in {"0", "false", "off", "skip"}:
        print("GPU preflight skipped by ROGII_GPU_PREFLIGHT", flush=True)
        return

    args = parse_args(argv)
    config = load_config(args.config)
    lgb_gpu = lightgbm_gpu_models(config)
    cat_gpu = catboost_gpu_models(config)
    if not lgb_gpu and not cat_gpu:
        print("GPU preflight: no GPU models requested", flush=True)
        return

    print(
        "GPU preflight: requested "
        f"catboost_gpu={len(cat_gpu)} lightgbm_gpu={len(lgb_gpu)}",
        flush=True,
    )
    nvidia_smi = require_command(["nvidia-smi", "-L"], "nvidia-smi -L")
    print(f"GPU preflight: {nvidia_smi}", flush=True)

    if lgb_gpu:
        vendors_dir = Path("/etc/OpenCL/vendors")
        vendor_files = sorted(path.name for path in vendors_dir.glob("*.icd"))
        print(f"GPU preflight: OpenCL ICD files={vendor_files}", flush=True)
        clinfo = require_command(["clinfo", "-l"], "clinfo -l")
        print(f"GPU preflight: {clinfo}", flush=True)
        if "Platform" not in clinfo or "Device" not in clinfo:
            raise RuntimeError(f"OpenCL has no visible device:\n{clinfo}")
        run_lightgbm_opencl_smoke(lgb_gpu[0])
        print("GPU preflight: LightGBM OpenCL smoke passed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GPU preflight failed: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
