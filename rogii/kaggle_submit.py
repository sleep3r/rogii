from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from .kaggle_package import DEFAULT_INCLUDE, iter_package_files

DEFAULT_COMPETITION = "rogii-wellbore-geology-prediction"
DEFAULT_KERNEL = "rogii-hgb-submit"
DEFAULT_TITLE = "ROGII HGB Submit"
DEFAULT_CONFIG = Path("configs/best.yml")
DEFAULT_KERNEL_DIR = Path("artifacts/kaggle_kernel")
DEFAULT_OUTPUT_DIR = Path("artifacts/kaggle_output")
DEFAULT_DATA_DIR = Path("/kaggle/input/rogii-wellbore-geology-prediction")
DEFAULT_ARTIFACT_DIR = Path("artifacts/submit")
DEFAULT_SUBMISSION = "submission.csv"
TERMINAL_ERROR_STATUSES = {
    "ERROR",
    "CANCEL_REQUESTED",
    "CANCEL_ACKNOWLEDGED",
}


def log(message: str) -> None:
    print(message, flush=True)


def kernel_ref(user: str, kernel: str) -> str:
    if "/" in kernel:
        return kernel
    return f"{user}/{kernel}"


def relpath(path: Path) -> Path:
    return path.resolve().relative_to(Path.cwd().resolve())


def source_zip_bytes(extra_paths: list[Path] | None = None) -> bytes:
    include = list(DEFAULT_INCLUDE)
    if extra_paths:
        include.extend(extra_paths)

    files = iter_package_files(include)
    seen: set[Path] = set()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = relpath(path)
            if relative in seen:
                continue
            seen.add(relative)
            archive.write(path, relative.as_posix())
    return buffer.getvalue()


def encoded_source_zip(config: Path) -> str:
    extra_paths = [] if config.parent == Path("configs") else [config]
    data = source_zip_bytes(extra_paths)
    return "\n".join(wrap(base64.b64encode(data).decode("ascii"), 88))


def runner_script(
    *,
    config: Path,
    data_dir: Path,
    artifact_dir: Path,
    submission_file: str,
) -> str:
    config_rel = relpath(config).as_posix()
    encoded = encoded_source_zip(config)
    return f'''from __future__ import annotations

import base64
import io
import shutil
import sys
import zipfile
from pathlib import Path


SOURCE_ZIP_B64 = """
{encoded}
"""


def has_competition_layout(path: Path) -> bool:
    return (path / "train").is_dir() and (path / "test").is_dir()


def candidate_roots(root: Path):
    yielded = set()
    stack = [root]
    while stack:
        current = stack.pop(0)
        if current in yielded:
            continue
        yielded.add(current)
        yield current
        if current.exists() and current.is_dir():
            try:
                stack.extend(path for path in sorted(current.iterdir()) if path.is_dir())
            except OSError:
                pass


def describe_tree(root: Path, limit: int = 80) -> list[str]:
    if not root.exists():
        return [str(root) + " [missing]"]
    items = []
    for index, path in enumerate(sorted(root.rglob("*"))):
        if index >= limit:
            items.append("... truncated ...")
            break
        try:
            kind = "dir" if path.is_dir() else "file"
            items.append("%s [%s]" % (path, kind))
        except OSError:
            items.append(str(path))
    return items


def resolve_competition_data_dir(preferred: Path, work_dir: Path) -> Path:
    search_roots = [preferred, Path("/kaggle/input")]
    for root in search_roots:
        for candidate in candidate_roots(root):
            if has_competition_layout(candidate):
                print("Resolved competition data dir: %s" % candidate, flush=True)
                return candidate

    zip_candidates = []
    for root in search_roots:
        if root.exists():
            zip_candidates.extend(sorted(root.rglob("*.zip")))

    if zip_candidates:
        extract_dir = work_dir / "_rogii_data"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_path = zip_candidates[0]
        print("Extracting competition archive: %s -> %s" % (zip_path, extract_dir), flush=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        for candidate in candidate_roots(extract_dir):
            if has_competition_layout(candidate):
                print("Resolved extracted competition data dir: %s" % candidate, flush=True)
                return candidate

    print("Could not find Kaggle competition train/test layout.", flush=True)
    for root in search_roots:
        print("Tree sample for %s:" % root, flush=True)
        for line in describe_tree(root):
            print("  " + line, flush=True)
    raise FileNotFoundError("No Kaggle competition data directory with train/ and test/ was found.")


def main() -> None:
    work_dir = Path.cwd()
    source_dir = work_dir / "_rogii_src"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    payload = base64.b64decode(SOURCE_ZIP_B64.encode("ascii"))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(source_dir)

    sys.path.insert(0, str(source_dir))
    from rogii.pipeline import main as run_pipeline

    resolved_data_dir = resolve_competition_data_dir(Path("{data_dir.as_posix()}"), work_dir)

    sys.argv = [
        "run.py",
        "--config",
        str(source_dir / "{config_rel}"),
        "--data-dir",
        str(resolved_data_dir),
        "--submission",
        str(work_dir / "{submission_file}"),
        "--output-dir",
        str(work_dir / "{artifact_dir.as_posix()}"),
    ]
    run_pipeline()


if __name__ == "__main__":
    main()
'''


def write_metadata(
    *,
    kernel_dir: Path,
    competition: str,
    user: str,
    kernel: str,
    title: str,
    code_file: str,
    private: bool,
    internet: bool,
    gpu: bool,
    tpu: bool,
) -> None:
    metadata = {
        "id": kernel_ref(user, kernel),
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": str(private).lower(),
        "enable_gpu": str(gpu).lower(),
        "enable_tpu": str(tpu).lower(),
        "enable_internet": str(internet).lower(),
        "dataset_sources": [],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_kernel(args: argparse.Namespace) -> Path:
    config = Path(args.config)
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    kernel_dir = Path(args.kernel_dir)
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    run_py = runner_script(
        config=config,
        data_dir=Path(args.data_dir),
        artifact_dir=Path(args.artifact_dir),
        submission_file=args.submission_file,
    )
    (kernel_dir / "run.py").write_text(run_py, encoding="utf-8")

    write_metadata(
        kernel_dir=kernel_dir,
        competition=args.competition,
        user=args.user,
        kernel=args.kernel,
        title=args.title,
        code_file="run.py",
        private=args.private,
        internet=args.internet,
        gpu=args.gpu,
        tpu=args.tpu,
    )
    log(f"Prepared Kaggle kernel workspace: {kernel_dir}")
    log(f"Kernel: {kernel_ref(args.user, args.kernel)}")
    log(f"Config: {config}")
    return kernel_dir


def make_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def status_name(status: object) -> str:
    name = getattr(status, "name", None)
    if name:
        return str(name).upper()
    return str(status).split(".")[-1].upper()


def push_kernel(api: KaggleApi, args: argparse.Namespace) -> int:
    log("Pushing Kaggle kernel...")
    accelerator = args.accelerator or None
    response = api.kernels_push(
        str(args.kernel_dir),
        timeout=str(args.kernel_timeout),
        acc=accelerator,
    )
    if response.error:
        raise RuntimeError(response.error)
    version = int(response.version_number)
    log(f"Pushed kernel version: {version}")
    if response.url:
        log(f"Kernel URL: {response.url}")
    return version


def wait_for_kernel(api: KaggleApi, args: argparse.Namespace) -> None:
    ref = kernel_ref(args.user, args.kernel)
    deadline = time.monotonic() + args.wait_timeout
    log(f"Waiting for Kaggle kernel to finish: {ref}")
    while True:
        response = api.kernels_status(ref)
        name = status_name(response.status)
        failure_message = response.failure_message
        log(f"Kernel status: {name}")
        if name == "COMPLETE":
            return
        if name in TERMINAL_ERROR_STATUSES:
            details = f": {failure_message}" if failure_message else ""
            raise RuntimeError(f"Kaggle kernel ended with {name}{details}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {ref}; last status was {name}")
        time.sleep(args.poll_interval)


def download_output(api: KaggleApi, args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref = kernel_ref(args.user, args.kernel)
    log(f"Downloading kernel output to {output_dir}...")
    file_pattern = None if args.download_all_output else args.submission_file
    api.kernels_output(
        ref,
        str(output_dir),
        file_pattern=file_pattern,
        force=True,
        quiet=False,
    )

    candidates = sorted(output_dir.rglob(args.submission_file))
    if not candidates:
        raise FileNotFoundError(f"{args.submission_file} was not found in {output_dir}")
    return candidates[0]


def validate_submission(path: Path) -> None:
    log(f"Validating submission: {path}")
    frame = pd.read_csv(path)
    expected_columns = ["id", "tvt"]
    if list(frame.columns) != expected_columns:
        raise ValueError(
            f"Expected columns {expected_columns}, got {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("submission.csv is empty")
    if frame["id"].isna().any():
        raise ValueError("submission.csv contains empty ids")
    tvt = frame["tvt"].to_numpy(dtype=float)
    if not np.isfinite(tvt).all():
        raise ValueError("submission.csv contains non-finite tvt values")
    log(f"Submission rows: {len(frame):,}")


def submit_code(api: KaggleApi, args: argparse.Namespace, version: int) -> None:
    ref = kernel_ref(args.user, args.kernel)
    log(f"Submitting code output: kernel={ref} version={version}")
    response = api.competition_submit_code(
        file_name=args.submission_file,
        message=args.message,
        competition=args.competition,
        kernel=ref,
        kernel_version=version,
        quiet=False,
    )
    log(f"Kaggle response: {response.message}")
    if response.ref:
        log(f"Submission ref: {response.ref}")


def run_end_to_end(args: argparse.Namespace) -> None:
    prepare_kernel(args)
    if args.dry_run:
        log("Dry run complete. No Kaggle push or competition submit was executed.")
        return

    api = make_api()
    version = push_kernel(api, args)
    wait_for_kernel(api, args)
    submission_path = download_output(api, args)
    validate_submission(submission_path)
    if args.skip_competition_submit:
        log("Skipping competition submit. Kaggle training run is complete.")
        return
    submit_code(api, args, version)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--user", default="sleep3r")
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--kernel-dir", type=Path, default=DEFAULT_KERNEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--submission-file", default=DEFAULT_SUBMISSION)
    parser.add_argument(
        "--private", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--internet", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tpu", action=argparse.BooleanOptionalAction, default=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, run, and submit a Kaggle code competition kernel."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Build the self-contained Kaggle kernel workspace."
    )
    add_common_args(prepare)

    run = subparsers.add_parser(
        "run", help="Push kernel, wait for output, validate, and submit."
    )
    add_common_args(run)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--kernel-timeout", type=int, default=9 * 60 * 60)
    run.add_argument("--wait-timeout", type=int, default=10 * 60 * 60)
    run.add_argument("--poll-interval", type=int, default=60)
    run.add_argument("--accelerator", default="")
    run.add_argument("--message", default="Submission")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-competition-submit", action="store_true")
    run.add_argument("--download-all-output", action="store_true")

    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    kernel_slug = args.kernel.split("/")[-1]
    if args.title == DEFAULT_TITLE and kernel_slug != DEFAULT_KERNEL:
        args.title = kernel_slug
    return args


def main(argv: list[str] | None = None) -> None:
    args = normalize_args(parse_args(argv))
    if args.command == "prepare":
        prepare_kernel(args)
    elif args.command == "run":
        run_end_to_end(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
