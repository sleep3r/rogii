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
from requests import HTTPError

from .kaggle_package import DEFAULT_INCLUDE, iter_package_files

DEFAULT_COMPETITION = "rogii-wellbore-geology-prediction"
DEFAULT_KERNEL = "rogii-baseline-infer"
DEFAULT_TITLE = "ROGII Baseline Infer"
DEFAULT_CONFIG = Path("configs/stack.yml")
DEFAULT_KERNEL_DIR = Path("artifacts/kaggle_kernel")
DEFAULT_OUTPUT_DIR = Path("artifacts/kaggle_output")
DEFAULT_DATA_DIR = Path("/kaggle/input/rogii-wellbore-geology-prediction")
DEFAULT_ARTIFACT_DIR = Path("artifacts/stack")
DEFAULT_MODEL_DIR = Path("artifacts/stack")
DEFAULT_MODEL_DATASET_DIR = Path("artifacts/kaggle_model_dataset")
DEFAULT_SUBMISSION = "submission.csv"
CLEARML_MODEL_ARTIFACTS = (
    "model.pkl",
    "features.json",
    "metrics.json",
    "config.yml",
    "source_config.yml",
)
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


def dataset_input_dir(dataset: str) -> str:
    parts = dataset.split("/")
    if len(parts) < 2:
        raise ValueError("Dataset source must be in owner/dataset-slug format.")
    return f"/kaggle/input/{parts[1]}"


def http_error_details(error: HTTPError) -> str:
    response = error.response
    if response is None:
        return str(error)
    body = response.text.strip()
    if len(body) > 2000:
        body = body[:2000] + "... truncated ..."
    return f"{error} | response={body}"


def iter_extra_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix not in {".pyc", ".pyo"}
            )
        elif path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            files.append(path)
    return sorted(files)


def source_zip_bytes(extra_paths: list[Path] | None = None) -> bytes:
    files = iter_package_files(list(DEFAULT_INCLUDE))
    if extra_paths:
        files.extend(iter_extra_files(extra_paths))
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


def encoded_source_zip(
    config: Path,
    mode: str,
    model_dir: Path | None = None,
    bundle_model: bool = False,
) -> str:
    extra_paths = [] if config.parent == Path("configs") else [config]
    if mode == "infer" and bundle_model:
        if model_dir is None:
            raise ValueError("Inference mode requires model_dir.")
        extra_paths.append(model_dir)
    data = source_zip_bytes(extra_paths)
    return "\n".join(wrap(base64.b64encode(data).decode("ascii"), 88))


def runner_script(
    *,
    config: Path,
    data_dir: Path,
    artifact_dir: Path,
    model_dir: Path | None,
    model_dataset: str,
    mode: str,
    submission_file: str,
) -> str:
    config_rel = relpath(config).as_posix()
    bundle_model = mode == "infer" and not model_dataset
    model_rel = (
        relpath(model_dir).as_posix() if bundle_model and model_dir is not None else ""
    )
    model_arg = ""
    if mode == "infer" and model_dataset:
        model_arg = dataset_input_dir(model_dataset)
    encoded = encoded_source_zip(
        config, mode=mode, model_dir=model_dir, bundle_model=bundle_model
    )
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
    if "{mode}" == "infer":
        from rogii.inference import main as run_job
    else:
        from rogii.pipeline import main as run_job

    resolved_data_dir = resolve_competition_data_dir(Path("{data_dir.as_posix()}"), work_dir)

    sys.argv = ["run.py"]
    if "{mode}" != "infer":
        sys.argv.extend(["--config", str(source_dir / "{config_rel}")])
    sys.argv.extend([
        "--data-dir",
        str(resolved_data_dir),
        "--submission",
        str(work_dir / "{submission_file}"),
        "--output-dir",
        str(work_dir / "{artifact_dir.as_posix()}"),
    ])
    if "{mode}" == "infer" and "{model_dataset}":
        sys.argv.extend(["--model-dir", "{model_arg}"])
    elif "{mode}" == "infer":
        sys.argv.extend(["--model-dir", str(source_dir / "{model_rel}")])
    run_job()


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
    dataset_sources: list[str] | None = None,
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
        "dataset_sources": dataset_sources or [],
        "competition_sources": [competition],
        "kernel_sources": [],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_model_dataset(args: argparse.Namespace) -> Path:
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    for name in ("model.pkl", "features.json", "metrics.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Missing model artifact: {model_dir / name}")

    dataset_dir = Path(args.model_dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    artifact_names = {
        "model.pkl",
        "features.json",
        "metrics.json",
        "config.yml",
        "source_config.yml",
    }
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.name in artifact_names:
            shutil.copyfile(path, dataset_dir / path.name)

    feature_names = json.loads((dataset_dir / "features.json").read_text())
    metrics = json.loads((dataset_dir / "metrics.json").read_text())
    feature_info = metrics.get("features", {}) if isinstance(metrics, dict) else {}
    dataset_slug = args.model_dataset.split("/", 1)[1]
    title_suffix = dataset_slug.rsplit("-", 1)[-1][:12]
    log(
        "Prepared model artifact files: "
        f"features={len(feature_names)} "
        f"schema={feature_info.get('schema_version')} "
        f"robust={sum('robust' in str(name) for name in feature_names)} "
        f"hmm={sum('hmm' in str(name) for name in feature_names)}"
    )

    metadata = {
        "title": f"ROGII Artifacts {title_suffix}",
        "id": args.model_dataset,
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": f"Trained ROGII model artifacts for {dataset_slug}",
        "description": (
            "Private model artifact dataset generated from the local ROGII repository."
        ),
        "resources": [
            {"path": path.name}
            for path in sorted(dataset_dir.iterdir())
            if path.is_file() and path.name != "dataset-metadata.json"
        ],
    }
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_dir


def is_already_exists_error(error: object) -> bool:
    text = str(error or "").lower()
    return any(
        marker in text
        for marker in (
            "already in use",
            "already exists",
            "duplicate",
            "title",
        )
    )


def wait_for_dataset(api: KaggleApi, dataset: str, timeout: int = 600) -> None:
    deadline = time.monotonic() + timeout
    slug = dataset.split("/", 1)[1] if "/" in dataset else dataset
    while True:
        try:
            status = str(api.dataset_status(dataset)).lower()
            log(f"Dataset status: {status}")
            if status in {"ready", "complete", "active"}:
                return
            if status in {"error", "failed"}:
                raise RuntimeError(f"Dataset {dataset} ended with status={status}")
        except HTTPError as error:
            if getattr(error.response, "status_code", None) != 403:
                raise
            try:
                rows = api.dataset_list(search=slug, mine=True) or []
            except HTTPError:
                rows = []
            refs = {str(getattr(row, "ref", "")) for row in rows if row is not None}
            if dataset in refs:
                log(
                    "Dataset status is not readable, but dataset is visible in mine list."
                )
                time.sleep(10)
                return
            log("Dataset status is not readable yet; waiting.")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for dataset {dataset}.")
        time.sleep(10)


def publish_model_dataset(api: KaggleApi, args: argparse.Namespace) -> None:
    if not args.model_dataset:
        raise ValueError("--publish-model-dataset requires --model-dataset.")
    dataset_dir = prepare_model_dataset(args)
    log(f"Prepared model dataset workspace: {dataset_dir}")
    log(f"Model dataset: {args.model_dataset}")
    try:
        log("Creating model dataset...")
        response = api.dataset_create_new(
            str(dataset_dir),
            public=False,
            quiet=False,
            convert_to_csv=False,
            dir_mode="skip",
        )
        error = getattr(response, "error", None)
        if is_already_exists_error(error):
            log("Model dataset already exists; creating a new version...")
            response = api.dataset_create_version(
                str(dataset_dir),
                version_notes=args.message,
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="skip",
            )
    except HTTPError as error:
        if getattr(error.response, "status_code", None) == 403:
            raise RuntimeError(
                "Kaggle dataset publish failed because this access token does not "
                "have dataset permissions. Create/refresh a Kaggle access token "
                "with dataset read/write permissions, then rerun submit-infer. "
                f"Details: {http_error_details(error)}"
            ) from error
        raise RuntimeError(
            f"Kaggle dataset publish failed: {http_error_details(error)}"
        ) from error

    error = getattr(response, "error", None)
    if error:
        raise RuntimeError(f"Kaggle dataset publish failed: {error}")
    url = getattr(response, "url", None)
    if url:
        log(f"Dataset URL: {url}")
    wait_for_dataset(api, args.model_dataset)


def prepare_kernel(args: argparse.Namespace) -> Path:
    config = Path(args.config)
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")
    model_dir = Path(args.model_dir) if args.mode == "infer" else None
    if args.mode == "infer" and not args.model_dataset:
        if model_dir is None or not model_dir.is_dir():
            raise FileNotFoundError(f"Model dir not found: {model_dir}")
        for name in ("model.pkl", "features.json"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"Missing model artifact: {model_dir / name}")

    kernel_dir = Path(args.kernel_dir)
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    run_py = runner_script(
        config=config,
        data_dir=Path(args.data_dir),
        artifact_dir=Path(args.artifact_dir),
        model_dir=model_dir,
        model_dataset=args.model_dataset,
        mode=args.mode,
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
        dataset_sources=[args.model_dataset] if args.model_dataset else None,
    )
    log(f"Prepared Kaggle kernel workspace: {kernel_dir}")
    log(f"Kernel: {kernel_ref(args.user, args.kernel)}")
    log(f"Config: {config}")
    if args.mode == "infer":
        log("Mode: inference-only")
        if args.model_dataset:
            log(f"Model dataset source: {args.model_dataset}")
        else:
            log(f"Bundled model dir: {model_dir}")
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


def api_push_kernel(api: KaggleApi, args: argparse.Namespace) -> object:
    accelerator = args.accelerator or None
    try:
        return api.kernels_push(
            str(args.kernel_dir),
            timeout=str(args.kernel_timeout),
            acc=accelerator,
        )
    except HTTPError as error:
        raise RuntimeError(
            f"Kaggle kernel push failed: {http_error_details(error)}"
        ) from error


def bootstrap_missing_kernel(api: KaggleApi, args: argparse.Namespace) -> None:
    kernel_dir = Path(args.kernel_dir)
    metadata_path = kernel_dir / "kernel-metadata.json"
    run_path = kernel_dir / "run.py"
    original_metadata = metadata_path.read_text(encoding="utf-8")
    original_run = run_path.read_text(encoding="utf-8")

    metadata = json.loads(original_metadata)
    metadata["dataset_sources"] = []
    metadata["competition_sources"] = []
    metadata["kernel_sources"] = []
    metadata["model_sources"] = []

    try:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        run_path.write_text(
            "print('ROGII inference kernel bootstrap')\n", encoding="utf-8"
        )
        log("Creating empty Kaggle kernel shell before attaching private sources...")
        response = api_push_kernel(api, args)
        if response.error:
            raise RuntimeError(f"Kaggle bootstrap kernel push failed: {response.error}")
        version = getattr(response, "version_number", None)
        if version:
            log(f"Bootstrap kernel version: {version}")
    finally:
        metadata_path.write_text(original_metadata, encoding="utf-8")
        run_path.write_text(original_run, encoding="utf-8")


def push_kernel(api: KaggleApi, args: argparse.Namespace) -> int:
    log("Pushing Kaggle kernel...")
    response = api_push_kernel(api, args)
    if response.error and "notebook not found" in str(response.error).lower():
        log("Kaggle returned 'Notebook not found'; bootstrapping kernel slug.")
        bootstrap_missing_kernel(api, args)
        log("Retrying Kaggle kernel push...")
        response = api_push_kernel(api, args)
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


def clearml_artifact_candidates(name: str, prefix: str) -> list[str]:
    normalized = prefix.strip("/")
    candidates = []
    if normalized:
        candidates.append(f"{normalized}/{name}")
    candidates.append(name)
    return candidates


def find_clearml_artifact(
    artifacts: dict[str, object],
    name: str,
    prefix: str,
) -> tuple[str, object]:
    for key in clearml_artifact_candidates(name, prefix):
        if key in artifacts:
            return key, artifacts[key]
    available = ", ".join(sorted(artifacts)) or "<none>"
    raise FileNotFoundError(
        f"ClearML artifact {name!r} was not found. Available artifacts: {available}"
    )


def copy_clearml_artifact(artifact: object, destination: Path) -> None:
    if not hasattr(artifact, "get_local_copy"):
        raise TypeError(f"ClearML artifact has no get_local_copy(): {artifact!r}")
    local_copy = Path(artifact.get_local_copy())  # type: ignore[attr-defined]
    if local_copy.is_dir():
        candidates = sorted(path for path in local_copy.rglob(destination.name))
        if not candidates:
            raise FileNotFoundError(
                f"Downloaded ClearML artifact directory has no {destination.name}: "
                f"{local_copy}"
            )
        local_copy = candidates[0]
    if not local_copy.is_file():
        raise FileNotFoundError(
            f"Downloaded ClearML artifact is not a file: {local_copy}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if local_copy.resolve() != destination.resolve():
        shutil.copyfile(local_copy, destination)


def download_clearml_artifacts_from_task(
    *,
    task: object,
    model_dir: Path,
    artifact_prefix: str = "output",
    names: tuple[str, ...] = CLEARML_MODEL_ARTIFACTS,
) -> Path:
    artifacts = getattr(task, "artifacts", None)
    if artifacts is None:
        raise AttributeError("ClearML task has no artifacts attribute.")
    artifacts = dict(artifacts)
    task_id = getattr(task, "id", "")
    task_name = getattr(task, "name", "")
    log(f"Fetching ClearML model artifacts: task_id={task_id} task_name={task_name}")
    log(f"ClearML artifact keys: {', '.join(sorted(artifacts))}")

    if model_dir.exists():
        for name in names:
            path = model_dir / name
            if path.exists():
                path.unlink()
    model_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        key, artifact = find_clearml_artifact(artifacts, name, artifact_prefix)
        destination = model_dir / name
        log(f"Downloading ClearML artifact: {key} -> {destination}")
        copy_clearml_artifact(artifact, destination)

    return model_dir


def download_clearml_model_artifacts(
    task_id: str,
    model_dir: Path,
    artifact_prefix: str = "output",
) -> Path:
    if not task_id:
        raise ValueError("ClearML task id is required.")
    from clearml import Task  # type: ignore

    task = Task.get_task(task_id=task_id)
    return download_clearml_artifacts_from_task(
        task=task,
        model_dir=model_dir,
        artifact_prefix=artifact_prefix,
    )


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
    if args.mode == "infer" and args.clearml_task_id:
        download_clearml_model_artifacts(
            args.clearml_task_id,
            Path(args.model_dir),
            artifact_prefix=args.clearml_artifact_prefix,
        )

    api = None
    if args.mode == "infer" and args.publish_model_dataset and not args.dry_run:
        api = make_api()
        publish_model_dataset(api, args)

    prepare_kernel(args)
    if args.dry_run:
        log("Dry run complete. No Kaggle push or competition submit was executed.")
        return

    if api is None:
        api = make_api()
    version = push_kernel(api, args)
    if args.push_only:
        ref = kernel_ref(args.user, args.kernel)
        log(
            f"Push-only mode complete: {ref} version {version} was submitted to Kaggle."
        )
        return
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
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-dataset", default="")
    parser.add_argument(
        "--model-dataset-dir", type=Path, default=DEFAULT_MODEL_DATASET_DIR
    )
    parser.add_argument("--clearml-task-id", "--cml-id", default="")
    parser.add_argument("--clearml-artifact-prefix", default="output")
    parser.add_argument("--publish-model-dataset", action="store_true")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
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
    run.add_argument("--push-only", action="store_true")
    run.add_argument("--skip-competition-submit", action="store_true")
    run.add_argument("--download-all-output", action="store_true")

    fetch_clearml = subparsers.add_parser(
        "fetch-clearml", help="Download trained model artifacts from a ClearML task."
    )
    fetch_clearml.add_argument("--clearml-task-id", "--cml-id", required=True)
    fetch_clearml.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    fetch_clearml.add_argument("--clearml-artifact-prefix", default="output")

    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not hasattr(args, "kernel"):
        return args
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
    elif args.command == "fetch-clearml":
        download_clearml_model_artifacts(
            args.clearml_task_id,
            Path(args.model_dir),
            artifact_prefix=args.clearml_artifact_prefix,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
