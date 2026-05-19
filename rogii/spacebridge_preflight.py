from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local spacebridge config.")
    parser.add_argument("--config", type=Path, default=Path("portainer.yml"))
    parser.add_argument("--instance", default=None)
    return parser.parse_args()


def require_text(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value in (None, ""):
        raise ValueError(f"portainer.yml must define {key}.")
    return str(value)


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Spacebridge config not found: {args.config}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    clearml_conf = Path(require_text(config, "CLEAR_ML_CONF_PATH")).expanduser()
    if not clearml_conf.is_file():
        raise FileNotFoundError(f"ClearML config not found: {clearml_conf}")

    require_text(config, "REGISTRY_USERNAME")

    instances = config.get("INSTANCES") or {}
    if not isinstance(instances, dict) or not instances:
        raise ValueError("portainer.yml must define at least one INSTANCES entry.")
    if args.instance and args.instance not in instances:
        available = ", ".join(sorted(instances))
        raise ValueError(
            f"INSTANCE={args.instance!r} not found in portainer.yml. "
            f"Available: {available}"
        )

    device_ids = config.get("DEVICE_IDS")
    if not isinstance(device_ids, list) or not device_ids:
        raise ValueError("portainer.yml must define non-empty DEVICE_IDS.")
    if config.get("SHARED_MEMORY_SIZE") in (None, ""):
        raise ValueError("portainer.yml must define SHARED_MEMORY_SIZE.")

    instance_text = args.instance or "<not selected>"
    print(
        f"Spacebridge config OK | instance={instance_text} registry_username=<set>",
        flush=True,
    )


if __name__ == "__main__":
    main()
