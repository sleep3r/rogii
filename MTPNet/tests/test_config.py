from pathlib import Path

import pytest
import yaml

from mtpnet.config import load_config


def write_config(tmp_path: Path, k_wells: int) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "data_dir": "data",
                    "train_dir": "data/train",
                    "test_dir": "data/test",
                    "k_wells": k_wells,
                },
                "run": {"name": "unit"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_k_wells_minus_one_is_all(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, -1))
    assert cfg.data.k_wells == -1
    assert cfg.data.use_all_wells is True


def test_positive_k_wells_is_limited(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, 7))
    assert cfg.data.k_wells == 7
    assert cfg.data.use_all_wells is False


def test_zero_k_wells_fails(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="data.k_wells must be -1 or a positive integer"
    ):
        load_config(write_config(tmp_path, 0))
