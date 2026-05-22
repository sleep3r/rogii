from pathlib import Path

import pandas as pd
import yaml

from mtpnet.train import train_from_config


def make_well(root: Path, well_id: str, shift: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    n = 160
    tvt = [1000.0 + shift + i * 0.5 for i in range(n)]
    gr = [80.0 + ((i % 20) - 10) * 1.5 for i in range(n)]
    tvt_input = [value if i < 80 else None for i, value in enumerate(tvt)]
    pd.DataFrame(
        {
            "id": [f"{well_id}_{i}" for i in range(n)],
            "TVT": tvt,
            "TVT_input": tvt_input,
            "GR": gr,
        }
    ).to_csv(root / f"{well_id}__horizontal_well.csv", index=False)
    type_tvt = [970.0 + shift + i * 0.5 for i in range(260)]
    type_gr = [80.0 + ((i % 20) - 10) * 1.5 for i in range(260)]
    pd.DataFrame({"TVT": type_tvt, "GR": type_gr}).to_csv(
        root / f"{well_id}__typewell.csv", index=False
    )


def test_train_from_config_writes_metrics(tmp_path: Path) -> None:
    train_dir = tmp_path / "data" / "train"
    make_well(train_dir, "well_a", 0.0)
    make_well(train_dir, "well_b", 5.0)
    config = {
        "data": {"data_dir": str(tmp_path / "data"), "train_dir": str(train_dir), "k_wells": -1},
        "window": {
            "rows_per_step": 4,
            "history_steps": 4,
            "future_steps": 6,
            "vertical_bins": 32,
            "vertical_radius_ft": 60.0,
            "stride_steps": 4,
            "max_windows_per_well": 4,
            "channels": ["gr_diff", "abs_gr_diff", "history_mask", "history_sdf", "finite_mask"],
        },
        "model": {"k_modes": 3, "conv_channels": [8, 16], "hidden_dims": [32], "dropout": 0.0},
        "train": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.001,
            "device": "cpu",
            "num_workers": 0,
            "seed": 42,
        },
        "validation": {"valid_fraction": 0.5, "seed": 42},
        "run": {"name": "unit", "output_dir": str(tmp_path / "artifacts" / "unit")},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = train_from_config(config_path)
    assert summary["valid"]["num_windows"] > 0
    assert (tmp_path / "artifacts" / "unit" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / "unit" / "checkpoints" / "best.pt").exists()
