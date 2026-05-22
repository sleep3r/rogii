from pathlib import Path

import pandas as pd
import pytest

from mtpnet.config import DataConfig
from mtpnet.io import copy_data_tree, discover_wells


def make_well(root: Path, well_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"id": [f"{well_id}_0"], "TVT": [1.0], "TVT_input": [1.0], "GR": [10.0]}
    ).to_csv(root / f"{well_id}__horizontal_well.csv", index=False)
    pd.DataFrame({"TVT": [1.0], "GR": [10.0]}).to_csv(
        root / f"{well_id}__typewell.csv", index=False
    )


def test_discover_wells_respects_positive_k(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    make_well(train, "b_well")
    make_well(train, "a_well")
    wells = discover_wells(
        DataConfig(data_dir=tmp_path / "data", train_dir=train, k_wells=1)
    )
    assert [well.well_id for well in wells] == ["a_well"]


def test_discover_wells_minus_one_uses_all(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    make_well(train, "b_well")
    make_well(train, "a_well")
    wells = discover_wells(
        DataConfig(data_dir=tmp_path / "data", train_dir=train, k_wells=-1)
    )
    assert [well.well_id for well in wells] == ["a_well", "b_well"]


def test_missing_typewell_fails(tmp_path: Path) -> None:
    train = tmp_path / "data" / "train"
    train.mkdir(parents=True)
    pd.DataFrame({"TVT": [1.0], "TVT_input": [1.0], "GR": [10.0]}).to_csv(
        train / "lonely__horizontal_well.csv", index=False
    )
    with pytest.raises(FileNotFoundError, match="Missing typewell"):
        discover_wells(DataConfig(data_dir=tmp_path / "data", train_dir=train, k_wells=-1))


def test_copy_data_tree_copies_known_subdirs(tmp_path: Path) -> None:
    source = tmp_path / "old_data"
    make_well(source / "train", "well")
    target = tmp_path / "new_data"
    copy_data_tree(source, target)
    assert (target / "train" / "well__horizontal_well.csv").exists()
    assert (target / "train" / "well__typewell.csv").exists()
