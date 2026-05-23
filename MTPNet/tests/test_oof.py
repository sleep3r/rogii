from pathlib import Path

import pytest

from mtpnet.config import MTPConfig, RunConfig
from mtpnet.oof import _aggregate_named_metrics, make_oof_folds, run_oof


def test_make_oof_folds_cover_each_valid_well_once() -> None:
    folds = make_oof_folds([f"w{i}" for i in range(7)], n_folds=3, seed=13)

    held_out = [well for fold in folds for well in fold.valid_wells]

    assert sorted(held_out) == [f"w{i}" for i in range(7)]
    assert all(set(fold.train_wells).isdisjoint(fold.valid_wells) for fold in folds)
    assert all(fold.train_wells for fold in folds)
    assert {fold.fold for fold in folds} == {0, 1, 2}


def test_make_oof_folds_can_run_prefix_stress_test() -> None:
    folds = make_oof_folds([f"w{i}" for i in range(8)], n_folds=4, seed=1, max_folds=2)

    assert len(folds) == 2
    assert {fold.fold for fold in folds} == {0, 1}


def test_make_oof_folds_rejects_invalid_fold_count() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        make_oof_folds(["a"], n_folds=2, seed=1)


def test_aggregate_named_metrics_sorts_nan_rmse_last() -> None:
    metrics = _aggregate_named_metrics(
        [
            {"candidate": "bad", "rows": 0, "rmse": float("nan")},
            {"candidate": "good", "rows": 10, "rmse": 9.0},
        ]
    )

    assert metrics[0]["candidate"] == "good"


def test_run_oof_writes_fold_and_aggregate_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("run:\n  name: unit\n  output_dir: artifacts/unit\n")
    output_dir = tmp_path / "oof"

    class FakeWell:
        def __init__(self, well_id: str):
            self.well_id = well_id

    def fake_load_config(path: Path) -> MTPConfig:
        return MTPConfig(run=RunConfig(name="unit", output_dir=tmp_path / "unit"))

    def fake_discover_wells(_data):
        return [FakeWell(f"w{i}") for i in range(4)]

    train_calls = []

    def fake_train_from_config(
        path: Path,
        *,
        output_dir: Path,
        run_name: str,
        train_wells: tuple[str, ...],
        valid_wells: tuple[str, ...],
    ):
        train_calls.append((run_name, train_wells, valid_wells))
        output_dir.mkdir(parents=True, exist_ok=True)
        return {"run_name": run_name}

    def fake_run_stitch(run_dir: Path):
        return {"run_dir": str(run_dir)}

    def fake_write_mode_windows(run_dir: Path):
        return [object(), object()]

    def fake_run_tracker(run_dir: Path, **_kwargs):
        fold = int(run_dir.name.split("_")[-1])
        summary = {
            "baselines": {
                "b2_guarded_submit": {"candidate": "b2_guarded_submit", "rows": 10, "rmse": 10.0}
            },
            "coverage": {"covered_hidden_rows": 10, "total_hidden_rows": 10, "coverage_frac": 1.0},
            "candidates": [
                {
                    "candidate": "mtp_track_anchored_weighted_a0.2_clip20",
                    "rows": 10,
                    "rmse": 9.0 + fold,
                    "covered_rows": 10,
                    "covered_rmse": 9.0 + fold,
                    "p95_abs_shift_vs_b2": 1.0,
                    "worst_well_rmse": 20.0 + fold,
                }
            ],
        }
        (run_dir / "track_metrics.json").write_text("{}")
        return summary

    monkeypatch.setattr("mtpnet.oof.load_config", fake_load_config)
    monkeypatch.setattr("mtpnet.oof.discover_wells", fake_discover_wells)
    monkeypatch.setattr("mtpnet.oof.train_from_config", fake_train_from_config)
    monkeypatch.setattr("mtpnet.oof.run_stitch", fake_run_stitch)
    monkeypatch.setattr("mtpnet.oof.write_mode_windows", fake_write_mode_windows)
    monkeypatch.setattr("mtpnet.oof.run_tracker", fake_run_tracker)

    summary = run_oof(
        config_path,
        output_dir=output_dir,
        n_folds=2,
        seed=7,
        logit_source="nn",
    )

    assert len(train_calls) == 2
    assert all(set(train).isdisjoint(valid) for _, train, valid in train_calls)
    assert summary["aggregate"]["b2_guarded_submit"]["rmse"] == pytest.approx(10.0)
    assert summary["aggregate"]["best_candidate"]["candidate"] == (
        "mtp_track_anchored_weighted_a0.2_clip20"
    )
    assert (output_dir / "oof_metrics.json").exists()
    assert (output_dir / "oof_report.md").exists()
    assert (output_dir / "oof_candidates.csv").exists()
