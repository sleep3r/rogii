from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from racformer.model import RACFormerOutput
from racformer.oof_diagnostics import (
    DEFAULT_VARIANTS,
    build_row_records,
    compute_variant_predictions,
    sample_diagnostic_fields,
    summarize_oof_dataframe,
)


def _dummy_output() -> RACFormerOutput:
    return RACFormerOutput(
        s_pred=torch.tensor([[1.0, 10.0]], dtype=torch.float32),
        bucket_logits=torch.zeros((1, 4, 2), dtype=torch.float32),
        direct_resid_step=torch.tensor([[0.0, 2.0, 5.0, 7.0]], dtype=torch.float32),
        event_logits=torch.zeros((1, 4), dtype=torch.float32),
        top_event_logits=torch.zeros((1, 4), dtype=torch.float32),
        top_dir_logits=torch.zeros((1, 4, 3), dtype=torch.float32),
        encoder_out=torch.zeros((1, 4, 3), dtype=torch.float32),
    )


def _dummy_batch() -> dict:
    return {
        "anchor_tvt": torch.tensor([100.0], dtype=torch.float32),
        "anchor_z": torch.tensor([1000.0], dtype=torch.float32),
        "z_hidden": torch.tensor([[997.0, 994.0, 991.0]], dtype=torch.float32),
        "base_tvt_hidden": torch.tensor([[103.3, 106.6, 109.9]], dtype=torch.float32),
        "n_hidden_rows": torch.tensor([3], dtype=torch.long),
        "hidden_row_to_step": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "anchor_step": torch.tensor([1], dtype=torch.long),
    }


def test_compute_variant_predictions_zeroes_each_ablation_head() -> None:
    preds = compute_variant_predictions(_dummy_batch(), _dummy_output(), k_seg=2)

    assert list(preds) == list(DEFAULT_VARIANTS)
    np.testing.assert_allclose(preds["last_known_tvt"][0, :3].numpy(), [100.0, 100.0, 100.0])
    np.testing.assert_allclose(preds["base_no_c0"][0, :3].numpy(), [103.0, 106.0, 109.0])
    np.testing.assert_allclose(preds["base_with_c0"][0, :3].numpy(), [103.3, 106.6, 109.9])
    np.testing.assert_allclose(preds["model_full"][0, :3].numpy(), [104.3, 120.6, 135.9], rtol=1e-6)
    np.testing.assert_allclose(preds["model_no_direct"][0, :3].numpy(), [104.3, 117.6, 130.9], rtol=1e-6)
    np.testing.assert_allclose(preds["model_no_s_pred"][0, :3].numpy(), [103.3, 109.6, 114.9])


def test_sample_diagnostic_fields_uses_hidden_gr_events_and_base_rmse() -> None:
    sample = SimpleNamespace(
        n_hidden_rows=3,
        features=np.array(
            [
                [0.0] * 26 + [1.0],
                [0.0] * 26 + [0.5],
                [0.0] * 26 + [0.0],
            ],
            dtype=np.float32,
        ),
        hidden_mask=np.array([True, True, False]),
        c0=-0.25,
        dC_forward=np.array([0.05, -0.20, 0.30], dtype=np.float32),
    )

    fields = sample_diagnostic_fields(
        sample,
        base_with_c0=np.array([10.0, 12.0, 14.0], dtype=np.float32),
        tvt_true=np.array([11.0, 12.0, 17.0], dtype=np.float32),
        event_threshold=0.10,
    )

    assert fields["hidden_length"] == 3
    assert fields["GR_valid_frac"] == 0.75
    assert fields["abs_c0"] == 0.25
    assert fields["event_count"] == 2
    assert fields["base_only_rmse"] == np.sqrt((1.0 + 0.0 + 9.0) / 3.0)


def test_build_row_records_emits_every_variant_and_diagnostics() -> None:
    sample = SimpleNamespace(
        well_id="well_a",
        anchor_row=10,
        hidden_row_ids=["well_a_11", "well_a_12"],
        tvt_rows=np.array([np.nan] * 11 + [101.0, 102.0], dtype=np.float32),
    )
    preds = {
        name: torch.tensor([[100.0 + i, 200.0 + i]], dtype=torch.float32)
        for i, name in enumerate(DEFAULT_VARIANTS)
    }
    diagnostics = {
        "hidden_length": 2,
        "GR_valid_frac": 0.5,
        "abs_c0": 0.1,
        "event_count": 1,
        "base_only_rmse": 3.0,
    }

    rows = build_row_records(fold=1, sample=sample, variant_predictions=preds, diagnostics=diagnostics)

    assert rows[0]["fold"] == 1
    assert rows[0]["well_id"] == "well_a"
    assert rows[0]["id"] == "well_a_11"
    assert rows[0]["row_idx"] == 11
    assert rows[0]["tvt_true"] == 101.0
    assert rows[1]["model_no_s_pred"] == 205.0
    assert rows[1]["base_only_rmse"] == 3.0


def test_summarize_oof_dataframe_reports_pooled_per_well_and_buckets() -> None:
    df = pd.DataFrame(
        {
            "well_id": ["a", "a", "b", "b"],
            "id": ["a_1", "a_2", "b_1", "b_2"],
            "tvt_true": [10.0, 12.0, 20.0, 24.0],
            "last_known_tvt": [10.0, 10.0, 20.0, 20.0],
            "base_no_c0": [11.0, 11.0, 21.0, 21.0],
            "base_with_c0": [10.0, 13.0, 20.0, 26.0],
            "model_full": [10.0, 12.0, 20.0, 24.0],
            "model_no_direct": [10.0, 13.0, 20.0, 26.0],
            "model_no_s_pred": [11.0, 12.0, 22.0, 24.0],
            "hidden_length": [2, 2, 8, 8],
            "GR_valid_frac": [0.8, 0.8, 0.2, 0.2],
            "abs_c0": [0.1, 0.1, 0.4, 0.4],
            "event_count": [0, 0, 3, 3],
            "base_only_rmse": [1.0, 1.0, 2.0, 2.0],
        }
    )

    summary = summarize_oof_dataframe(df, n_bins=2)

    assert summary["pooled_rmse"]["model_full"] == 0.0
    assert summary["pooled_rmse"]["last_known_tvt"] == np.sqrt((0.0 + 4.0 + 0.0 + 16.0) / 4.0)
    assert {row["well_id"] for row in summary["per_well_rmse"]} == {"a", "b"}
    assert "hidden_length" in summary["bucket_rmse"]
    assert len(summary["bucket_rmse"]["hidden_length"]) == 2
    first_bucket = summary["bucket_rmse"]["hidden_length"][0]
    assert "model_full" in first_bucket["rmse"]
    assert first_bucket["n_wells"] == 1

    one_bucket = summarize_oof_dataframe(df, n_bins=1)["bucket_rmse"]["hidden_length"][0]
    assert one_bucket["rmse"]["base_with_c0"] == np.sqrt((0.0 + 1.0 + 0.0 + 4.0) / 4.0)
