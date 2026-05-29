from __future__ import annotations

import numpy as np
import pandas as pd

from rogii.formation_b2_inference import B2InferenceConfig, apply_b2_guarded_correction


def test_apply_b2_guarded_correction_fixed_policy_clips_delta() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"w1_{i}" for i in range(4)],
            "well_id": ["w1"] * 4,
            "fold": [1] * 4,
            "TVT": [100.0, 101.0, 102.0, 103.0],
            "GR": [50.0, 51.0, 52.0, 53.0],
            "schema10_oof_raw": [100.0, 100.0, 100.0, 100.0],
            "cand_a": [120.0, 90.0, 103.0, 100.0],
        }
    )
    choices = pd.DataFrame(
        {
            "selector_name": ["fixed_selector"],
            "well_id": ["w1"],
            "candidate_name": ["cand_a"],
        }
    )
    metadata = pd.DataFrame(
        {
            "well_id": ["w1"],
            "candidate_name": ["cand_a"],
            "hidden_rmse": [1.0],
            "b_combined_score": [0.0],
            "a_rank": [1.0],
            "b_rank": [1.0],
            "late_anchor_rmse": [0.0],
            "surface_std": [0.0],
            "b_std": [0.0],
            "roughness": [0.0],
        }
    )
    config = B2InferenceConfig(
        policy_name="fixed_selector__fixed_a0.5_clip10",
        selector="fixed_selector",
        mode="fixed",
        action="none",
        alpha=0.5,
        clip_high=10.0,
    )

    predictions, diagnostics, summary = apply_b2_guarded_correction(
        frame,
        choices,
        metadata,
        config,
    )

    expected = np.array([105.0, 95.0, 101.5, 100.0])
    np.testing.assert_allclose(predictions["b2_guarded_submit"], expected)
    assert predictions["b2_submit_policy_name"].nunique() == 1
    assert predictions["b2_submit_policy_name"].iloc[0] == config.policy_name
    assert diagnostics["selected_candidate"].iloc[0] == "cand_a"
    assert summary["policy_name"] == config.policy_name
