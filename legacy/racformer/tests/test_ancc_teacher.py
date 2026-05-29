import numpy as np
import pandas as pd
import torch
from racformer.config import K_SEG, ROWS_PER_STEP
from racformer.dataset import RACDataset, WellSample, _build_top_teacher_steps


def test_build_top_teacher_steps_from_ancc_direction_events() -> None:
    horizontal = pd.DataFrame(
        {
            "ANCC": [100.0, 100.0, 100.3, 100.7, 100.2, 99.8],
        }
    )
    row_to_step = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)

    top_state, top_event, top_mask = _build_top_teacher_steps(
        horizontal=horizontal,
        row_to_step=row_to_step,
        n_steps=3,
        eps=0.10,
        has_tvt=True,
    )

    np.testing.assert_array_equal(top_state, np.array([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(top_event, np.array([0.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(top_mask, np.array([True, True, True]))


def test_build_top_teacher_steps_missing_ancc_is_ignored() -> None:
    horizontal = pd.DataFrame({"Z": [1.0, 2.0, 3.0]})

    top_state, top_event, top_mask = _build_top_teacher_steps(
        horizontal=horizontal,
        row_to_step=np.array([0, 0, 1], dtype=np.int32),
        n_steps=2,
        eps=0.10,
        has_tvt=True,
    )

    np.testing.assert_array_equal(top_state, np.array([-100, -100], dtype=np.int64))
    np.testing.assert_allclose(top_event, np.array([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(top_mask, np.array([False, False]))


def test_racdataset_returns_padded_top_teacher_keys() -> None:
    sample = WellSample(
        well_id="well_a",
        features=np.zeros((2, 74), dtype=np.float32),
        region_ids=np.array([1, 3], dtype=np.int32),
        hidden_mask=np.array([False, True]),
        base_tvt_rows=np.array([10.0, 11.0], dtype=np.float32),
        tvt_rows=np.array([10.0, 11.0], dtype=np.float32),
        tvt_input_rows=np.array([10.0, np.nan], dtype=np.float32),
        z_rows=np.array([100.0, 99.0], dtype=np.float32),
        anchor_row=0,
        anchor_step=0,
        n_hidden_rows=1,
        n_rows=2,
        row_to_step=np.array([0, 1], dtype=np.int32),
        step_offset=0,
        seq_len=2,
        s_star=np.zeros(K_SEG, dtype=np.float32),
        dC_forward=np.zeros(1, dtype=np.float32),
        top_state_step=np.array([0, 1], dtype=np.int64),
        top_event_step=np.array([0.0, 1.0], dtype=np.float32),
        top_teacher_mask=np.array([True, True]),
        row_ids=["well_a_0", "well_a_1"],
        hidden_row_ids=["well_a_1"],
        last_known_tvt=10.0,
        last_known_z=100.0,
        c0=0.0,
    )

    item = RACDataset([sample], max_seq_len=4, rows_per_step=ROWS_PER_STEP)[0]

    assert torch.equal(item["top_state_step"], torch.tensor([0, 1, -100, -100]))
    assert torch.allclose(item["top_event_step"], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(item["top_teacher_mask"], torch.tensor([True, True, False, False]))
