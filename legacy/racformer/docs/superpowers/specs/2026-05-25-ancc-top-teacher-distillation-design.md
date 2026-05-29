# ANCC Top Teacher Distillation Design

## Goal

Use train-only ANCC formation surfaces as auxiliary teacher labels during training without adding any train-only surface as an input feature.

## Design

The model input remains `MD, X, Y, Z, GR, TVT_input` and derived safe features. During training, `dataset.py` reads `ANCC` only when `TVT` is present and builds step-level teacher targets from `dANCC` using the same `row_to_step` mapping as the model tokens.

The teacher is split into two heads:

- `top_event_step`: binary label for whether `abs(mean_dANCC_step) > top_teacher_eps`.
- `top_state_step`: direction class, `0=flat`, `1=top moving down`, `2=top moving up`, with `-100` for unavailable labels.

`RACDataset.__getitem__` returns `top_event_step`, `top_state_step`, and `top_teacher_mask`, padded to `max_seq_len`.

`RACFormer` keeps the old binary top-state head as the event head and adds a 3-class direction head. `RACLoss` adds masked `BCEWithLogits` for event and masked `CrossEntropy` for direction only on event steps. Default weights start at `w_top_event=0.03` and `w_top_dir=0.03`.

## Non-Goals

ANCC and other formation surfaces are not fed into test-time features. This patch does not run the 0.01/0.03/0.07 sweep; it only makes the weights configurable.

## Testing

Unit tests cover ANCC-to-step teacher aggregation, dataset batch keys and padding, model output shapes, and masked top-event/top-direction loss behavior.
