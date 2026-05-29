"""MTPNet — Offset-State Decoder for TVT prediction.

Architecture:
    candidate generator  : K-segment offset posterior (LightGBM multiclass)
    state prior          : top-event / top-direction teacher
    decoder              : DP/beam over offset states
    selector             : path scorer + dictionary prior
    materializer         : TVT = anchor + cumsum(-dZ + decoded_offset_state)

Modules:
    offsets   — canonical physical integrators (THE source of truth)
    data      — OffsetSample dataclass + data loader
    oracle    — oracle ceiling computation (global + K-segment)
    metrics   — row RMSE, top-k oracle, scoreboard
    features  — well-level + segment feature engineering
    models    — LightGBM/CatBoost offset bin classifier
    decode    — Viterbi DP + beam search decoder
    eval      — OOF evaluation pipeline
"""
from __future__ import annotations
