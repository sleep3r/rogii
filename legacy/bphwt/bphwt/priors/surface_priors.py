"""Spatial formation-surface models for test-time formation top imputation.

Thin wrapper that trains per-formation surface models on training wells
and exposes them through TopContext (see features/top_context.py).
"""

from __future__ import annotations

# Re-export the TopContext for convenience so importers can do:
#   from bphwt.priors.surface_priors import TopContext
from bphwt.features.top_context import TopContext


def build_top_context(train_wells: list[dict], k_neighbors: int = 15, verbose: bool = False) -> TopContext:
    """Build spatial formation-top context from train well records.

    Each record needs: X0, Y0, formations (dict col->value).
    """
    return TopContext.build_from_train(train_wells, k_neighbors=k_neighbors, verbose=verbose)
