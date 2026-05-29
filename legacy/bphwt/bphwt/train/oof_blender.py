"""OOF blender: combine multiple OOF prediction sources using Ridge/LightGBM."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OOFBlender:
    """
    Meta-blender trained on OOF predictions from multiple models.
    Uses per-row features + predictions to output a residual correction.

    Constraint: blender is trained only on OOF rows (no in-fold leakage).
    Final output: TVT_final = TVT_nn_opt + clip(meta_residual, -15, +15)
    """

    def __init__(self, clip_residual: float = 15.0) -> None:
        self.clip_residual = clip_residual
        self.model = None
        self.feature_names: list[str] = []

    def _build_features(self, oof_df: pd.DataFrame) -> np.ndarray:
        """Build blender input features from OOF dataframe."""
        feat_cols = [
            c
            for c in oof_df.columns
            if c.startswith("pred_")
            or c
            in [
                "gr_valid_ratio",
                "hidden_ratio",
                "dist_head_norm",
                "tvt_base",
                "nn_sigma",
                "hmm_entropy",
                "neighbor_corr",
                "typewell_freq",
            ]
        ]
        self.feature_names = feat_cols
        return oof_df[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)

    def fit(self, oof_df: pd.DataFrame, primary_col: str = "pred_nn_opt") -> None:
        """Fit blender on OOF predictions. Target = TVT_true - primary_pred."""
        hidden = oof_df[oof_df["is_hidden"] == 1].copy()
        if len(hidden) == 0:
            logger.warning("No hidden rows in OOF, blender not fitted")
            return

        X = self._build_features(hidden)
        y = (hidden["tvt_true"] - hidden[primary_col]).to_numpy(dtype=np.float32)

        try:
            from sklearn.linear_model import Ridge

            self.model = Ridge(alpha=10.0)
            self.model.fit(X, y)
            preds = self.model.predict(X)
            rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
            logger.info(f"Blender fitted (Ridge): in-sample residual RMSE = {rmse:.4f}")
        except Exception as e:
            logger.warning(f"Blender fitting failed: {e}")
            self.model = None

    def predict(self, df: pd.DataFrame, primary_col: str = "pred_nn_opt") -> np.ndarray:
        """Return corrected TVT predictions."""
        base = df[primary_col].to_numpy(dtype=np.float32)
        if self.model is None or not self.feature_names:
            return base

        X = df[self.feature_names].fillna(0.0).to_numpy(dtype=np.float32)
        try:
            correction = self.model.predict(X).astype(np.float32)
            correction = np.clip(correction, -self.clip_residual, self.clip_residual)
            return base + correction
        except Exception:
            return base

    def save(self, path: Path | str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "feature_names": self.feature_names, "clip": self.clip_residual}, f
            )

    @classmethod
    def load(cls, path: Path | str) -> OOFBlender:
        import pickle

        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(clip_residual=d["clip"])
        obj.model = d["model"]
        obj.feature_names = d["feature_names"]
        return obj
