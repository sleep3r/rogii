"""Formation top spatial imputation from train neighbours.

At train time the horizontal CSV contains ANCC, ASTNU, ASTNL, EGFDU, EGFDL,
BUDA columns.  At test time these are absent.  This module fits spatial surface
models from train data and applies them to test wells.

Usage
-----
ctx = TopContext.build_from_train(train_well_list, cfg)
ctx.save(path)
ctx = TopContext.load(path)

per_row_feats = ctx.impute_formations(df_horizontal)   # works on test
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


class TopContext:
    """Fits and applies spatial formation-top imputation models."""

    def __init__(
        self,
        train_tops_df: pd.DataFrame | None = None,
        formation_cols: list[str] | None = None,
        k_neighbors: int = 15,
    ) -> None:
        self.formation_cols = formation_cols or FORMATION_COLS
        self.scalers: dict[str, StandardScaler] = {}
        self.models: dict[str, KNeighborsRegressor] = {}
        self.residual_std: dict[str, float] = {}
        self.global_means: dict[str, float] = {}
        self.train_X: np.ndarray | None = None
        self.fitted = False
        if train_tops_df is not None:
            self.fit(train_tops_df, k_neighbors=k_neighbors)

    # ------------------------------------------------------------------
    # Build from train wells
    # ------------------------------------------------------------------

    @classmethod
    def build_from_train(
        cls,
        train_wells: list[dict],
        k_neighbors: int = 15,
        verbose: bool = False,
    ) -> TopContext:
        """
        train_wells — list of dicts, each with keys:
            'X0', 'Y0', 'Z0'  — well head coordinates
            'formations'       — dict[str, float] e.g. {'ANCC': -9395.0, ...}
        """
        ctx = cls(formation_cols=FORMATION_COLS)
        records = []
        for w in train_wells:
            row = {"X": w["X0"], "Y": w["Y0"]}
            for col in FORMATION_COLS:
                row[col] = w["formations"].get(col, np.nan)
            records.append(row)

        ctx.fit(pd.DataFrame(records), k_neighbors=k_neighbors)
        if verbose:
            print(f"TopContext: fitted {len(ctx.models)} formation models from {len(train_wells)} wells")
        return ctx

    def fit(self, df: pd.DataFrame, k_neighbors: int = 15) -> None:
        X_coords = df[["X", "Y"]].to_numpy(dtype=np.float64)
        self.train_X = X_coords
        for col in self.formation_cols:
            y = df[col].to_numpy(dtype=np.float64)
            valid = np.isfinite(y)
            if valid.any():
                self.global_means[col] = float(np.nanmean(y))
            if valid.sum() < 2:
                continue

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_coords[valid])
            model = KNeighborsRegressor(n_neighbors=min(k_neighbors, valid.sum()), weights="distance")
            model.fit(Xs, y[valid])
            pred_train = model.predict(Xs)
            self.scalers[col] = scaler
            self.models[col] = model
            self.residual_std[col] = float(np.std(pred_train - y[valid]) + 1e-6)

        self.fitted = True

    # ------------------------------------------------------------------
    # Impute for a single well
    # ------------------------------------------------------------------

    def impute_formations(
        self, df: pd.DataFrame, use_actual_if_present: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Impute formation top depths for every row in *df* (train or test).

        By default this never reads train-only formation columns from *df*.
        Set use_actual_if_present=True only for diagnostics, not model features.

        Returns dict[str, np.ndarray] — one entry per formation.
        """
        n = len(df)
        X_coords = df[["X", "Y"]].to_numpy(dtype=np.float64)
        result: dict[str, np.ndarray] = {}

        for col in self.formation_cols:
            pred = np.full(n, np.nan, dtype=np.float32)

            if col in self.models:
                scaler = self.scalers[col]
                model = self.models[col]
                Xs = scaler.transform(X_coords)
                pred_all = model.predict(Xs).astype(np.float32)
                pred[:] = pred_all

            elif col in self.global_means:
                pred[:] = self.global_means[col]

            if use_actual_if_present and col in df.columns:
                actual = df[col].to_numpy(dtype=np.float32)
                valid = np.isfinite(actual)
                pred[valid] = actual[valid]

            result[col] = pred
            result[f"{col}_uncertainty"] = np.full(n, self.residual_std.get(col, 100.0), dtype=np.float32)

        # Compute TVT-relative offsets: Z_row - top_k  (for each formation)
        z = df["Z"].to_numpy(dtype=np.float32)
        for col in self.formation_cols:
            result[f"z_minus_{col.lower()}"] = z - result[col]

        # Formation interval widths
        for i in range(len(self.formation_cols) - 1):
            a, b = self.formation_cols[i], self.formation_cols[i + 1]
            result[f"interval_{a}_{b}"] = np.abs(result[b] - result[a]).astype(np.float32)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> TopContext:
        with open(path, "rb") as f:
            return pickle.load(f)
