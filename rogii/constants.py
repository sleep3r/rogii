from __future__ import annotations

from pathlib import Path

KAGGLE_INPUT_DIR = Path("/kaggle/input/rogii-wellbore-geology-prediction")
FORMATION_ORDER = {
    "ANCC": 0,
    "ASTNU": 1,
    "ASTNL": 2,
    "EGFDU": 3,
    "EGFDL": 4,
    "BUDA": 5,
}
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
