"""Loaders for the processed dataset splits produced by ``data.prepare``.

The processed CSVs have columns ``uid, text, text_clean, label``. Models consume
``text_clean`` (the normalised text) and predict ``label``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
EVAL_NAME = "shae_eval"
REQUIRED_COLUMNS: tuple[str, ...] = ("uid", "text_clean", "label")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `inv download-data` and `inv prepare-data` first."
        )
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    return frame


def load_split(name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load one SentiMix split (``train``/``val``/``test``)."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
    return _read(processed_dir / f"sentimix_{name}.csv")


def load_eval(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load the out-of-distribution evaluation set."""
    return _read(processed_dir / f"{EVAL_NAME}.csv")


def load_all(processed_dir: Path = PROCESSED_DIR) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ``({train, val, test}, eval)``."""
    splits = {name: load_split(name, processed_dir) for name in SPLIT_NAMES}
    return splits, load_eval(processed_dir)
