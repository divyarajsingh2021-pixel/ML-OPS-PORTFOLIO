"""Project-wide paths and constants.

All paths are derived from the repository root so that scripts work regardless
of the current working directory. Nothing here touches the network or disk.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

MANIFEST_PATH: Path = DATA_DIR / "manifest.sha256"
DATA_CARD_PATH: Path = REPORTS_DIR / "data_card.md"

# Canonical label order used everywhere (class ids follow this order).
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

RANDOM_SEED: int = 42
