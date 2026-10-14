"""Loader for the out-of-distribution YouTube Hinglish sentiment dataset.

``shae2977/hinglish-youtube-sentiments-dataset`` (CC-BY-4.0) contains 3,190
manually annotated YouTube comments with columns ``video_id, comment, likes,
sentiment``. It is used *only* to measure cross-dataset generalisation -- it must
never be used for training or model selection, so ``prepare.py`` additionally
guards against any text overlap with the SentiMix splits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RAW_DIR

RELATIVE_PATH = "shae2977/yt_hinglish_comments_dataset.csv"


def load_shae(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return the YouTube Hinglish dataset as ``[uid, text, label]``."""
    path = raw_dir / RELATIVE_PATH
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame = frame.rename(columns={"comment": "text", "sentiment": "label"})
    frame["uid"] = [f"shae-{i:05d}" for i in range(len(frame))]
    frame["label"] = frame["label"].str.strip().str.lower()
    return frame[["uid", "text", "label"]]
