"""Parser for the SemEval-2020 Task 9 Hinglish (SentiMix) CoNLL files.

The released files are token/LID annotated: each example is a ``meta`` header
line (``meta<TAB>uid<TAB>label``) followed by one line per token
(``token<TAB>LID``) and a blank line. The training/dev files carry the label in
the header; the test file is unlabelled and its labels are shipped separately in
``test_labels.csv`` (``Uid,Sentiment``).

We reconstruct the sentence by joining tokens with a single space. This matches
the widely used community reconstruction and is sufficient for bag-of-words and
character n-gram models. Its main artefact -- URLs and @mentions being split
across tokens (e.g. ``https // t . co / abc``) -- is normalised later in
``clean.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR

SENTIMIX_DIR = RAW_DIR / "sentimix"

_META = "meta"


def parse_conll(text: str) -> list[dict[str, str | None]]:
    """Parse a SentiMix CoNLL file into ``[{uid, label, text}, ...]``.

    ``label`` is ``None`` for unlabelled (test) files.
    """
    records: list[dict[str, str | None]] = []
    uid: str | None = None
    label: str | None = None
    tokens: list[str] = []

    def flush() -> None:
        nonlocal uid, label, tokens
        if uid is not None:
            records.append({"uid": uid, "label": label, "text": " ".join(tokens)})
        uid, label, tokens = None, None, []

    for line in text.splitlines():
        if not line.strip():
            flush()
            continue
        parts = line.split("\t")
        # A header is "meta<TAB><numeric uid>"; a token line may itself be the
        # word "meta", so require the second field to be numeric.
        if parts[0] == _META and len(parts) >= 2 and parts[1].isdigit():
            flush()
            uid = parts[1]
            label = parts[2] if len(parts) > 2 else None
        else:
            tokens.append(parts[0])
    flush()
    return records


def load_test_labels(path: Path) -> dict[str, str]:
    """Read ``Uid,Sentiment`` labels into a ``{uid: label}`` mapping."""
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = (row.get("Uid") or "").strip()
            sentiment = (row.get("Sentiment") or "").strip().lower()
            if uid:
                labels[uid] = sentiment
    return labels


def _to_frame(records: list[dict[str, str | None]]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=["uid", "label", "text"])
    frame["uid"] = frame["uid"].astype(str)
    return frame


def load_split(path: Path) -> pd.DataFrame:
    """Load a labelled or unlabelled SentiMix split (labels from the header)."""
    return _to_frame(parse_conll(path.read_text(encoding="utf-8")))


def load_test_split(text_path: Path, labels_path: Path) -> pd.DataFrame:
    """Load the unlabelled test split and attach labels from ``labels_path``."""
    frame = _to_frame(parse_conll(text_path.read_text(encoding="utf-8")))
    labels = load_test_labels(labels_path)
    frame["label"] = frame["uid"].map(labels)
    return frame


def load_all(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load the official train/val/test splits as a name -> DataFrame mapping."""
    base = raw_dir / "sentimix"
    return {
        "train": load_split(base / "train.txt"),
        "val": load_split(base / "dev.txt"),
        "test": load_test_split(base / "test.txt", base / "test_labels.csv"),
    }
