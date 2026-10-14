"""Canonical download sources for the datasets used by the project.

Each key is a path relative to ``data/raw`` and each value is the URL the file
is fetched from. The expected SHA-256 of every file lives in
``data/manifest.sha256`` (the integrity source of truth); see ``download.py``.

Two datasets are used:

* ``sentimix``  -- SemEval-2020 Task 9 Hinglish (Hindi-English) sentiment,
  mirrored on the Hugging Face Hub. Used for training and (official) validation
  and test evaluation.
* ``shae2977``  -- an independently annotated YouTube-comment Hinglish dataset
  (CC-BY-4.0). Used *only* for out-of-distribution cross-dataset evaluation; it
  is never part of training.
"""

from __future__ import annotations

SENTIMIX_BASE = "https://huggingface.co/datasets/RTT1/SentiMix/resolve/main"
SHAE_BASE = (
    "https://huggingface.co/datasets/shae2977/hinglish-youtube-sentiments-dataset/resolve/main"
)

SOURCES: dict[str, str] = {
    # SemEval-2020 Task 9 Hinglish (Roman-script) sentiment, token/LID CoNLL format.
    "sentimix/train.txt": f"{SENTIMIX_BASE}/train_14k_split_conll.txt",
    "sentimix/dev.txt": f"{SENTIMIX_BASE}/dev_3k_split_conll.txt",
    "sentimix/test.txt": f"{SENTIMIX_BASE}/Hindi_test_unalbelled_conll_updated.txt",
    "sentimix/test_labels.csv": f"{SENTIMIX_BASE}/test_labels_hinglish.txt",
    # Out-of-distribution cross-dataset evaluation only (never trained on).
    "shae2977/yt_hinglish_comments_dataset.csv": (f"{SHAE_BASE}/yt_hinglish_comments_dataset.csv"),
}
