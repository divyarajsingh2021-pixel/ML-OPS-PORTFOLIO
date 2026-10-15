"""Model builders and persistence.

The baseline model is a scikit-learn pipeline of a word-level TF-IDF vectoriser
and logistic regression. Keeping the vectoriser inside the pipeline guarantees
that the exact same text preprocessing is applied at fit and inference time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_SEED


def word_tfidf(**overrides) -> TfidfVectorizer:
    """Word-level TF-IDF vectoriser (unigrams + bigrams by default)."""
    params: dict[str, object] = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
        "strip_accents": None,
    }
    params.update(overrides)
    return TfidfVectorizer(**params)


def build_baseline_pipeline(random_state: int = RANDOM_SEED) -> Pipeline:
    """TF-IDF (word) + logistic regression baseline."""
    return Pipeline(
        [
            ("tfidf", word_tfidf()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )


def save_model(model: object, path: Path) -> Path:
    """Persist a fitted model with joblib, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> object:
    """Load a model previously written by :func:`save_model`."""
    return joblib.load(path)
