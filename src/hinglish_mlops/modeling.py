"""Model builders and persistence.

Two models are registered:

* ``baseline_tfidf_logreg`` -- word-level TF-IDF + logistic regression.
* ``improved_tfidf_svc``    -- word- and character-level TF-IDF + linear SVM.

Keeping the vectorisers inside the pipeline guarantees that the exact same text
preprocessing is applied at fit and inference time. ``MODEL_BUILDERS`` maps a
model name to its builder so training, serving and monitoring share one source
of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import RANDOM_SEED

BASELINE_NAME = "baseline_tfidf_logreg"
IMPROVED_NAME = "improved_tfidf_svc"


def word_tfidf(**overrides) -> TfidfVectorizer:
    """Word-level TF-IDF vectoriser (unigrams + bigrams by default)."""
    params: dict[str, object] = {
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    }
    params.update(overrides)
    return TfidfVectorizer(**params)


def char_tfidf(**overrides) -> TfidfVectorizer:
    """Character n-gram TF-IDF vectoriser (robust to Hinglish spelling noise)."""
    params: dict[str, object] = {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "min_df": 2,
        "sublinear_tf": True,
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


# Chosen on the validation split (see reports/model_comparison.md); a small C
# strongly reduces the overfitting of char n-grams on this short-text task.
IMPROVED_SVC_C = 0.05


def build_improved_pipeline(random_state: int = RANDOM_SEED) -> Pipeline:
    """TF-IDF (word + char) + linear SVM."""
    features = FeatureUnion(
        [
            ("word", word_tfidf()),
            ("char", char_tfidf()),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LinearSVC(C=IMPROVED_SVC_C, random_state=random_state, max_iter=5000)),
        ]
    )


MODEL_BUILDERS: dict[str, Callable[[int], Pipeline]] = {
    BASELINE_NAME: build_baseline_pipeline,
    IMPROVED_NAME: build_improved_pipeline,
}


def build_model(name: str, random_state: int = RANDOM_SEED) -> Pipeline:
    """Build a registered model by name."""
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; available: {sorted(MODEL_BUILDERS)}") from None
    return builder(random_state)


def save_model(model: object, path: Path) -> Path:
    """Persist a fitted model with joblib, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> object:
    """Load a model previously written by :func:`save_model`."""
    return joblib.load(path)
