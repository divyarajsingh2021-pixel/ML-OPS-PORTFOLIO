"""FastAPI inference service for the Hinglish sentiment model.

Endpoints:

* ``GET  /health``         -- liveness + which model is configured.
* ``POST /predict``        -- classify one text.
* ``POST /predict/batch``  -- classify a list of texts in one call.

The model is loaded lazily on first use and cached, so starting the server is
instant and tests can point the app at a fixture artifact. Configuration comes
from environment variables with the ``HINGLISH_`` prefix
(``HINGLISH_MODEL_NAME``, ``HINGLISH_MODEL_DIR``).

CLI::

    uvicorn hinglish_mlops.api:app --port 8000
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import __version__
from .config import LABEL2ID, LABELS, MODELS_DIR
from .modeling import BASELINE_NAME, load_model

MAX_TEXT_CHARS = 5000
MAX_BATCH_SIZE = 128

# A bounded, non-empty text string; blank/oversized input is rejected by the API.
TextStr = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]


def _clean_text(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("text must not be blank")
    return stripped


class ApiSettings(BaseSettings):
    """Service configuration, overridable through ``HINGLISH_*`` env vars."""

    model_config = SettingsConfigDict(env_prefix="hinglish_", extra="ignore")

    model_name: str = BASELINE_NAME
    model_dir: Path = MODELS_DIR


class PredictIn(BaseModel):
    text: TextStr

    @field_validator("text")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        return _clean_text(value)


class PredictBatchIn(BaseModel):
    texts: list[TextStr] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def _reject_blank_items(cls, values: list[str]) -> list[str]:
        return [_clean_text(value) for value in values]


class Prediction(BaseModel):
    text: str
    label: str
    label_id: int
    scores: dict[str, float]


class PredictOut(Prediction):
    model: str


class PredictBatchOut(BaseModel):
    model: str
    predictions: list[Prediction]


def softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (used for models without ``predict_proba``)."""
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def class_scores(pipeline: Any, text: str) -> dict[str, float]:
    """Per-class scores, falling back to softmax over SVM decision values."""
    proba = getattr(pipeline, "predict_proba", None)
    if callable(proba):
        values = np.asarray(proba([text])[0], dtype=float)
    else:
        raw = np.asarray(pipeline.decision_function([text]), dtype=float)
        if len(pipeline.classes_) == 2 and raw.size == 1:
            positive = float(softmax(np.array([0.0, float(raw.ravel()[0])]))[1])
            values = np.array([1.0 - positive, positive])
        else:
            values = softmax(raw.ravel())
    return {str(cls): float(score) for cls, score in zip(pipeline.classes_, values, strict=True)}


class ModelStore:
    """Thread-safe lazy loader for the configured model artifact."""

    def __init__(self, model_dir: Path, model_name: str) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self._lock = threading.Lock()
        self._pipeline: Any | None = None

    @property
    def artifact_path(self) -> Path:
        return self.model_dir / f"{self.model_name}.joblib"

    def get(self) -> Any:
        with self._lock:
            if self._pipeline is None:
                if not self.artifact_path.exists():
                    raise FileNotFoundError(
                        f"model artifact not found: {self.artifact_path} "
                        "(train it first: `inv train`)"
                    )
                self._pipeline = load_model(self.artifact_path)
            return self._pipeline


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Build the FastAPI application (a fresh instance per ``settings``)."""
    settings = settings or ApiSettings()
    store = ModelStore(settings.model_dir, settings.model_name)

    app = FastAPI(
        title="Hinglish Sentiment API",
        version=__version__,
        description="3-class (negative/neutral/positive) sentiment for Roman-script Hinglish text.",
    )
    app.state.store = store
    app.state.settings = settings

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"service": app.title, "version": __version__, "docs": "/docs"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model": settings.model_name,
            "model_ready": store.artifact_path.exists(),
            "labels": list(LABELS),
        }

    @app.post("/predict", response_model=PredictOut)
    def predict(payload: PredictIn) -> PredictOut:
        prediction = _predict(store, payload.text)
        return PredictOut(**prediction.model_dump(), model=settings.model_name)

    @app.post("/predict/batch", response_model=PredictBatchOut)
    def predict_batch(payload: PredictBatchIn) -> PredictBatchOut:
        predictions = [_predict(store, text) for text in payload.texts]
        return PredictBatchOut(model=settings.model_name, predictions=predictions)

    return app


def _predict(store: ModelStore, text: str) -> Prediction:
    try:
        pipeline = store.get()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    scores = class_scores(pipeline, text)
    label = str(pipeline.predict([text])[0])
    return Prediction(text=text, label=label, label_id=LABEL2ID.get(label, -1), scores=scores)


app = create_app()


def main(argv: list[str] | None = None) -> int:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the Hinglish sentiment API.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--model-name", default=None, help="override HINGLISH_MODEL_NAME")
    args = parser.parse_args(argv)

    settings = ApiSettings(model_name=args.model_name) if args.model_name else ApiSettings()
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
