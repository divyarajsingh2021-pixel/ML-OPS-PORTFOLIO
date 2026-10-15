"""Tests for the FastAPI inference service."""

import pytest
from fastapi.testclient import TestClient

from hinglish_mlops.api import ApiSettings, create_app
from hinglish_mlops.config import LABEL2ID, LABELS
from hinglish_mlops.modeling import BASELINE_NAME, IMPROVED_NAME, build_model, save_model

CORPUS = [
    ("bahut accha hai ye product mujhe pasand aaya", "positive"),
    ("kaam badhiya karta hai ye bilkul sahi hai", "positive"),
    ("bilkul bekaar hai ye product waste money", "negative"),
    ("kharab nikla bilkul mat lena ise", "negative"),
    ("theek hai average product chalta hai", "neutral"),
    ("normal hai kuch khaas nahi hai ye", "neutral"),
]


def _fit(model_name: str, target_dir) -> str:
    texts, labels = [], []
    for _ in range(15):
        for text, label in CORPUS:
            texts.append(text)
            labels.append(label)
    pipeline = build_model(model_name)
    pipeline.fit(texts, labels)
    save_model(pipeline, target_dir / f"{model_name}.joblib")
    return model_name


@pytest.fixture()
def app(tmp_path):
    name = _fit(BASELINE_NAME, tmp_path)
    return create_app(ApiSettings(model_dir=tmp_path, model_name=name))


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    body = client.get("/").json()
    assert body["service"] == "Hinglish Sentiment API"
    assert body["docs"] == "/docs"


def test_health_reports_model_and_labels(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == BASELINE_NAME
    assert body["model_ready"] is True
    assert body["labels"] == list(LABELS)


def test_predict_returns_label_scores_and_model(client):
    body = client.post("/predict", json={"text": "bahut accha hai ye product"}).json()
    assert body["model"] == BASELINE_NAME
    assert body["label"] in LABELS
    assert body["label_id"] == LABEL2ID[body["label"]]
    assert set(body["scores"]) == set(LABELS)
    assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_rejects_blank_text(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_reports_missing_artifact(tmp_path):
    missing = create_app(ApiSettings(model_dir=tmp_path, model_name=BASELINE_NAME))
    with TestClient(missing) as test_client:
        response = test_client.post("/predict", json={"text": "hello duniya"})
    assert response.status_code == 503
    assert "not found" in response.json()["detail"]


def test_batch_predict_preserves_order(client):
    texts = ["bahut accha hai ye product", "bilkul bekaar hai ye product"]
    body = client.post("/predict/batch", json={"texts": texts}).json()
    assert body["model"] == BASELINE_NAME
    assert [item["text"] for item in body["predictions"]] == texts
    assert len(body["predictions"]) == 2


def test_batch_predict_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422
    assert client.post("/predict/batch", json={"texts": ["ok", "   "]}).status_code == 422


def test_svc_model_scores_are_probabilistic(tmp_path):
    name = _fit(IMPROVED_NAME, tmp_path)
    with TestClient(create_app(ApiSettings(model_dir=tmp_path, model_name=name))) as test_client:
        body = test_client.post("/predict", json={"text": "bilkul bekaar hai ye product"}).json()
    assert body["model"] == name
    assert set(body["scores"]) == set(LABELS)
    assert sum(body["scores"].values()) == pytest.approx(1.0, abs=1e-6)


def test_serve_cli_builds_app(monkeypatch):
    import uvicorn
    from fastapi import FastAPI

    from hinglish_mlops import api

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, host, port: seen.update(app=app, host=host, port=port),
    )

    return_code = api.main(["--host", "0.0.0.0", "--port", "9000", "--model-name", BASELINE_NAME])

    assert return_code == 0
    assert seen["host"] == "0.0.0.0"
    assert seen["port"] == 9000
    assert isinstance(seen["app"], FastAPI)
