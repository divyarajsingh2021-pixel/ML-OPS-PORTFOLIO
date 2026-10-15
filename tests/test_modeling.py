"""Tests for model construction and persistence."""

from hinglish_mlops.modeling import build_baseline_pipeline, load_model, save_model


def test_baseline_pipeline_fits_and_predicts():
    texts = ["good good", "bad bad", "good bad"]
    labels = ["positive", "negative", "positive"]
    pipeline = build_baseline_pipeline()
    pipeline.fit(texts, labels)
    assert list(pipeline.predict(["good good"])) == ["positive"]


def test_save_load_roundtrip(tmp_path):
    pipeline = build_baseline_pipeline()
    pipeline.fit(["good good", "good bad", "bad bad"], ["positive", "positive", "negative"])

    path = save_model(pipeline, tmp_path / "nested" / "model.joblib")
    reloaded = load_model(path)

    probes = ["good good", "good bad", "bad bad"]
    assert list(reloaded.predict(probes)) == list(pipeline.predict(probes))
