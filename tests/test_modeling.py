"""Tests for model construction and persistence."""

import pytest

from hinglish_mlops.modeling import (
    BASELINE_NAME,
    IMPROVED_NAME,
    build_baseline_pipeline,
    build_improved_pipeline,
    build_model,
    load_model,
    save_model,
)


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


def test_improved_pipeline_uses_word_and_char_features():
    texts = ["good good", "good bad", "bad bad"]
    labels = ["positive", "positive", "negative"]
    pipeline = build_improved_pipeline()
    pipeline.fit(texts, labels)
    feature_names = {name for name, _ in pipeline.named_steps["features"].transformer_list}
    assert feature_names == {"word", "char"}
    assert list(pipeline.predict(["bad bad"])) == ["negative"]


def test_build_model_registry():
    assert build_model(BASELINE_NAME).named_steps["clf"].__class__.__name__ == "LogisticRegression"
    assert build_model(IMPROVED_NAME).named_steps["clf"].__class__.__name__ == "LinearSVC"
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("does-not-exist")
