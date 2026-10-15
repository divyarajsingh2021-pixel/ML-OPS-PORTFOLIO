"""Tests for MLflow experiment tracking."""

import mlflow

from hinglish_mlops import tracking


def test_scalar_params_drops_non_primitives():
    class Fake:
        def get_params(self, deep=True):
            return {"C": 1.0, "name": "tfidf", "ngram_range": (1, 2), "clf": object(), "flag": True}

    assert tracking._scalar_params(Fake()) == {"C": 1.0, "name": "tfidf", "flag": True}


def test_scalar_metrics_drops_nested_entries():
    metrics = {"accuracy": 0.9, "per_class": {"x": {"f1": 0.1}}}
    assert tracking._scalar_metrics(metrics) == {"accuracy": 0.9}


def test_log_run_writes_to_local_store(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = tracking.log_run(
        model_name="unit-model",
        metrics={
            "train": {"accuracy": 0.9, "macro_f1": 0.9, "per_class": {"x": {}}},
            "val": {"accuracy": 0.5, "macro_f1": 0.4},
        },
        tracking_uri=uri,
    )
    assert run_id

    mlflow.set_tracking_uri(uri)
    run = mlflow.MlflowClient().get_run(run_id)
    assert run.data.tags["model"] == "unit-model"
    assert run.data.metrics["train_accuracy"] == 0.9
    assert run.data.metrics["val_macro_f1"] == 0.4
    # nested structures must not be logged as metrics
    assert "train_per_class" not in run.data.metrics


def test_log_to_mlflow_reports_failure(monkeypatch, capsys):
    from hinglish_mlops import train

    def boom(**kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(tracking, "log_run", boom)
    result = train._log_to_mlflow("m", {}, None, None, None)
    assert result is None
    assert "MLflow tracking skipped" in capsys.readouterr().out
