"""Tests for evaluation figures."""

import pandas as pd

from hinglish_mlops.figures import confusion_figure
from hinglish_mlops.figures import main as figures_main
from hinglish_mlops.figures import run as figures_run
from hinglish_mlops.modeling import BASELINE_NAME, build_model, save_model

CORPUS = [
    ("bahut accha hai ye product pasand aaya", "positive"),
    ("kaam badhiya karta hai ye sahi hai", "positive"),
    ("bilkul bekaar hai ye product waste", "negative"),
    ("kharab nikla bilkul mat lena ise", "negative"),
    ("theek hai average product chalta hai", "neutral"),
    ("normal hai kuch khaas nahi hai ye", "neutral"),
]


def _prepare(tmp_path):
    model_dir = tmp_path / "models"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    texts, labels = [], []
    for _ in range(10):
        for text, label in CORPUS:
            texts.append(text)
            labels.append(label)

    pipeline = build_model(BASELINE_NAME)
    pipeline.fit(texts, labels)
    save_model(pipeline, model_dir / f"{BASELINE_NAME}.joblib")

    frame = pd.DataFrame(
        {
            "uid": [str(i) for i in range(len(labels))],
            "text": texts,
            "text_clean": texts,
            "label": labels,
        }
    )
    frame.to_csv(processed_dir / "sentimix_test.csv", index=False)
    return model_dir, processed_dir


def test_confusion_figure_writes_png(tmp_path):
    y_true = pd.Series(["positive", "negative", "neutral", "positive"])
    y_pred = ["positive", "negative", "neutral", "negative"]

    out_path = confusion_figure(y_true, y_pred, tmp_path / "cm.png", title="demo")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_run_renders_figure_for_trained_model(tmp_path):
    model_dir, processed_dir = _prepare(tmp_path)
    figures_dir = tmp_path / "figures"

    out_path = figures_run(BASELINE_NAME, "test", model_dir, processed_dir, figures_dir)

    assert out_path == figures_dir / f"{BASELINE_NAME}_confusion_matrix.png"
    assert out_path.exists()


def test_main_reports_missing_model(tmp_path, capsys):
    return_code = figures_main(
        [
            "--model",
            BASELINE_NAME,
            "--model-dir",
            str(tmp_path),
            "--processed-dir",
            str(tmp_path),
        ]
    )

    assert return_code == 2
    assert "error:" in capsys.readouterr().err


def test_main_reports_unknown_split(tmp_path, capsys):
    model_dir, processed_dir = _prepare(tmp_path)

    return_code = figures_main(
        [
            "--model",
            BASELINE_NAME,
            "--split",
            "ghost",
            "--model-dir",
            str(model_dir),
            "--processed-dir",
            str(processed_dir),
        ]
    )

    assert return_code == 2
    assert "error:" in capsys.readouterr().err
