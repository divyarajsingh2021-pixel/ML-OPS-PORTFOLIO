"""Tests for the processed-split loaders."""

import pandas as pd
import pytest

from hinglish_mlops import dataset


def _write(processed_dir, name, rows):
    processed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["uid", "text", "text_clean", "label"]).to_csv(
        processed_dir / f"{name}.csv", index=False
    )


def test_load_all_returns_splits_and_eval(tmp_path):
    _write(tmp_path, "sentimix_train", [["1", "a", "a", "neutral"]])
    _write(tmp_path, "sentimix_val", [["2", "b", "b", "positive"]])
    _write(tmp_path, "sentimix_test", [["3", "c", "c", "negative"]])
    _write(tmp_path, "shae_eval", [["s1", "d", "d", "positive"]])

    splits, eval_frame = dataset.load_all(tmp_path)

    assert set(splits) == {"train", "val", "test"}
    assert splits["train"]["text_clean"].tolist() == ["a"]
    assert eval_frame["label"].tolist() == ["positive"]


def test_unknown_split_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown split"):
        dataset.load_split("bogus", tmp_path)


def test_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare-data"):
        dataset.load_split("train", tmp_path)


def test_missing_columns_raise(tmp_path):
    (tmp_path / "sentimix_train.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        dataset.load_split("train", tmp_path)
