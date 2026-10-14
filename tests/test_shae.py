"""Unit tests for the out-of-distribution YouTube Hinglish loader."""

from hinglish_mlops.data.shae import load_shae

CSV = "video_id,comment,likes,sentiment\nv1,Nice video,3,Positive\nv2,bakwas hai,0,Negative\n"


def _write_dataset(tmp_path):
    folder = tmp_path / "shae2977"
    folder.mkdir(parents=True)
    (folder / "yt_hinglish_comments_dataset.csv").write_text(CSV, encoding="utf-8")


def test_load_shae_columns_and_labels(tmp_path):
    _write_dataset(tmp_path)
    frame = load_shae(tmp_path)
    assert list(frame.columns) == ["uid", "text", "label"]
    assert frame["text"].tolist() == ["Nice video", "bakwas hai"]
    assert frame["label"].tolist() == ["positive", "negative"]
    assert frame["uid"].tolist() == ["shae-00000", "shae-00001"]
