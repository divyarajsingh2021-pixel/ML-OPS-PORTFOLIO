"""Integration test for the prepare pipeline using tiny in-memory fixtures."""

import json

import pandas as pd

from hinglish_mlops.data.prepare import prepare

TRAIN = (
    "meta\t1\tpositive\ngood\tEng\nbetter\tEng\n\n"
    "meta\t2\tnegative\nbad\tHin\n\n"
    "meta\t3\tpositive\ngood\tEng\nbetter\tEng\n\n"
)
VAL = "meta\t10\tneutral\nok\tEng\nok\tEng\n\nmeta\t11\tpositive\ngood\tEng\nbetter\tEng\n\n"
TEST = "meta\t20\tnegative\nworst\tHin\n\nmeta\t21\tneutral\nok\tEng\nok\tEng\n\n"
LABELS = "Uid,Sentiment\n20,negative\n21,neutral\n"
SHAE = (
    "video_id,comment,likes,sentiment\nv1,good better,0,Positive\nv2,brand new comment,1,Negative\n"
)


def _build_raw(tmp_path):
    sentimix = tmp_path / "sentimix"
    sentimix.mkdir(parents=True)
    (sentimix / "train.txt").write_text(TRAIN, encoding="utf-8")
    (sentimix / "dev.txt").write_text(VAL, encoding="utf-8")
    (sentimix / "test.txt").write_text(TEST, encoding="utf-8")
    (sentimix / "test_labels.csv").write_text(LABELS, encoding="utf-8")
    shae = tmp_path / "shae2977"
    shae.mkdir(parents=True)
    (shae / "yt_hinglish_comments_dataset.csv").write_text(SHAE, encoding="utf-8")


def test_prepare_writes_splits_and_enforces_leakage_guard(tmp_path):
    raw, processed, reports = tmp_path / "raw", tmp_path / "processed", tmp_path / "reports"
    _build_raw(raw)

    stats = prepare(raw, processed, reports)

    train = pd.read_csv(processed / "sentimix_train.csv")
    val = pd.read_csv(processed / "sentimix_val.csv")
    test = pd.read_csv(processed / "sentimix_test.csv")
    shae = pd.read_csv(processed / "shae_eval.csv")

    # intra-split duplicate ("good better") collapsed in train
    assert stats["filters"]["train"]["duplicate_removed"] == 1
    assert len(train) == 2
    # val/test share text with earlier splits, which must be dropped
    assert stats["filters"]["val"]["cross_split_removed"] == 1
    assert stats["filters"]["test"]["cross_split_removed"] == 1
    assert len(val) == 1 and len(test) == 1
    # eval row overlapping SentiMix is dropped, leaving the non-overlapping one
    assert stats["filters"]["shae_eval"]["overlap_with_sentimix_removed"] == 1
    assert len(shae) == 1

    # no cleaned text shared across the three SentiMix splits
    sets = [set(frame["text_clean"]) for frame in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])

    # labels are restricted to the canonical set
    assert set(train["label"]) <= {"negative", "neutral", "positive"}


def test_prepare_writes_stats_and_data_card(tmp_path):
    raw, processed, reports = tmp_path / "raw", tmp_path / "processed", tmp_path / "reports"
    _build_raw(raw)

    prepare(raw, processed, reports)

    payload = json.loads((reports / "data_stats.json").read_text(encoding="utf-8"))
    assert set(payload["splits"]) == {"train", "val", "test", "shae_eval"}
    card = (reports / "data_card.md").read_text(encoding="utf-8")
    assert "# Data card" in card
    assert "Leakage guard" in card
