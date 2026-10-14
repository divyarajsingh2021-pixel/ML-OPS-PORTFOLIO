"""Unit tests for the SentiMix CoNLL parser and label join."""

from hinglish_mlops.data.sentimix import load_test_labels, parse_conll

LABELLED = "meta\t1\tpositive\n@\tO\nhello\tEng\n\nmeta\t2\tnegative\nbad\tHin\n"


def test_parse_labelled_split():
    records = parse_conll(LABELLED)
    assert records == [
        {"uid": "1", "label": "positive", "text": "@ hello"},
        {"uid": "2", "label": "negative", "text": "bad"},
    ]


def test_token_spelling_meta_is_not_a_header():
    text = "meta\t9\tneutral\nmeta\tEng\nstuff\tHin\n\n"
    records = parse_conll(text)
    assert len(records) == 1
    assert records[0]["text"] == "meta stuff"
    assert records[0]["label"] == "neutral"


def test_unlabelled_header_has_none_label():
    records = parse_conll("meta\t5\nfoo\tEng\n\n")
    assert records == [{"uid": "5", "label": None, "text": "foo"}]


def test_load_test_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("Uid,Sentiment\n10,neutral\n11,Negative\n", encoding="utf-8")
    assert load_test_labels(path) == {"10": "neutral", "11": "negative"}
