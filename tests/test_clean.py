"""Unit tests for Hinglish text cleaning."""

from hinglish_mlops.data.clean import clean_text


def test_lowercases_and_trims():
    assert clean_text("  Hello World  ") == "hello world"


def test_removes_url_with_scheme():
    assert clean_text("check https://t.co/abc now") == "check now"


def test_removes_split_url_artefact():
    assert clean_text("rt https // t . co / abcdef") == "rt"


def test_removes_split_url_without_dot():
    assert clean_text("go https // t co / AbC123 end") == "go end"


def test_removes_mentions_including_split():
    assert clean_text("@ user hello @friend") == "hello"


def test_unwraps_hashtags_and_keeps_words():
    assert clean_text("# Good vibes #Happy") == "good vibes happy"


def test_keeps_sentence_punctuation():
    assert clean_text("hello . world .") == "hello . world ."


def test_preserves_emoji():
    assert clean_text("bahut acha \U0001f60a") == "bahut acha \U0001f60a"


def test_non_string_returns_empty():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_html_entities_unescaped():
    assert clean_text("a &amp; b") == "a & b"
