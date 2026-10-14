"""Text normalisation for Hinglish (Roman-script, code-mixed) social media text.

The rules are deliberately conservative -- we only strip artefacts that carry no
sentiment signal (links and user handles), and we unwrap hashtags rather than
delete them. Hinglish is written in Roman script, so there is no transliteration
or Devanagari handling here.

Because the SentiMix source is tokenised and reconstructed by joining tokens with
spaces, URLs, mentions and hashtags arrive fragmented (``https // t . co / abc``,
``@ user``, ``# tag``). The patterns below tolerate that optional whitespace so
both raw and reconstructed inputs are cleaned identically.
"""

from __future__ import annotations

import html
import re

import pandas as pd

# URLs, including the whitespace-split artefacts from CoNLL reconstruction.
# Covers both "https // t . co / code" and the dotless "https // t co / code".
_URL_RE = re.compile(
    r"(?:https?|ftp|www)\s*:?\s*(?:/\s*)*"
    r"(?:(?:[A-Za-z0-9_\-]+\s*\.\s*)+[A-Za-z0-9_\-]{1,}"
    r"|[A-Za-z0-9_\-]{1,}(?:\s+[A-Za-z0-9_\-]{1,})?\s*/\s*\S+)"
    r"(?:\s*/\s*\S+)*",
    re.IGNORECASE,
)
# Bare domains whose scheme was dropped, e.g. "bit . ly" or "youtu.be".
_DOMAIN_RE = re.compile(
    r"\b[A-Za-z0-9_\-]+\s*\.\s*"
    r"(?:com|co|org|net|in|me|ly|be|gl|io|tv|info|biz|xyz|link|to|ru)\b",
    re.IGNORECASE,
)
_MENTION_RE = re.compile(r"@\s*\w+")
_HASHTAG_RE = re.compile(r"#\s*(\w+)")
_SYMBOL_RE = re.compile(r"[@#]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Return a normalised form of ``text`` (lowercased, links/handles removed)."""
    if not isinstance(text, str):
        return ""
    normalised = html.unescape(text).lower()
    normalised = _URL_RE.sub(" ", normalised)
    normalised = _DOMAIN_RE.sub(" ", normalised)
    normalised = _MENTION_RE.sub(" ", normalised)
    normalised = _HASHTAG_RE.sub(r"\1", normalised)
    normalised = _SYMBOL_RE.sub(" ", normalised)
    return _WHITESPACE_RE.sub(" ", normalised).strip()


def clean_series(texts: pd.Series) -> pd.Series:
    """Apply :func:`clean_text` element-wise to a pandas Series."""
    return texts.map(clean_text)
