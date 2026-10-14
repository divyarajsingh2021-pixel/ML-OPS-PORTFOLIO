"""Tests for the checksum manifest and download helpers."""

import re

import pytest

from hinglish_mlops.config import MANIFEST_PATH
from hinglish_mlops.data.download import load_manifest, sha256_file
from hinglish_mlops.data.sources import SOURCES


def test_manifest_covers_sources_with_valid_hashes():
    manifest = load_manifest(MANIFEST_PATH)
    assert set(manifest) == set(SOURCES)
    for digest in manifest.values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)


def test_load_manifest_rejects_malformed(tmp_path):
    path = tmp_path / "bad.sha256"
    path.write_text("no-separator-here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed manifest line"):
        load_manifest(path)


def test_sha256_file(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
