"""Hermetic tests for the download helpers (no network access)."""

import hashlib

import pytest

from hinglish_mlops.data import download


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, data: bytes):
        self.data = data
        self.calls: list[str] = []

    def get(self, url, stream=None, timeout=None, headers=None):
        self.calls.append(url)
        return _FakeResponse(self.data)


def test_download_file_writes_bytes(tmp_path):
    session = _FakeSession(b"hello world")
    dest = tmp_path / "sub" / "file.bin"
    download.download_file("http://example/x", dest, session=session)
    assert dest.read_bytes() == b"hello world"
    assert session.calls == ["http://example/x"]


def test_fetch_source_verifies_and_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    payload = b"payload"
    digest = hashlib.sha256(payload).hexdigest()
    session = _FakeSession(payload)

    path = download.fetch_source("a/b.bin", "http://u", digest, session=session)
    assert path.read_bytes() == payload

    download.fetch_source("a/b.bin", "http://u", digest, session=session)
    assert session.calls == ["http://u"]  # second call served from cache


def test_fetch_source_force_redownloads(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    payload = b"payload"
    digest = hashlib.sha256(payload).hexdigest()
    session = _FakeSession(payload)

    download.fetch_source("a/b.bin", "http://u", digest, session=session)
    download.fetch_source("a/b.bin", "http://u", digest, force=True, session=session)
    assert session.calls == ["http://u", "http://u"]


def test_fetch_source_detects_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    session = _FakeSession(b"payload")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download.fetch_source("a/b.bin", "http://u", "0" * 64, session=session)
