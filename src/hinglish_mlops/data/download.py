"""Download the raw datasets and verify them against ``data/manifest.sha256``.

This is the project's lightweight, dependency-free stand-in for DVC: rather than
a content-addressed cache, we keep a plain-text checksum manifest in git and
re-download on demand. Every file is verified after download and re-verified on
subsequent runs, so a corrupted or truncated file can never silently reach the
rest of the pipeline.

CLI::

    python -m hinglish_mlops.data.download [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

from ..config import MANIFEST_PATH, RAW_DIR
from .sources import SOURCES

USER_AGENT = "hinglish-mlops/0.1 (+https://github.com/divyarajsingh2021-pixel/ML-OPS-PORTFOLIO)"
CHUNK_SIZE = 1 << 15  # 32 KiB


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, str]:
    """Parse a sha256sum-style manifest into ``{relative_path: hex_digest}``."""
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, separator, relative = stripped.partition("  ")
        if not separator:
            raise ValueError(f"Malformed manifest line: {line!r}")
        checksums[relative.strip()] = digest.strip().lower()
    return checksums


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, *, session: requests.Session | None = None) -> Path:
    """Stream ``url`` to ``dest``, creating parent directories as needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    http = session or requests
    headers = {"User-Agent": USER_AGENT}
    with http.get(url, stream=True, timeout=60, headers=headers) as response:
        response.raise_for_status()
        tmp = dest.with_name(dest.name + ".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
        tmp.replace(dest)
    return dest


def fetch_source(
    relative: str,
    url: str,
    expected: str,
    *,
    force: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Ensure ``data/raw/<relative>`` exists and matches ``expected``."""
    dest = RAW_DIR / relative
    if dest.exists() and not force and sha256_file(dest) == expected:
        print(f"[{'ok':>10}] {relative}")
        return dest
    download_file(url, dest, session=session)
    actual = sha256_file(dest)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {relative}: expected {expected}, got {actual}")
    print(f"[{'downloaded':>10}] {relative}")
    return dest


def download_all(*, force: bool = False, session: requests.Session | None = None) -> list[Path]:
    """Download and verify every source declared in ``sources.SOURCES``."""
    manifest = load_manifest()
    missing = sorted(set(SOURCES) - set(manifest))
    if missing:
        raise ValueError(f"Sources missing from manifest: {missing}")
    http = session or requests.Session()
    return [
        fetch_source(relative, SOURCES[relative], manifest[relative], force=force, session=http)
        for relative in SOURCES
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download + verify raw datasets.")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)
    paths = download_all(force=args.force)
    total = sum(p.stat().st_size for p in paths)
    print(f"Verified {len(paths)} files ({total / 1e6:.2f} MB) under {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
