"""Phase 1 smoke tests: the package imports and exposes a version."""

import hinglish_mlops


def test_package_imports():
    assert hinglish_mlops.__version__


def test_version_is_semver_like():
    parts = hinglish_mlops.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
