"""Cross-platform task runner (invoke) for the hinglish-mlops project.

The tasks use the *currently running* Python interpreter (``sys.executable``),
so they work whether or not the virtualenv is activated, as long as you invoke
them with the project's interpreter, e.g.::

    .venv\\Scripts\\python.exe -m invoke --list
    .venv\\Scripts\\python.exe -m invoke lint
    .venv\\Scripts\\python.exe -m invoke test

Once the venv is activated you can simply use ``inv <task>``.
"""

import shutil
import sys

from invoke import task

PY = f'"{sys.executable}"'


@task
def install(c, extras="dev"):
    """Install the package in editable mode with dev extras."""
    c.run(f'{PY} -m pip install -e ".[{extras}]"')


@task
def lint(c):
    """Run ruff lint checks (no changes)."""
    c.run(f"{PY} -m ruff check src tests")


@task
def format(c):
    """Auto-format and auto-fix with ruff."""
    c.run(f"{PY} -m ruff format src tests")
    c.run(f"{PY} -m ruff check --fix src tests")


@task
def test(c, cov=True):
    """Run the test suite (with coverage by default)."""
    cmd = f"{PY} -m pytest"
    if cov:
        cmd += " --cov=hinglish_mlops --cov-report=term-missing"
    c.run(cmd)


@task
def precommit(c):
    """Run all pre-commit hooks against every file."""
    c.run(f"{PY} -m pre_commit run --all-files")


@task
def clean(c):
    """Remove common Python build/test caches."""
    for path in (".pytest_cache", ".ruff_cache", "build", "dist"):
        shutil.rmtree(path, ignore_errors=True)
    print("cleaned")
