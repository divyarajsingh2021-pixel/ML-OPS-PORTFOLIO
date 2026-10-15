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
def download_data(c, force=False):
    """Download + verify raw datasets against data/manifest.sha256."""
    cmd = f"{PY} -m hinglish_mlops.data.download"
    if force:
        cmd += " --force"
    c.run(cmd)


@task
def prepare_data(c):
    """Clean + split datasets; write reports/data_card.md and data_stats.json."""
    c.run(f"{PY} -m hinglish_mlops.data.prepare")


@task(download_data, prepare_data)
def data(c):
    """Download and prepare all datasets (convenience wrapper)."""


@task
def train_baseline(c):
    """Train + evaluate the TF-IDF (word) + logistic-regression baseline."""
    c.run(f"{PY} -m hinglish_mlops.train --model baseline_tfidf_logreg")


@task
def train_improved(c):
    """Train + evaluate the TF-IDF (word + char) + linear-SVM model."""
    c.run(f"{PY} -m hinglish_mlops.train --model improved_tfidf_svc")


@task(train_baseline, train_improved)
def train(c):
    """Train + evaluate all registered models."""


@task
def compare(c):
    """Write reports/model_comparison.md from the metrics reports."""
    c.run(f"{PY} -m hinglish_mlops.compare")


@task
def mlflow_ui(c):
    """Launch the MLflow tracking UI for the local mlflow.db store (port 5000)."""
    from hinglish_mlops.tracking import DEFAULT_TRACKING_URI

    c.run(f'{PY} -m mlflow ui --backend-store-uri "{DEFAULT_TRACKING_URI}"')


@task
def serve(c, port=8000, host="127.0.0.1"):
    """Run the FastAPI inference service with uvicorn (docs at /docs)."""
    c.run(f"{PY} -m uvicorn hinglish_mlops.api:app --host {host} --port {port}")


@task
def monitor(c, current="val", fail_on_drift=False):
    """Write an Evidently drift report (current split vs train) under reports/."""
    flag = " --fail-on-drift" if fail_on_drift else ""
    c.run(f"{PY} -m hinglish_mlops.monitoring --current {current}{flag}")


@task
def figures(c, model="improved_tfidf_svc", split="test"):
    """Render the confusion-matrix figure for a trained model."""
    c.run(f"{PY} -m hinglish_mlops.figures --model {model} --split {split}")


@task
def demo(c, port=8501):
    """Launch the Streamlit demo (needs the demo extra: pip install -e '.[demo]')."""
    c.run(f"{PY} -m streamlit run demo/app.py --server.port {port}")


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
