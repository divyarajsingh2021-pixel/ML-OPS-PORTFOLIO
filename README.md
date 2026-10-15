# hinglish-mlops

An end-to-end **MLOps pipeline for Hinglish** (Hindi + English written in Roman
script) sentiment classification: 3 classes — *negative / neutral / positive*.

Built to be **cloned and run in ~10 minutes** on a free, CPU-only setup (tested on a
Windows laptop with 8 GB RAM, no GPU). Everything is a free tool, every stage is
scripted, and every claim below is reproducible from the committed reports.

---

## Results

Models are selected on the **validation** split; `test` and `shae_eval` are held out.
`shae_eval` is an out-of-distribution YouTube comment set that is **never trained on**
(only used for an honest generalization check).

### Macro-F1

| Model | train | val | test | shae_eval (OOD) |
|---|---|---|---|---|
| `baseline_tfidf_logreg` | 0.848 | 0.614 | 0.665 | 0.399 |
| `improved_tfidf_svc` | 0.800 | **0.632** | **0.691** | **0.414** |

### Accuracy

| Model | train | val | test | shae_eval (OOD) |
|---|---|---|---|---|
| `baseline_tfidf_logreg` | 0.847 | 0.611 | 0.661 | 0.401 |
| `improved_tfidf_svc` | 0.799 | 0.629 | 0.687 | 0.417 |

Per-class F1 on the test split for the improved model: negative **0.716**, neutral
**0.614**, positive **0.744** — the *neutral* class is the hard one, as usual for
sentiment.

Numbers come from [`reports/model_comparison.md`](reports/model_comparison.md)
(regenerate with `inv compare`) and the per-model JSON reports in `reports/`.

---

## Quickstart (~10 minutes)

```bash
git clone <your-fork-url> hinglish-mlops
cd hinglish-mlops

# 1. environment
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"

# 2. data: download + SHA-256 verify, then clean/split (~5 MB)
invoke data

# 3. train both models (writes models/*.joblib + reports/*.json, logs to MLflow)
invoke train

# 4. try it
invoke serve          # FastAPI on http://127.0.0.1:8000/docs
invoke demo           # Streamlit UI on http://localhost:8501  (pip install -e '.[demo]')
invoke mlflow-ui      # MLflow tracking UI on http://localhost:5000
invoke monitor        # Evidently drift report: val vs train
```

Sanity check: `invoke lint && invoke test` runs Ruff and the pytest suite (78 tests,
~15 s, 85 %+ coverage).

### All tasks

| Task | What it does |
|---|---|
| `invoke data` | download raw data, verify checksums, clean + split |
| `invoke train` | train + evaluate both models, log to MLflow |
| `invoke compare` | write `reports/model_comparison.md` |
| `invoke figures` | render confusion-matrix PNGs under `reports/figures/` |
| `invoke serve` | run the FastAPI inference service |
| `invoke demo` | run the Streamlit demo (needs the `demo` extra) |
| `invoke monitor` | Evidently drift report (`--current val` by default) |
| `invoke mlflow-ui` | MLflow tracking UI |
| `invoke lint` / `invoke test` / `invoke precommit` | Ruff / pytest / pre-commit |

---

## How it fits together

```
data raw (HF mirrors) ─SHA-256─▶ data/processed/*.csv ──▶ train ──▶ models/*.joblib
        │                              │                  │              │
        │                              │                  │              ├─▶ FastAPI  /predict
        │                              │                  │              └─▶ Streamlit demo
        │                              │                  └─▶ mlflow.db (params, metrics, artifacts)
        │                              └─▶ Evidently drift report (train vs any batch)
        └─ data/manifest.sha256 + reports/data_card.md (provenance, committed)
```

Repository layout:

```
src/hinglish_mlops/
├── config.py            # paths, label order, seed
├── data/                # download (checksums), SentiMix parsing, cleaning, splitting
├── dataset.py           # loaders for the processed splits
├── modeling.py          # model builders (TF-IDF + LogReg / LinearSVC), save/load
├── train.py             # training + evaluation + metrics reports + MLflow logging
├── tracking.py          # MLflow experiment tracking (SQLite backend)
├── evaluation.py        # accuracy, per-class/macro F1, confusion matrix
├── compare.py           # model comparison table
├── monitoring.py        # Evidently drift reports (HTML + JSON summary)
├── figures.py           # confusion-matrix PNGs (Agg backend, headless-safe)
└── api.py               # FastAPI service (/health, /predict, /predict/batch)
demo/app.py              # Streamlit demo
tests/                   # 78 tests, every phase has its own
reports/                 # data card, metrics, drift summaries (committed, small)
.github/workflows/ci.yml # lint, tests (3.11/3.12), end-to-end pipeline, build
```

---

## The data

| Split | n | Source |
|---|---|---|
| train | 13,854 | SemEval-2020 Task 9 **SentiMix** (Hinglish) |
| val | 2,932 | SentiMix dev |
| test | 2,987 | SentiMix test |
| `shae_eval` | 3,181 | `shae2977/hinglish-youtube-sentiments-dataset` (CC-BY-4.0), OOD only |

Reproducibility without DVC (free-tier friendly):

- `data/manifest.sha256` pins the exact upstream files by hash — `invoke data`
  refuses to proceed if a download does not match.
- `reports/data_card.md` and `reports/data_stats.json` document sources, licences,
  class balance and every filter that was applied.
- **Leakage guard**: cross-split duplicates are dropped (59 from val, 7 from test) as
  well as intra-split duplicates (146/9/6/9 across train/val/test/shae), and the OOD
  set has **0** overlap with SentiMix.
- Text is normalised (lowercase, HTML unescape, URL/mention removal, hashtag
  unwrapping) by `data/clean.py`; the raw text is kept alongside.

See [`reports/data_card.md`](reports/data_card.md) for the full card.

---

## Models

Both models keep the vectoriser **inside** the scikit-learn pipeline, so training and
inference apply identical preprocessing — no train/serve skew.

| | `baseline_tfidf_logreg` | `improved_tfidf_svc` |
|---|---|---|
| Features | word TF-IDF (1,2) | word TF-IDF (1,2) + char_wb TF-IDF (3,5) |
| Classifier | logistic regression | `LinearSVC(C=0.05)` |
| Scores served | `predict_proba` | softmax over decision values |

**Why not transformers?** No GPU and an 8 GB RAM budget; a fine-tuned mBERT/XLM-R
would not run locally, and TF-IDF + linear models give a strong, fast, debuggable
baseline on this dataset. **Why `C=0.05`?** The default `C=1.0` overfits badly on the
char features (train macro-F1 0.99 → test 0.635, worse than the baseline). Sweeping C
on the validation split fixed it — see [`reports/model_comparison.md`](reports/model_comparison.md).

---

## Serving the model

```bash
invoke serve                      # or: python -m hinglish_mlops.api --port 8000
```

Interactive docs at <http://127.0.0.1:8000/docs>.

```python
import httpx

httpx.post("http://127.0.0.1:8000/predict", json={"text": "yaar ye product bahut accha hai"}).json()
# {'label': 'neutral', 'label_id': 1, 'scores': {...}, 'model': 'baseline_tfidf_logreg', ...}

httpx.post(
    "http://127.0.0.1:8000/predict/batch", json={"texts": ["bilkul bekaar", "theek hai chalta hai"]}
).json()
```

- `GET /health` — liveness, configured model, label set.
- `POST /predict` — one text → label + per-class scores.
- `POST /predict/batch` — up to 128 texts per call.
- Configurable with env vars: `HINGLISH_MODEL_NAME`, `HINGLISH_MODEL_DIR`.
- A missing model returns `503` with instructions instead of crashing.

---

## Experiment tracking

Every `invoke train` logs params, per-split metrics and the model artifact to a local
**MLflow** SQLite store (`mlflow.db`, git-ignored), then prints the run id.

```bash
invoke mlflow-ui     # browse runs side by side (port 5000)
```

> MLflow 3 deprecated the file store, so this project uses `sqlite:///mlflow.db` —
> no server process needed, still fully offline. On Windows the UI can take 30–60 s
> to finish starting; it is healthy when `/_stcore/health` (MLflow) or `/health`
> (API) responds.

Tracking is optional and can never break training: `--no-track` skips it, and any
tracking failure is printed as a warning while the training run still completes.

---

## Monitoring

```bash
invoke monitor                      # val vs train
invoke monitor --current shae_eval  # OOD set vs train
python -m hinglish_mlops.monitoring --current val --fail-on-drift   # CI-style gate
```

Text is summarised into numeric features (length, word counts, digit/uppercase/
special-character shares) plus the label distribution, then compared with Evidently's
statistical drift tests. Outputs: `reports/drift_report_<name>.html` (git-ignored) and
`reports/drift_summary_<name>.json` (committed).

Findings on the real data:

| Comparison | Dataset drift | Drifted columns |
|---|---|---|
| train → val | **false** | 0 / 6 |
| train → `shae_eval` (OOD) | **true** | 6 / 6 |

That OOD drift is the story of the 0.41 accuracy on `shae_eval`: YouTube comments are
shorter and stylistically different from SentiMix tweets. The drift gate exits non-zero
when drift is detected, so CI can catch distribution shifts before they reach users.

---

## API + CI

CI (`.github/workflows/ci.yml`) runs on `ubuntu-latest` for every push/PR:

| Job | What it checks |
|---|---|
| `quality` | Ruff lint + format, pre-commit hooks |
| `tests` | pytest + coverage gate (≥ 80 %) on Python 3.11 and 3.12 |
| `pipeline` | download → checksums → clean/split → train → drift gate on real data |
| `package` | builds the sdist + wheel |

Everything runs on free runners; no secrets, no paid services.

---

## Design decisions

- **Linear models, not transformers**: the target machine has no GPU and 8 GB RAM.
  TF-IDF + linear classifiers train in seconds and stay fully reproducible.
- **No DVC**: it needs a remote storage account (not free-tier friendly). A checksum
  manifest + download scripts + data card give the same integrity guarantees.
- **MLflow on SQLite**: the file store is deprecated in MLflow 3; SQLite needs no
  server and is a single git-ignored file.
- **Official SentiMix splits** are used unchanged (they are already ~70/15/15); the
  OOD set is kept strictly out of training and tuning.
- **Fail loudly**: checksum mismatches, missing data, missing model artifacts and
  tracking outages are surfaced (non-zero exits, `503` responses, warnings), never
  silently swallowed.

## Limitations

- Accuracy ~0.69 macro-F1 on the held-out SentiMix test set is modest — this is a
  deliberately lightweight model on a genuinely noisy task (romanised, code-mixed,
  emoji-heavy text).
- The *neutral* class is weakest (F1 0.61); sarcastic and mixed-sentiment text is
  frequently misclassified.
- Performance drops sharply out of distribution (0.41 on YouTube comments) — see the
  drift report above. Retraining on in-domain data is the fix, not tuning.
- The SVM scores are a softmax over decision values, so they are **not calibrated
  probabilities**.
- Single-sentence inputs only; no emoji/emoticon handling beyond what the text
  carries.

### Ideas for next steps

- Log live API predictions and monitor *their* drift against the training set.
- Add a transformer baseline (e.g. distilled multilingual model) behind a CPU-friendly
  inference runtime.
- Package the model with `skops` for safer sharing, and add a DVC-style remote if a
  free storage option is ever needed.
- Threshold-per-class tuning for the weak *neutral* class.

---

## Development

```bash
invoke lint        # ruff check
invoke format      # ruff format + autofix
invoke test        # pytest with coverage
invoke precommit   # all pre-commit hooks
```

- Python `>=3.11,<3.13`; deps in `pyproject.toml` (`demo` extra adds Streamlit).
- Pre-commit hooks: Ruff, YAML/TOML checks, whitespace, large-file and private-key
  guards.
- Tests never touch the network or the real `data/`/`models/` directories — they use
  tiny synthetic corpora and `tmp_path`.

## License

MIT — see [LICENSE](LICENSE).

Data licences: SentiMix (SemEval-2020 Task 9) and
`shae2977/hinglish-youtube-sentiments-dataset` (CC-BY-4.0). See the
[data card](reports/data_card.md) for details and attribution.
