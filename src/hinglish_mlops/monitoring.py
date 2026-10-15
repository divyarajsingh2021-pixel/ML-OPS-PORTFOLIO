"""Dataset-drift monitoring with Evidently.

The reference is the training split (what the model was fit on); the current
data is any batch we want to compare against it -- validation, the
out-of-distribution YouTube set, or live requests exported to CSV. Raw text is
summarised into numeric features (length, word counts, digit/uppercase/
special-character shares) plus the label distribution, which is what Evidently's
statistical drift tests run on.

Outputs a self-contained HTML report for humans and a JSON summary that CI can
gate on with ``--fail-on-drift``::

    python -m hinglish_mlops.monitoring --current val
    python -m hinglish_mlops.monitoring --current shae_eval --fail-on-drift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from .config import PROCESSED_DIR, REPORTS_DIR
from .dataset import EVAL_NAME, SPLIT_NAMES, load_eval, load_split

TEXT_COLUMN = "text_clean"
LABEL_COLUMN = "label"
DRIFT_SHARE = 0.5
REPORT_FILENAME = "drift_report.html"
SUMMARY_FILENAME = "drift_summary.json"


def _mean_word_length(words: list[str]) -> float:
    return sum(len(word) for word in words) / len(words) if words else 0.0


def text_features(texts: pd.Series) -> pd.DataFrame:
    """Numeric summaries of raw text, used as drift-monitoring columns."""
    raw = texts.astype(str)
    words = raw.str.lower().str.split()
    lengths = raw.str.len().clip(lower=1)
    return pd.DataFrame(
        {
            "text_length": raw.str.len(),
            "word_count": words.str.len(),
            "avg_word_length": words.apply(_mean_word_length),
            "digit_share": raw.str.count(r"\d") / lengths,
            "uppercase_share": raw.str.count(r"[A-Z]") / lengths,
            "special_char_share": raw.str.count(r"[^0-9A-Za-z\s]") / lengths,
        },
        index=texts.index,
    )


def features_from_frame(frame: pd.DataFrame, text_column: str = TEXT_COLUMN) -> pd.DataFrame:
    """Monitoring features for a dataset frame (adds labels when available)."""
    if text_column not in frame.columns:
        raise ValueError(f"missing text column {text_column!r}; available: {list(frame.columns)}")
    features = text_features(frame[text_column])
    if LABEL_COLUMN in frame.columns:
        features[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str)
    return features


def drop_constant_columns(reference: pd.DataFrame, current: pd.DataFrame) -> tuple:
    """Keep columns that vary somewhere; constant columns carry no drift signal."""
    keep = [
        column
        for column in reference.columns
        if reference[column].nunique() > 1 or current[column].nunique() > 1
    ]
    if not keep:
        raise ValueError("no comparable columns: every feature is constant in both datasets")
    return reference[keep], current[keep]


def _column_drift(metric: dict[str, object]) -> dict[str, object]:
    """Read one ``ValueDrift`` metric.

    Evidently compares columns with different statistical tests: distances
    (Wasserstein, Jensen-Shannon, ...) drift when the score is *above* the
    threshold, while p-value tests (K-S, z-test, ...) drift when it is *below*.
    """
    config = metric["config"]
    method = str(config.get("method", ""))
    threshold = config.get("threshold")
    score = float(metric["value"])
    if threshold is None:
        detected = None
    elif "p_value" in method:
        detected = bool(score < float(threshold))
    else:
        detected = bool(score > float(threshold))
    return {
        "score": score,
        "method": method,
        "threshold": threshold,
        "drift_detected": detected,
    }


def summarize(snapshot: object, drift_share: float) -> dict[str, object]:
    """Extract drift scores/flags from an Evidently snapshot."""
    columns: dict[str, dict[str, object]] = {}
    drifted_count: int | None = None
    drifted_share: float | None = None

    for metric in snapshot.dict()["metrics"]:  # type: ignore[attr-defined]
        name = str(metric["metric_name"])
        if name.startswith("ValueDrift(column="):
            column = name.split("column=", 1)[1].split(",")[0]
            columns[column] = _column_drift(metric)
        elif name.startswith("DriftedColumnsCount"):
            value = metric["value"]
            drifted_count = int(value["count"])
            drifted_share = float(value["share"])

    dataset_drift = drifted_share is not None and drifted_share >= drift_share
    return {
        "dataset_drift": bool(dataset_drift),
        "drift_share_threshold": drift_share,
        "drifted_column_count": drifted_count,
        "drifted_column_share": drifted_share,
        "columns": columns,
    }


def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    drift_share: float = DRIFT_SHARE,
) -> tuple[object, dict[str, object]]:
    """Run Evidently drift tests on two feature frames -> (snapshot, summary)."""
    snapshot = Report(metrics=[DataDriftPreset(drift_share=drift_share)]).run(
        current_data=current, reference_data=reference
    )
    return snapshot, summarize(snapshot, drift_share)


def _output_paths(out_dir: Path, name: str | None) -> tuple[Path, Path]:
    if name:
        return out_dir / f"drift_report_{name}.html", out_dir / f"drift_summary_{name}.json"
    return out_dir / REPORT_FILENAME, out_dir / SUMMARY_FILENAME


def run_monitoring(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_dir: Path = REPORTS_DIR,
    *,
    drift_share: float = DRIFT_SHARE,
    name: str | None = None,
) -> dict[str, object]:
    """Write the HTML report + JSON summary into ``out_dir`` and return the summary."""
    reference_features, current_features = drop_constant_columns(
        features_from_frame(reference), features_from_frame(current)
    )
    snapshot, summary = build_report(reference_features, current_features, drift_share=drift_share)
    report_path, summary_path = _output_paths(out_dir, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(report_path))
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def _load_frame(name_or_path: str, processed_dir: Path) -> pd.DataFrame:
    if name_or_path in SPLIT_NAMES:
        return load_split(name_or_path, processed_dir)
    if name_or_path == EVAL_NAME:
        return load_eval(processed_dir)

    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Unknown split {name_or_path!r} and no such CSV file: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    if TEXT_COLUMN not in frame.columns:
        raise ValueError(f"{path} is missing the {TEXT_COLUMN!r} column")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Evidently drift monitoring.")
    parser.add_argument("--reference", default="train", help="split name or CSV path")
    parser.add_argument(
        "--current",
        default="val",
        help=f"split name ({', '.join((*SPLIT_NAMES, EVAL_NAME))}) or CSV path",
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--drift-share", type=float, default=DRIFT_SHARE)
    parser.add_argument(
        "--fail-on-drift", action="store_true", help="exit 1 when dataset drift is detected"
    )
    args = parser.parse_args(argv)

    label = args.current
    if label not in (*SPLIT_NAMES, EVAL_NAME):
        label = Path(label).stem

    try:
        reference = _load_frame(args.reference, args.processed_dir)
        current = _load_frame(args.current, args.processed_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = run_monitoring(
        reference, current, args.out_dir, drift_share=args.drift_share, name=label
    )

    print(f"dataset_drift: {summary['dataset_drift']}")
    print(f"drifted_columns: {summary['drifted_column_count']}")
    for column, info in sorted(summary["columns"].items()):
        marker = "DRIFT" if info["drift_detected"] else "    "
        print(f"  [{marker}] {column:<20} score={info['score']:.4f} ({info['method']})")
    report_path, summary_path = _output_paths(args.out_dir, label)
    print(f"wrote {report_path} and {summary_path}")

    if args.fail_on_drift and summary["dataset_drift"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
