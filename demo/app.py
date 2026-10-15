"""Streamlit demo for the Hinglish sentiment classifier.

Run it with ``inv demo`` (or ``streamlit run demo/app.py``). The app loads a
trained model straight from ``models/`` -- no API server needed -- and shows the
predicted class, per-class scores, the benchmark metrics and the confusion
matrix. Train the models first with ``inv train``.
"""

from __future__ import annotations

import json

import streamlit as st

from hinglish_mlops import __version__
from hinglish_mlops.api import ModelStore, class_scores
from hinglish_mlops.config import LABELS, MODELS_DIR, REPORTS_DIR
from hinglish_mlops.modeling import BASELINE_NAME, IMPROVED_NAME

MODEL_CHOICES = {
    "TF-IDF + logistic regression (baseline)": BASELINE_NAME,
    "TF-IDF (word + char) + linear SVM (improved)": IMPROVED_NAME,
}

st.set_page_config(page_title="Hinglish Sentiment", page_icon="🗣️", layout="wide")


@st.cache_resource
def load_pipeline(model_name: str):
    store = ModelStore(MODELS_DIR, model_name)
    return store.get()


@st.cache_data
def load_metrics(model_name: str) -> dict | None:
    path = REPORTS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


st.title("Hinglish Sentiment Classifier")
st.caption(
    f"3-class sentiment for Roman-script Hinglish · v{__version__} · "
    "trained on SemEval-2020 Task 9 SentiMix"
)

with st.sidebar:
    st.header("Settings")
    choice = st.radio("Model", list(MODEL_CHOICES), index=1)
    model_name = MODEL_CHOICES[choice]

    metrics = load_metrics(model_name)
    if metrics:
        st.subheader("Benchmark")
        st.dataframe(
            {
                split: {
                    "accuracy": round(values["accuracy"], 3),
                    "macro F1": round(values["macro_f1"], 3),
                }
                for split, values in metrics.items()
            },
            width="stretch",
        )
        st.caption("shae_eval is an out-of-distribution YouTube set (never trained on).")

    figure = REPORTS_DIR / "figures" / f"{model_name}_confusion_matrix.png"
    if figure.exists():
        st.subheader("Confusion matrix (test split)")
        st.image(str(figure), width="stretch")
    else:
        st.info("Run `inv figures` to render the confusion matrix.")

st.subheader("Try it")
text = st.text_area(
    "Hinglish text (Roman script)",
    height=120,
    placeholder="e.g. yaar ye product bilkul bekaar nikal, paisa barbaad",
)

if st.button("Predict", type="primary") and text.strip():
    try:
        pipeline = load_pipeline(model_name)
    except FileNotFoundError:
        st.error("No trained model found. Run `inv train` first.")
        st.stop()

    scores = class_scores(pipeline, text)
    label = str(pipeline.predict([text])[0])

    left, right = st.columns([1, 2])
    left.metric("Predicted sentiment", label.upper())
    right.bar_chart({name: scores[name] for name in LABELS})
    with left, st.expander("Raw scores"):
        st.json({name: round(score, 4) for name, score in scores.items()})
elif st.button("Predict"):
    st.warning("Enter some text first.")

st.divider()
st.caption(
    "Scores are logistic-regression probabilities for the baseline and a softmax over "
    "SVM decision values for the improved model. Model quality is modest on slang-heavy "
    "text — see the limitations in the README."
)
