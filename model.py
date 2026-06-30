"""
model.py
========
Everything related to the machine-learning model: pipeline definitions,
training/evaluation, and inference.

Sections
--------
1. PIPELINE DEFINITIONS   — TF-IDF + classifier pipelines (NB, LR, SVM)
2. TRAINING & EVALUATION  — fit, cross-validate, compare, persist
3. INFERENCE              — load a saved model and predict a genre

Usage
-----
    from dataset import build_dataset
    from model import train_and_evaluate, predict_genre

    df = build_dataset()
    train_and_evaluate(df)                       # trains, saves best model

    result = predict_genre("A detective hunts a serial killer.")
    print(result["genre"])                       # 'Thriller'
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

MODEL_PATH = Path("movie_genre_model.pkl")
RESULTS_PATH = Path("model_results.json")


# ═════════════════════════════════════════════════════════════════════════
# 1. PIPELINE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════
#
# All three pipelines share the same TF-IDF front-end so results are
# directly comparable:
#   - ngram_range=(1, 2)   unigrams + bigrams ("serial killer")
#   - max_features=10_000  vocabulary cap
#   - sublinear_tf=True    log(1+tf) scaling, dampens frequent terms
#   - stop_words="english" drop common English words
#
# Classifiers:
#   Naive Bayes           MultinomialNB        fast probabilistic baseline
#   Logistic Regression   L-BFGS, C=5           best cross-validated F1
#   SVM (LinearSVC)        C=1                  highest single test F1

_TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    max_features=10_000,
    sublinear_tf=True,
    min_df=1,
    stop_words="english",
)


def build_pipelines() -> dict[str, Pipeline]:
    """Return a ``{name: Pipeline}`` dict, one per classifier.

    Every pipeline has two named steps — ``"tfidf"`` and ``"clf"`` — so
    individual steps can be inspected or swapped, e.g.::

        pipe = build_pipelines()["Logistic Regression"]
        vocab = pipe.named_steps["tfidf"].vocabulary_
    """
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**_TFIDF_KWARGS)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**_TFIDF_KWARGS)),
            ("clf",   LogisticRegression(
                C=5.0, solver="lbfgs", max_iter=1_000, random_state=42,
            )),
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(**_TFIDF_KWARGS)),
            ("clf",   LinearSVC(C=1.0, max_iter=2_000, random_state=42)),
        ]),
    }


# ═════════════════════════════════════════════════════════════════════════
# 2. TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[dict, str]:
    """Fit, cross-validate, and compare all classifier pipelines.

    Prints a comparison table and the classification report for the
    best-performing model, then persists the best pipeline to
    ``movie_genre_model.pkl`` and a results summary to
    ``model_results.json``.

    Parameters
    ----------
    df:
        DataFrame with columns ``plot`` (str) and ``genre`` (str).
    test_size:
        Fraction of data held out for the final test set.
    cv_folds:
        Number of cross-validation folds.
    random_state:
        Controls train/test split reproducibility.

    Returns
    -------
    (results, best_name) :
        ``results`` is keyed by model name; each value holds ``pipeline``,
        ``f1``, ``cv_mean``, ``cv_std``, ``y_test``, ``y_pred``.
        ``best_name`` is the model with the highest CV F1.
    """
    X, y = df["plot"].values, df["genre"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    sep = "=" * 65
    print(f"\n{sep}\n  MOVIE GENRE PREDICTOR — Model Comparison\n{sep}")
    print(f"  Dataset : {len(df)} samples  |  {df['genre'].nunique()} genres")
    print(f"  Train   : {len(X_train)}      |  Test  : {len(X_test)}")
    print(f"{sep}\n")

    results = {}
    for name, pipe in build_pipelines().items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")
        cv = cross_val_score(pipe, X, y, cv=cv_folds, scoring="f1_weighted")

        results[name] = {
            "pipeline": pipe, "f1": f1,
            "cv_mean": cv.mean(), "cv_std": cv.std(),
            "y_test": y_test, "y_pred": y_pred,
        }
        print(f"  ── {name}")
        print(f"     Test F1 : {f1:.4f}")
        print(f"     CV F1   : {cv.mean():.4f} ± {cv.std():.4f}\n")

    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    best = results[best_name]

    print(f"\n{sep}\n  Best model : {best_name}  (CV F1 = {best['cv_mean']:.4f})\n{sep}\n")
    print(classification_report(best["y_test"], best["y_pred"]))

    _save_model(best["pipeline"], best_name)
    _save_results(results, best_name, df)

    return results, best_name


def _save_model(pipeline: Pipeline, name: str) -> None:
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": pipeline, "name": name}, fh)
    print(f"\n  ✓ Model saved   → {MODEL_PATH}")


def _save_results(results: dict, best_name: str, df: pd.DataFrame) -> None:
    summary = {
        name: {
            "f1": round(r["f1"], 4),
            "cv_mean": round(r["cv_mean"], 4),
            "cv_std": round(r["cv_std"], 4),
        }
        for name, r in results.items()
    }
    summary["best"] = best_name
    summary["genres"] = sorted(df["genre"].unique().tolist())
    with open(RESULTS_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  ✓ Results saved → {RESULTS_PATH}\n")


# ═════════════════════════════════════════════════════════════════════════
# 3. INFERENCE
# ═════════════════════════════════════════════════════════════════════════

def predict_genre(text: str, model_path: str | Path = MODEL_PATH) -> dict:
    """Predict the genre of *text* using the saved pipeline.

    Returns a dict::

        {
            "genre": "Thriller",
            "top3":  [("Thriller", 48.2), ("Action", 22.1), ("Drama", 11.5)],
            "model": "Logistic Regression",
        }

    ``LinearSVC`` has no ``predict_proba``; for SVM models the
    decision-function scores are converted to pseudo-probabilities via a
    numerically stable softmax.

    Raises
    ------
    FileNotFoundError
        If *model_path* doesn't exist. Run training (``python main.py``) first.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run `python main.py` to train and save a model first."
        )
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)

    pipe = bundle["model"]
    clf = pipe.named_steps["clf"]
    vec = pipe.named_steps["tfidf"].transform([text])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(vec)[0]
        classes = clf.classes_
        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [(classes[i], round(float(probs[i]) * 100, 1)) for i in top_idx]
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(vec)[0]
        classes = clf.classes_
        exp_s = np.exp(scores - scores.max())
        softmax = exp_s / exp_s.sum()
        top_idx = np.argsort(softmax)[::-1][:3]
        top3 = [(classes[i], round(float(softmax[i]) * 100, 1)) for i in top_idx]
    else:
        top3 = [(pipe.predict([text])[0], 100.0)]

    return {"genre": top3[0][0], "top3": top3, "model": bundle["name"]}
