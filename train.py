"""Train and evaluate resume category classifiers.

Usage:  python train.py [--data data/Resume.csv]

Outputs (in models/):
    model.joblib    best fitted pipeline (TF-IDF + classifier with predict_proba)
    metrics.json    model comparison, test metrics, per-class report, top words
    confusion_matrix.png
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from utils import clean_text

SEED = 42


def ensure_data(path):
    """Download the Kaggle resume dataset (public, no login needed) if missing."""
    if os.path.exists(path):
        return
    import shutil
    import kagglehub
    print("Downloading dataset from Kaggle...")
    folder = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    shutil.copy(os.path.join(folder, "Resume", "Resume.csv"), path)


def tfidf():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                           min_df=2, max_df=0.9, sublinear_tf=True,
                           max_features=30000)


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=10),
        "Linear SVM": LinearSVC(C=1.0),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Resume.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- 1. load + clean --------------------------------------------------
    ensure_data(args.data)
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["Resume_str", "Category"]).drop_duplicates("Resume_str")
    df["text"] = df["Resume_str"].map(clean_text)
    df = df[df["text"].str.len() > 50]
    print(f"{len(df)} resumes, {df['Category'].nunique()} categories")
    print(df["Category"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["Category"], test_size=0.2,
        stratify=df["Category"], random_state=SEED)

    # ---- 2. compare models with 5-fold CV ---------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    comparison = []
    for name, clf in candidates().items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        comparison.append({"model": name, "cv_f1_macro": float(scores.mean()),
                           "cv_std": float(scores.std())})
        print(f"{name:20s} CV macro-F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

    # ---- 3. tune the two linear models, keep the best ---------------------
    grids = {
        "Logistic Regression": (LogisticRegression(max_iter=2000),
                                {"clf__C": [1, 10, 100]}),
        "Linear SVM": (LinearSVC(), {"clf__C": [0.1, 0.5, 1, 2]}),
    }
    best_name, best_search = None, None
    for name, (clf, grid) in grids.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        grid["tfidf__ngram_range"] = [(1, 1), (1, 2)]
        search = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        print(f"Tuned {name}: {search.best_params_} -> {search.best_score_:.4f}")
        for row in comparison:
            if row["model"] == name:
                row["tuned_cv_f1_macro"] = float(search.best_score_)
                row["best_params"] = {k: str(v) for k, v in search.best_params_.items()}
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_name, best_search = name, search

    # ---- 4. final model (needs probabilities for the web app) -------------
    best = best_search.best_estimator_
    if not hasattr(best.named_steps["clf"], "predict_proba"):
        best = Pipeline([("tfidf", best.named_steps["tfidf"]),
                         ("clf", CalibratedClassifierCV(best.named_steps["clf"], cv=3))])
        best.fit(X_train, y_train)

    pred = best.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"\nBest: {best_name}  test accuracy={acc:.4f}  macro-F1={f1:.4f}")
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    print(classification_report(y_test, pred, zero_division=0))

    # ---- 5. explainability: top words per class (linear models) -----------
    top_words = {}
    base = best.named_steps["clf"]
    if hasattr(base, "coef_"):
        vocab = np.array(best.named_steps["tfidf"].get_feature_names_out())
        for i, label in enumerate(base.classes_):
            top_words[label] = vocab[np.argsort(base.coef_[i])[-10:][::-1]].tolist()

    # ---- 6. confusion matrix image ----------------------------------------
    labels = sorted(df["Category"].unique())
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_test, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix - {best_name}")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=130)
    except ImportError:
        print("matplotlib not installed - skipping confusion matrix image")

    joblib.dump(best, os.path.join(args.out, "model.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "n_resumes": int(len(df)),
            "n_categories": len(labels),
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "comparison": comparison,
            "per_class": {k: v for k, v in report.items() if k in labels},
            "top_words": top_words,
        }, f, indent=2)
    print(f"Saved model + metrics to {args.out}/")


if __name__ == "__main__":
    main()
