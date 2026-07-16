"""Lesson 06 — Model evaluation, with scikit-learn.

Same model (LogisticRegression on lesson 05's features), evaluated with
scikit-learn's own machinery: KFold + cross_val_score for cross-validation
(inside a Pipeline so imputation/scaling are refit per fold, exactly as the
manual loop in from_scratch.py does), then confusion_matrix, precision,
recall, F1 and roc_auc_score on the fixed split — those must match
from_scratch.py exactly.

Run from the repo root:

    python src/lesson06_model_evaluation/with_library.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
SEED = 42
K = 5

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]


def main() -> None:
    df = pd.read_csv(DATA)
    df["Sex"] = (df["Sex"] == "male").astype(float)  # male=1, female=0
    X = df[FEATURES].to_numpy(dtype=float)  # Age keeps NaN where missing
    y = df["Survived"].to_numpy(dtype=float)
    n = len(y)

    print("== Model evaluation: with scikit-learn ==")
    print(f"rows: {n}")

    # --- 1. k-fold cross-validation. The Pipeline refits the imputer and
    # scaler on each fold's training part — lesson 05's no-leakage rule,
    # automated. KFold's shuffle differs from our manual permutation, so the
    # folds (and per-fold numbers) differ from from_scratch.py; the mean
    # should be close. ---
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    cv = KFold(n_splits=K, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv)
    print()
    print(f"-- {K}-fold cross-validation (KFold, shuffle=True, random_state={SEED}) --")
    for k, acc in enumerate(scores):
        print(f"fold {k + 1}  accuracy: {acc:.4f}")
    print(f"mean accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    # --- 2. Fixed 80/20 split: identical rows and preprocessing to
    # from_scratch.py, so identical predictions go into the metrics. ---
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx].copy(), y[train_idx]
    X_test, y_test = X[test_idx].copy(), y[test_idx]

    age_median = float(np.nanmedian(X_train[:, 2]))
    for split in (X_train, X_test):
        split[np.isnan(split[:, 2]), 2] = age_median
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print()
    print(f"-- Fixed 80/20 split (lesson 05's): train {len(train_idx)}, test {len(test_idx)} --")
    print()
    print("confusion matrix (rows = actual, columns = predicted):")
    (tn, fp), (fn, tp) = confusion_matrix(y_test, pred)
    print(f"{'':<18} {'pred died':>10} {'pred survived':>14}")
    print(f"{'actual died':<18} {tn:>10} {fp:>14}")
    print(f"{'actual survived':<18} {fn:>10} {tp:>14}")

    print()
    print(f"accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"precision: {precision_score(y_test, pred):.4f}")
    print(f"recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1       : {f1_score(y_test, pred):.4f}")
    print()
    print(f"ROC-AUC  : {roc_auc_score(y_test, prob):.4f}")


if __name__ == "__main__":
    main()
