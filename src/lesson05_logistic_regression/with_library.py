"""Lesson 05 — Logistic regression, with scikit-learn.

Builds the exact same feature matrix, split, imputation and standardization
as from_scratch.py (pandas for loading, NumPy for the split), then fits
scikit-learn's LogisticRegression with C=1e9 — regularization turned off in
practice — so its solution is the same maximum-likelihood fit that gradient
descent found. Also fits the default C=1.0 as a teaser for regularization.

Run from the repo root:

    python src/lesson05_logistic_regression/with_library.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

DATA = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
SEED = 42

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]


def main() -> None:
    df = pd.read_csv(DATA)
    df["Sex"] = (df["Sex"] == "male").astype(float)  # male=1, female=0
    X = df[FEATURES].to_numpy(dtype=float)  # Age keeps NaN where missing
    y = df["Survived"].to_numpy(dtype=float)
    n = len(y)

    # Identical split to from_scratch.py: same rng, same shuffle, same cut.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("== Logistic regression: with scikit-learn ==")
    print(f"rows: {n}  train: {len(y_train)}  test: {len(y_test)}")

    # Identical preprocessing: TRAIN-median Age imputation, TRAIN-stats
    # standardization.
    age_median = float(np.nanmedian(X_train[:, 2]))
    print(f"train median Age used for imputation: {age_median:.4f}")
    for split in (X_train, X_test):
        split[np.isnan(split[:, 2]), 2] = age_median
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # C is the INVERSE regularization strength; C=1e9 makes the L2 penalty
    # negligible, matching the unpenalized from-scratch objective.
    model = LogisticRegression(C=1e9, max_iter=10000, tol=1e-10)
    model.fit(X_train, y_train)

    print()
    print("-- LogisticRegression(C=1e9), effectively unregularized --")
    print(f"{'term':<11} {'coef':>9}")
    for name, coef in zip(["intercept"] + FEATURES, [model.intercept_[0]] + list(model.coef_[0])):
        print(f"{name:<11} {coef:>9.4f}")

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    print()
    print(
        f"train accuracy: {accuracy_score(y_train, p_train >= 0.5):.4f}  "
        f"train log-loss: {log_loss(y_train, p_train):.4f}"
    )
    print(
        f"test  accuracy: {accuracy_score(y_test, p_test >= 0.5):.4f}  "
        f"test  log-loss: {log_loss(y_test, p_test):.4f}"
    )

    # Teaser for lesson 06 and beyond: scikit-learn's default applies L2
    # regularization with C=1.0. More on why you'd want that later.
    default_model = LogisticRegression(max_iter=10000)
    default_model.fit(X_train, y_train)
    print()
    print(
        f"default LogisticRegression (C=1.0) test accuracy: "
        f"{default_model.score(X_test, y_test):.4f}"
    )


if __name__ == "__main__":
    main()
