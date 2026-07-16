"""Lesson 05 — Logistic regression, from scratch.

Predicts Titanic survival from 6 passenger features. The model is a linear
score squashed through the sigmoid; training minimizes binary cross-entropy
by batch gradient descent, all in NumPy. Compare against with_library.py,
which fits the same model with scikit-learn.

Run from the repo root:

    python src/lesson05_logistic_regression/from_scratch.py
"""

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
SEED = 42
LR = 0.5  # gradient descent learning rate
ITERS = 20000  # gradient descent iterations
CHECKPOINTS = (0, 10, 100, 1000, 20000)

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Features: Pclass, Sex (male=1, female=0), Age (NaN if missing),
    SibSp, Parch, Fare. Target: Survived."""
    with open(DATA, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array(
        [
            [
                float(r["Pclass"]),
                1.0 if r["Sex"] == "male" else 0.0,
                float(r["Age"]) if r["Age"] != "" else np.nan,
                float(r["SibSp"]),
                float(r["Parch"]),
                float(r["Fare"]),
            ]
            for r in rows
        ]
    )
    y = np.array([float(r["Survived"]) for r in rows])
    return X, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Map any real score to a probability in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy: the negative mean Bernoulli log-likelihood."""
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Fraction correct after thresholding the probability at 0.5."""
    return float(np.mean((p >= 0.5) == y))


def main() -> None:
    X, y = load_data()
    n = len(y)

    # Deterministic 80/20 split: same procedure as lesson 04.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("== Logistic regression: from scratch ==")
    print(f"rows: {n}  train: {len(y_train)}  test: {len(y_test)}")

    # Impute missing Age with the TRAIN median (train-only, to avoid leakage),
    # then standardize with TRAIN mean/std — also train-only.
    age_median = float(np.nanmedian(X_train[:, 2]))
    print(f"train median Age used for imputation: {age_median:.4f}")
    for split in (X_train, X_test):
        split[np.isnan(split[:, 2]), 2] = age_median
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    A_train = np.column_stack([np.ones(len(X_train)), X_train])
    A_test = np.column_stack([np.ones(len(X_test)), X_test])

    # Batch gradient descent on binary cross-entropy.
    # For p = sigmoid(A theta), the gradient is (1/n) A^T (p - y) — exactly
    # the same "features times (prediction - truth)" form as linear
    # regression. Not a coincidence: sigmoid + cross-entropy are paired so
    # the awkward sigmoid derivative cancels (both are the canonical
    # link/loss of an exponential-family model).
    print()
    print(f"-- Batch gradient descent (lr={LR}, {ITERS} iterations) --")
    theta = np.zeros(A_train.shape[1])
    print(f"iter {0:>5}  train log-loss: {log_loss(y_train, sigmoid(A_train @ theta)):.4f}")
    for it in range(1, ITERS + 1):
        p = sigmoid(A_train @ theta)
        grad = A_train.T @ (p - y_train) / len(y_train)
        theta -= LR * grad
        if it in CHECKPOINTS:
            print(f"iter {it:>5}  train log-loss: {log_loss(y_train, sigmoid(A_train @ theta)):.4f}")

    print()
    print(f"{'term':<11} {'coef':>9}")
    for name, coef in zip(["intercept"] + FEATURES, theta):
        print(f"{name:<11} {coef:>9.4f}")

    p_train, p_test = sigmoid(A_train @ theta), sigmoid(A_test @ theta)
    print()
    print(f"train accuracy: {accuracy(y_train, p_train):.4f}  train log-loss: {log_loss(y_train, p_train):.4f}")
    print(f"test  accuracy: {accuracy(y_test, p_test):.4f}  test  log-loss: {log_loss(y_test, p_test):.4f}")


if __name__ == "__main__":
    main()
