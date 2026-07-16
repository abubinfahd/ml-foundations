"""Lesson 04 — Linear regression, from scratch.

Fits quality ~ 11 physicochemical features of red wine two ways, using only
NumPy for the math: (1) the closed-form normal equation, (2) batch gradient
descent. The two solutions must agree. Compare against with_library.py,
which computes the same numbers with scikit-learn and statsmodels.

Run from the repo root:

    python src/lesson04_linear_regression/from_scratch.py
"""

import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "winequality-red.csv"
SEED = 42
LR = 0.1  # gradient descent learning rate
ITERS = 2000  # gradient descent iterations
CHECKPOINTS = (0, 100, 500, 2000)


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the semicolon-separated wine CSV: 11 features, target = quality."""
    with open(DATA) as f:
        header = [h.strip().strip('"') for h in f.readline().strip().split(";")]
    raw = np.genfromtxt(DATA, delimiter=";", skip_header=1)
    return raw[:, :-1], raw[:, -1], header[:-1]


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error: average squared residual."""
    return float(np.mean((y - y_hat) ** 2))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot: fraction of variance explained,
    relative to the always-predict-the-mean baseline."""
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot


def print_metrics(A_train, y_train, A_test, y_test, theta) -> None:
    yh_train, yh_test = A_train @ theta, A_test @ theta
    print(f"train MSE: {mse(y_train, yh_train):.4f}  train R^2: {r2(y_train, yh_train):.4f}")
    print(f"test  MSE: {mse(y_test, yh_test):.4f}  test  R^2: {r2(y_test, yh_test):.4f}")


def main() -> None:
    X, y, names = load_data()
    n = len(y)

    # Deterministic 80/20 split: shuffle all row indices once, first 80% train.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("== Linear regression: from scratch ==")
    print(f"rows: {n}  train: {len(y_train)}  test: {len(y_test)}")

    # Standardize with TRAIN statistics only — computing mean/std on all rows
    # would leak information about the test set into training.
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # Prepend a column of ones so theta[0] is the intercept.
    A_train = np.column_stack([np.ones(len(X_train)), X_train])
    A_test = np.column_stack([np.ones(len(X_test)), X_test])

    # --- Normal equation: solve (X^T X) theta = X^T y exactly. ---
    theta_ne = np.linalg.solve(A_train.T @ A_train, A_train.T @ y_train)

    print()
    print("-- Normal equation --")
    print(f"{'term':<22} {'coef':>9}")
    for name, coef in zip(["intercept"] + names, theta_ne):
        print(f"{name:<22} {coef:>9.4f}")
    print_metrics(A_train, y_train, A_test, y_test, theta_ne)

    # --- Batch gradient descent on the same standardized data. ---
    # Loss J(theta) = (1/n) ||A theta - y||^2 (the training MSE);
    # gradient = (2/n) A^T (A theta - y).
    print()
    print(f"-- Batch gradient descent (lr={LR}, {ITERS} iterations) --")
    theta_gd = np.zeros(A_train.shape[1])
    print(f"iter {0:>5}  train MSE: {mse(y_train, A_train @ theta_gd):.4f}")
    for it in range(1, ITERS + 1):
        grad = (2 / len(y_train)) * A_train.T @ (A_train @ theta_gd - y_train)
        theta_gd -= LR * grad
        if it in CHECKPOINTS:
            print(f"iter {it:>5}  train MSE: {mse(y_train, A_train @ theta_gd):.4f}")
    print_metrics(A_train, y_train, A_test, y_test, theta_gd)
    print(f"max |theta_gd - theta_ne|: {np.max(np.abs(theta_gd - theta_ne)):.4e}")


if __name__ == "__main__":
    main()
