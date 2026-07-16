"""Lesson 10 — PCA via covariance eigendecomposition, from scratch.

Standardizes the 11 wine features, builds the covariance matrix, and
eigendecomposes it with np.linalg.eigh. Eigenvectors are the principal
directions; eigenvalues are the variances along them. A fixed sign
convention (largest-|component| entry made positive) makes the output
reproducible. Compare against with_library.py, which uses sklearn's PCA
— every number must match to 4 decimals.

Run from the repo root:

    python src/lesson10_pca/from_scratch.py
"""

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "winequality-red.csv"

SEED = 42  # no randomness here, kept for convention


def load_data() -> tuple[np.ndarray, list[str]]:
    """Return (X standardized, feature_names). The file is SEMICOLON-separated;
    the last column (quality) is excluded. z-scoring uses ddof=0, the same
    convention as sklearn's StandardScaler."""
    with open(DATA, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        names = next(reader)[:-1]
        X = np.array([[float(v) for v in row[:-1]] for row in reader if row])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, names


def fix_signs(V: np.ndarray) -> np.ndarray:
    """Eigenvectors are defined only up to sign: flip each column so that its
    largest-absolute-value component is positive (first index wins ties)."""
    V = V.copy()
    for j in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    return V


def main() -> None:
    X, names = load_data()

    print("== PCA (covariance eigendecomposition): from scratch ==")
    print(f"rows: {len(X)}, features: {len(names)} (quality excluded), standardized")
    print()

    C = np.cov(X, rowvar=False)  # divides by n-1
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    order = np.argsort(-eigvals, kind="stable")  # descending, first wins ties
    eigvals = eigvals[order]
    V = fix_signs(eigvecs[:, order])

    ratio = eigvals / eigvals.sum()
    print(f"  {'PC':>4}  {'eigenvalue':>10}  {'explained':>9}  {'cumulative':>10}")
    for j in range(len(eigvals)):
        print(
            f"  {f'PC{j + 1}':>4}  {eigvals[j]:>10.4f}  {ratio[j]:>9.4f}  "
            f"{ratio[: j + 1].sum():>10.4f}"
        )
    print()

    for j in range(2):
        top = sorted(range(len(names)), key=lambda i: (-abs(V[i, j]), i))[:3]
        loads = ", ".join(f"{names[i]} {V[i, j]:+.4f}" for i in top)
        print(f"PC{j + 1} top-3 loadings by |value|: {loads}")
    print()

    Z = X @ V[:, :2]
    print("projection of the first 3 samples onto (PC1, PC2):")
    for i in range(3):
        print(f"  sample {i + 1}: ({Z[i, 0]:+.4f}, {Z[i, 1]:+.4f})")


if __name__ == "__main__":
    main()
