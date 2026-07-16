"""Lesson 10 — PCA, with scikit-learn.

Runs PCA(n_components=None) on the same standardized wine features as
from_scratch.py. sklearn's own sign choice for components differs from
ours, so the same sign convention (largest-|component| entry made
positive) is applied before printing — after which every number must
match the from-scratch output to 4 decimals.

Run from the repo root:

    python src/lesson10_pca/with_library.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data" / "winequality-red.csv"

SEED = 42  # no randomness here, kept for convention


def load_data() -> tuple[np.ndarray, list[str]]:
    """Same as from_scratch.py: semicolon-separated file, quality excluded,
    features z-scored (StandardScaler uses ddof=0)."""
    with open(DATA, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        names = next(reader)[:-1]
        X = np.array([[float(v) for v in row[:-1]] for row in reader if row])
    X = StandardScaler().fit_transform(X)
    return X, names


def fix_signs(V: np.ndarray) -> np.ndarray:
    """Same sign convention as from_scratch.py, applied to columns."""
    V = V.copy()
    for j in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    return V


def main() -> None:
    X, names = load_data()

    print("== PCA: with scikit-learn ==")
    print(f"rows: {len(X)}, features: {len(names)} (quality excluded), standardized")
    print()

    pca = PCA(n_components=None)
    pca.fit(X)
    eigvals = pca.explained_variance_  # eigenvalues of the covariance matrix
    V = fix_signs(pca.components_.T)  # columns = principal directions

    ratio = pca.explained_variance_ratio_
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

    Z = X @ V[:, :2]  # X is already centered, so this is pca.transform + sign fix
    print("projection of the first 3 samples onto (PC1, PC2):")
    for i in range(3):
        print(f"  sample {i + 1}: ({Z[i, 0]:+.4f}, {Z[i, 1]:+.4f})")


if __name__ == "__main__":
    main()
