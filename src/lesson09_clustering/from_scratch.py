"""Lesson 09 — k-means clustering (Lloyd's algorithm), from scratch.

Implements Lloyd's alternating minimization with NumPy: random init,
assign points to the nearest centroid, recompute centroids, repeat until
the assignments stop changing. Runs 10 restarts and keeps the lowest
inertia, then traces the elbow curve for k=1..8. Compare against
with_library.py, which runs sklearn's KMeans on the same data.

Run from the repo root:

    python src/lesson09_clustering/from_scratch.py
"""

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SEED = 42
K = 3
N_RESTARTS = 10


def load_data() -> tuple[np.ndarray, list[str]]:
    """Return (X standardized, species). Rows with any missing value are
    dropped (344 -> 333); features are z-scored (mean 0, std 1, ddof=0 —
    the same convention as sklearn's StandardScaler)."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.DictReader(f) if all(v not in ("", "NA") for v in r.values())]
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, [r["species"] for r in rows]


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 300):
    """One run of Lloyd's algorithm. Init: k distinct random data points.
    Returns (labels, inertia, n_iter) where n_iter counts the assign/update
    passes performed before the assignments stopped changing."""
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.full(len(X), -1)
    n_iter = 0
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)  # ties go to the lowest cluster index
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        n_iter += 1
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:  # an empty cluster keeps its old centroid
                centroids[j] = members.mean(axis=0)
    inertia = float(((X - centroids[labels]) ** 2).sum())
    return labels, inertia, n_iter


def best_of(X: np.ndarray, k: int, seeds: np.ndarray):
    """Run kmeans once per seed, keep the lowest inertia (first wins ties)."""
    best = None
    for s in seeds:
        result = kmeans(X, k, np.random.default_rng(int(s)))
        if best is None or result[1] < best[1]:
            best = result
    return best


def main() -> None:
    X, species = load_data()
    species_names = sorted(set(species))
    # One deterministic seed per restart, derived from the master seed.
    seeds = np.random.default_rng(SEED).integers(0, 2**31 - 1, size=N_RESTARTS)

    print("== k-means clustering (Lloyd's algorithm): from scratch ==")
    print(f"rows used: {len(X)} (rows with missing values dropped)")
    print(f"features (standardized): {', '.join(FEATURES)}")
    print()

    labels, inertia, n_iter = best_of(X, K, seeds)
    print(f"k={K}, {N_RESTARTS} restarts (seed {SEED}):")
    print(f"best inertia: {inertia:.4f}")
    print(f"iterations of best run: {n_iter}")

    # Cluster indices are arbitrary; relabel by size (largest first, ties by
    # original index) so the output is stable and comparable across runs.
    sizes = np.bincount(labels, minlength=K)
    order = sorted(range(K), key=lambda j: (-sizes[j], j))
    print(f"cluster sizes (largest first): {', '.join(str(sizes[j]) for j in order)}")
    print()

    print("cluster x species contingency (clusters sorted by size):")
    print(f"  {'':<10}" + "".join(f"{s:>11}" for s in species_names))
    for rank, j in enumerate(order, start=1):
        counts = [
            sum(1 for lab, sp in zip(labels, species) if lab == j and sp == name)
            for name in species_names
        ]
        print(f"  {f'cluster {rank}':<10}" + "".join(f"{c:>11}" for c in counts))
    print()

    print(f"elbow: best inertia by k ({N_RESTARTS} restarts each):")
    for k in range(1, 9):
        _, k_inertia, _ = best_of(X, k, seeds)
        print(f"  k={k}  inertia={k_inertia:9.4f}")


if __name__ == "__main__":
    main()
