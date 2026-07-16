"""Lesson 09 — k-means clustering, with scikit-learn.

Runs KMeans(n_clusters=3, n_init=10) on the same standardized penguin
measurements as from_scratch.py. The inertia should match the
from-scratch run to 4 decimals — same optimum, found from different
starting points — and the adjusted Rand index quantifies how well the
clusters recover the species labels k-means never saw.

Run from the repo root:

    python src/lesson09_clustering/with_library.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SEED = 42
K = 3


def load_data() -> tuple[np.ndarray, list[str]]:
    """Same preprocessing as from_scratch.py: drop rows with any missing
    value (344 -> 333), z-score the four measurement columns."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.DictReader(f) if all(v not in ("", "NA") for v in r.values())]
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    X = StandardScaler().fit_transform(X)
    return X, [r["species"] for r in rows]


def main() -> None:
    X, species = load_data()
    species_names = sorted(set(species))

    print("== k-means clustering: with scikit-learn ==")
    print(f"rows used: {len(X)} (rows with missing values dropped)")
    print(f"features (standardized): {', '.join(FEATURES)}")
    print()

    km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
    labels = km.fit_predict(X)
    print(f"KMeans(n_clusters={K}, n_init=10, random_state={SEED}):")
    print(f"inertia: {km.inertia_:.4f}")

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

    ari = adjusted_rand_score(species, labels)
    print(f"adjusted Rand index (clusters vs species): {ari:.4f}")


if __name__ == "__main__":
    main()
