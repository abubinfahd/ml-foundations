"""Lesson 07 — Decision trees (CART), with scikit-learn.

Fits DecisionTreeClassifier(max_depth=3) on exactly the same 80/20 split
as from_scratch.py and prints the learned tree and accuracies. The two
trees should agree in substance; see the lesson doc for a node-by-node
comparison and why tie-breaking can make the structure differ.

Run from the repo root:

    python src/lesson07_decision_trees/with_library.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

DATA = Path(__file__).resolve().parents[2] / "data" / "iris.data"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SEED = 42


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Same loader as from_scratch.py: skip the trailing empty line, encode
    the class as an index into the sorted class names."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    classes = sorted({r[4] for r in rows})
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    y = np.array([classes.index(r[4]) for r in rows])
    return X, y, classes


def main() -> None:
    X, y, classes = load_data()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    train, test = perm[:n_train], perm[n_train:]

    print("== Decision tree (CART): with scikit-learn ==")
    print(f"rows in file: {len(y)}")
    print(f"train/test split: {len(train)}/{len(test)} (seed {SEED})")
    print(f"classes (count order): {', '.join(classes)}")
    print()

    clf = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    clf.fit(X[train], y[train])

    print(export_text(clf, feature_names=FEATURES, class_names=classes, decimals=4), end="")
    print()
    print(f"train accuracy: {clf.score(X[train], y[train]):.4f}")
    print(f"test accuracy:  {clf.score(X[test], y[test]):.4f}")


if __name__ == "__main__":
    main()
