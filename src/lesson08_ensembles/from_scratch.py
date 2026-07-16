"""Lesson 08 — Ensembles (bagging / random forest), from scratch.

The ensemble logic is the lesson here, not the tree: the base learner is
sklearn's DecisionTreeClassifier (Lesson 07 showed what is inside it).
We implement bagging ourselves — bootstrap sampling, optional sqrt(d)
feature subsampling per tree, majority voting — and watch test accuracy
climb as the ensemble grows. Compare against with_library.py, which fits
sklearn's RandomForestClassifier and HistGradientBoostingClassifier.

Run from the repo root:

    python src/lesson08_ensembles/from_scratch.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

DATA = Path(__file__).resolve().parents[2] / "data" / "adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "sex"]
SEED = 42
N_TREES = 25


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Return (X, y, feature_names, rows_in_file).

    adult.data has no header, is ", "-separated, and marks missing values
    with "?". We drop every row containing a "?" and one-hot encode the
    CATEGORICAL columns in sorted category order (deterministic).
    """
    with open(DATA, newline="") as f:
        raw = [r for r in csv.reader(f, skipinitialspace=True) if r]
    rows = [dict(zip(COLUMNS, r)) for r in raw if "?" not in r]
    categories = {c: sorted({r[c] for r in rows}) for c in CATEGORICAL}
    names = list(NUMERIC) + [f"{c}={v}" for c in CATEGORICAL for v in categories[c]]
    X = np.zeros((len(rows), len(names)))
    for i, r in enumerate(rows):
        for j, c in enumerate(NUMERIC):
            X[i, j] = float(r[c])
        col = len(NUMERIC)
        for c in CATEGORICAL:
            X[i, col + categories[c].index(r[c])] = 1.0
            col += len(categories[c])
    y = np.array([1 if r["income"] == ">50K" else 0 for r in rows])
    return X, y, names, len(raw)


def fit_ensemble(
    X: np.ndarray, y: np.ndarray, n_trees: int, max_features: int | None
) -> list[tuple[DecisionTreeClassifier, np.ndarray]]:
    """Bagging: each tree sees a bootstrap sample of the rows and (optionally)
    a random subset of max_features columns. Tree i gets its own rng seeded
    from SEED + i, so the ensemble is fully deterministic."""
    n, d = X.shape
    trees = []
    for i in range(n_trees):
        rng = np.random.default_rng(SEED + i)
        boot = rng.integers(0, n, size=n)  # sample n rows *with replacement*
        if max_features is None:
            feats = np.arange(d)
        else:
            feats = np.sort(rng.choice(d, size=max_features, replace=False))
        tree = DecisionTreeClassifier(max_depth=None, random_state=i)
        tree.fit(X[boot][:, feats], y[boot])
        trees.append((tree, feats))
    return trees


def vote(trees: list, X: np.ndarray) -> np.ndarray:
    """Majority vote over the trees. An exact tie (possible when the number
    of trees is even) goes to class 0 ('<=50K', first in sorted label order)."""
    votes = np.zeros(len(X))
    for tree, feats in trees:
        votes += tree.predict(X[:, feats])
    return (2 * votes > len(trees)).astype(int)


def main() -> None:
    X, y, names, rows_in_file = load_data()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    train, test = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    print("== Ensembles (bagging / random forest): from scratch ==")
    print(f"rows in file: {rows_in_file}")
    print(f"rows after dropping '?': {len(y)}")
    print(f"features: {len(NUMERIC)} numeric + {len(names) - len(NUMERIC)} one-hot = {len(names)}")
    print(f"train/test split: {len(train)}/{len(test)} (seed {SEED})")
    print()

    single = DecisionTreeClassifier(max_depth=None, random_state=0)
    single.fit(X_train, y_train)
    print("single deep tree (max_depth=None), the overfit baseline:")
    print(f"  train accuracy: {single.score(X_train, y_train):.4f}")
    print(f"  test accuracy:  {single.score(X_test, y_test):.4f}")
    print()

    k = int(np.sqrt(X.shape[1]))  # sqrt(d) features per tree
    bagged = fit_ensemble(X_train, y_train, N_TREES, max_features=None)
    forest = fit_ensemble(X_train, y_train, N_TREES, max_features=k)

    acc_bag = float((vote(bagged, X_test) == y_test).mean())
    acc_rf = float((vote(forest, X_test) == y_test).mean())
    print(f"bagging, {N_TREES} deep trees (all features):")
    print(f"  test accuracy:  {acc_bag:.4f}")
    print(f"random-forest-style, {N_TREES} deep trees (sqrt(d)={k} features per tree):")
    print(f"  test accuracy:  {acc_rf:.4f}")
    print()

    print("test accuracy as the ensemble grows (same fitted trees):")
    print(f"  {'trees':>5}  {'bagging':>8}  {'rf-style':>8}")
    for m in [1, 5, 10, 25]:
        a_bag = float((vote(bagged[:m], X_test) == y_test).mean())
        a_rf = float((vote(forest[:m], X_test) == y_test).mean())
        print(f"  {m:>5}  {a_bag:>8.4f}  {a_rf:>8.4f}")


if __name__ == "__main__":
    main()
