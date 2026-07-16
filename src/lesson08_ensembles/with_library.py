"""Lesson 08 — Ensembles, with scikit-learn.

Fits RandomForestClassifier (bagging + feature subsampling, 100 trees)
and HistGradientBoostingClassifier (sequential boosting) on exactly the
same preprocessed data and 80/20 split as from_scratch.py, and prints
the forest's top-5 feature importances.

Run from the repo root:

    python src/lesson08_ensembles/with_library.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

DATA = Path(__file__).resolve().parents[2] / "data" / "adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
NUMERIC = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "sex"]
SEED = 42


def load_data() -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Same preprocessing as from_scratch.py: drop rows containing '?',
    one-hot encode the CATEGORICAL columns in sorted category order."""
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


def main() -> None:
    X, y, names, rows_in_file = load_data()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    train, test = perm[:n_train], perm[n_train:]

    print("== Ensembles: with scikit-learn ==")
    print(f"rows in file: {rows_in_file}")
    print(f"rows after dropping '?': {len(y)}")
    print(f"features: {len(NUMERIC)} numeric + {len(names) - len(NUMERIC)} one-hot = {len(names)}")
    print(f"train/test split: {len(train)}/{len(test)} (seed {SEED})")
    print()

    forest = RandomForestClassifier(n_estimators=100, random_state=SEED)
    forest.fit(X[train], y[train])
    print("random forest (100 trees):")
    print(f"  test accuracy:  {forest.score(X[test], y[test]):.4f}")

    boost = HistGradientBoostingClassifier(random_state=SEED)
    boost.fit(X[train], y[train])
    print("hist gradient boosting:")
    print(f"  test accuracy:  {boost.score(X[test], y[test]):.4f}")
    print()

    print("top-5 feature importances (random forest):")
    order = sorted(range(len(names)), key=lambda j: (-forest.feature_importances_[j], j))
    for j in order[:5]:
        print(f"  {names[j]:<34} {forest.feature_importances_[j]:.4f}")


if __name__ == "__main__":
    main()
