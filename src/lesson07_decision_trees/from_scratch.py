"""Lesson 07 — Decision trees (CART), from scratch.

Implements CART for classification with NumPy: Gini impurity, exhaustive
best-split search over midpoint thresholds, and greedy recursive
partitioning capped at max_depth=3. Compare against with_library.py,
which fits sklearn's DecisionTreeClassifier on the identical split.

Run from the repo root:

    python src/lesson07_decision_trees/from_scratch.py
"""

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "iris.data"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SEED = 42
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 1


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, class_names). iris.data has no header and ends with an
    empty line, which we skip. y holds indices into the sorted class names."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.reader(f) if r]  # skip the trailing empty line
    classes = sorted({r[4] for r in rows})
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    y = np.array([classes.index(r[4]) for r in rows])
    return X, y, classes


def gini(counts: np.ndarray) -> float:
    """Gini impurity G = 1 - sum_k p_k^2 from class counts."""
    p = counts / counts.sum()
    return 1.0 - float((p**2).sum())


def best_split(X: np.ndarray, y: np.ndarray, n_classes: int):
    """Return (decrease, feature, threshold) of the best split, or None.

    Candidate thresholds are the midpoints between consecutive sorted unique
    values of each feature. Ties in impurity decrease are broken by the first
    (lowest feature index, then lowest threshold) candidate examined.
    """
    n = len(y)
    parent_counts = np.bincount(y, minlength=n_classes)
    parent_gini = gini(parent_counts)
    best = None  # (decrease, feature, threshold)
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])  # sorted unique values
        for lo, hi in zip(values[:-1], values[1:]):
            thr = (lo + hi) / 2
            mask = X[:, f] <= thr
            n_left = int(mask.sum())
            if n_left < MIN_SAMPLES_LEAF or n - n_left < MIN_SAMPLES_LEAF:
                continue
            left = np.bincount(y[mask], minlength=n_classes)
            right = parent_counts - left
            child = (n_left * gini(left) + (n - n_left) * gini(right)) / n
            decrease = parent_gini - child
            if best is None or decrease > best[0]:  # strict > keeps the first tie
                best = (decrease, f, thr)
    return best


def build(X: np.ndarray, y: np.ndarray, n_classes: int, depth: int) -> dict:
    """Recursively grow the tree (greedy, top-down)."""
    counts = np.bincount(y, minlength=n_classes)
    # np.argmax returns the first maximum, i.e. ties go to the class earliest
    # in sorted class-name order.
    leaf = {"leaf": True, "counts": counts, "pred": int(np.argmax(counts))}
    if depth == MAX_DEPTH or gini(counts) == 0.0 or len(y) < 2 * MIN_SAMPLES_LEAF:
        return leaf
    split = best_split(X, y, n_classes)
    if split is None or split[0] <= 0.0:
        return leaf
    _, f, thr = split
    mask = X[:, f] <= thr
    return {
        "leaf": False,
        "feature": f,
        "threshold": thr,
        "left": build(X[mask], y[mask], n_classes, depth + 1),
        "right": build(X[~mask], y[~mask], n_classes, depth + 1),
    }


def print_tree(node: dict, classes: list[str], indent: str = "") -> None:
    if node["leaf"]:
        counts = ", ".join(str(c) for c in node["counts"])
        print(f"{indent}|--- leaf: {classes[node['pred']]}  counts=[{counts}]")
        return
    name, thr = FEATURES[node["feature"]], node["threshold"]
    print(f"{indent}|--- {name} <= {thr:.4f}")
    print_tree(node["left"], classes, indent + "|   ")
    print(f"{indent}|--- {name} >  {thr:.4f}")
    print_tree(node["right"], classes, indent + "|   ")


def predict_one(node: dict, x: np.ndarray) -> int:
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["pred"]


def accuracy(node: dict, X: np.ndarray, y: np.ndarray) -> float:
    pred = np.array([predict_one(node, x) for x in X])
    return float((pred == y).mean())


def main() -> None:
    X, y, classes = load_data()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    train, test = perm[:n_train], perm[n_train:]

    print("== Decision tree (CART): from scratch ==")
    print(f"rows in file: {len(y)}")
    print(f"train/test split: {len(train)}/{len(test)} (seed {SEED})")
    print(f"classes (count order): {', '.join(classes)}")
    print()

    tree = build(X[train], y[train], len(classes), depth=0)
    print_tree(tree, classes)
    print()
    print(f"train accuracy: {accuracy(tree, X[train], y[train]):.4f}")
    print(f"test accuracy:  {accuracy(tree, X[test], y[test]):.4f}")


if __name__ == "__main__":
    main()
