"""Lesson 06 — Model evaluation, from scratch.

The model under evaluation is scikit-learn's LogisticRegression (lesson 05's
features) — deliberately NOT reimplemented here. What is implemented from
scratch is everything used to judge it: 5-fold cross-validation, the
confusion matrix, precision/recall/F1, and ROC-AUC computed two independent
ways. Compare against with_library.py, which uses scikit-learn's metrics.

Run from the repo root:

    python src/lesson06_model_evaluation/from_scratch.py
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

DATA = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
SEED = 42
K = 5

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Same features as lesson 05: Pclass, Sex (male=1), Age (NaN if
    missing), SibSp, Parch, Fare. Target: Survived."""
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


def preprocess(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lesson 05's preprocessing: impute Age with the TRAIN median, then
    standardize with TRAIN mean/std. Done inside every fold, so no fold ever
    leaks statistics into the model evaluated on it."""
    X_train, X_eval = X_train.copy(), X_eval.copy()
    age_median = float(np.nanmedian(X_train[:, 2]))
    for split in (X_train, X_eval):
        split[np.isnan(split[:, 2]), 2] = age_median
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_eval - mu) / sigma


def fit_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(X_train, y_train)


def roc_auc_trapezoid(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    """ROC-AUC by construction: sweep the threshold from high to low. Each
    unique probability value contributes one (FPR, TPR) point (rows tied on
    probability enter together); integrate with the trapezoidal rule."""
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    tp = np.cumsum(y_sorted)  # positives captured at each cut
    fp = np.cumsum(1 - y_sorted)  # negatives wrongly captured
    last_of_tie = np.r_[np.nonzero(np.diff(p_sorted))[0], len(p_sorted) - 1]
    tpr = np.r_[0.0, tp[last_of_tie] / n_pos]
    fpr = np.r_[0.0, fp[last_of_tie] / n_neg]
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return auc, len(tpr)


def roc_auc_rank(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC as the Mann-Whitney statistic: the probability that a random
    positive is scored above a random negative (ties count half). Computed
    from ranks — no thresholds, no curve — as a cross-check."""
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.empty(len(p))
    i = 0
    while i < len(p):  # assign midranks to ties
        j = i
        while j + 1 < len(p) and p_sorted[j + 1] == p_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average of ranks i+1..j+1
        i = j + 1
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    X, y = load_data()
    n = len(y)

    print("== Model evaluation: from scratch ==")
    print(f"rows: {n}")

    # --- 1. k-fold cross-validation, by hand. ---
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    folds = np.array_split(perm, K)
    print()
    print(f"-- {K}-fold cross-validation (manual folds) --")
    accs = []
    for k, fold in enumerate(folds):
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])
        X_tr, X_te = preprocess(X[train_idx], X[fold])
        model = fit_model(X_tr, y[train_idx])
        acc = float(np.mean(model.predict(X_te) == y[fold]))
        accs.append(acc)
        print(f"fold {k + 1}  n={len(fold)}  accuracy: {acc:.4f}")
    accs = np.array(accs)
    print(f"mean accuracy: {accs.mean():.4f} +/- {accs.std():.4f}")

    # --- 2. Confusion matrix, precision, recall, F1 on one fixed split. ---
    n_train = int(0.8 * n)
    perm2 = np.random.default_rng(SEED).permutation(n)  # lesson 05's split
    train_idx, test_idx = perm2[:n_train], perm2[n_train:]
    X_tr, X_te = preprocess(X[train_idx], X[test_idx])
    y_test = y[test_idx]
    model = fit_model(X_tr, y[train_idx])
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]

    tp = int(np.sum((pred == 1) & (y_test == 1)))
    fp = int(np.sum((pred == 1) & (y_test == 0)))
    fn = int(np.sum((pred == 0) & (y_test == 1)))
    tn = int(np.sum((pred == 0) & (y_test == 0)))

    print()
    print(f"-- Fixed 80/20 split (lesson 05's): train {len(train_idx)}, test {len(test_idx)} --")
    print()
    print("confusion matrix (rows = actual, columns = predicted):")
    print(f"{'':<18} {'pred died':>10} {'pred survived':>14}")
    print(f"{'actual died':<18} {tn:>10} {fp:>14}")
    print(f"{'actual survived':<18} {fn:>10} {tp:>14}")

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print()
    print(f"accuracy : {(tp + tn) / len(y_test):.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    # --- 3. ROC-AUC, two independent ways. ---
    auc_trap, n_points = roc_auc_trapezoid(y_test, prob)
    auc_rank = roc_auc_rank(y_test, prob)
    print()
    print(f"ROC curve points (unique thresholds + origin): {n_points}")
    print(f"AUC via threshold sweep + trapezoid: {auc_trap:.4f}")
    print(f"AUC via Mann-Whitney rank formula  : {auc_rank:.4f}")


if __name__ == "__main__":
    main()
