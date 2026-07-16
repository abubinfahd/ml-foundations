"""Lesson 04 — Linear regression, with scikit-learn and statsmodels.

Reproduces the exact split and standardization of from_scratch.py, fits
scikit-learn's LinearRegression (the coefficients and metrics must match
the normal equation to 4 decimals), then fits the same model with
statsmodels OLS to introduce standard errors and p-values.

Run from the repo root:

    python src/lesson04_linear_regression/with_library.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA = Path(__file__).resolve().parents[2] / "data" / "winequality-red.csv"
SEED = 42
INFERENCE_FEATURES = ["alcohol", "volatile acidity", "sulphates"]


def main() -> None:
    df = pd.read_csv(DATA, sep=";")
    feature_names = [c for c in df.columns if c != "quality"]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["quality"].to_numpy(dtype=float)
    n = len(y)

    # Identical split to from_scratch.py: same rng, same shuffle, same cut.
    # (We deliberately avoid sklearn.model_selection.train_test_split so the
    # row assignment — and therefore every number — matches exactly.)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("== Linear regression: with scikit-learn & statsmodels ==")
    print(f"rows: {n}  train: {len(y_train)}  test: {len(y_test)}")

    # Standardize with TRAIN statistics only, exactly as in from_scratch.py.
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    # --- scikit-learn: fit_intercept=True adds the intercept for us. ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    print()
    print("-- scikit-learn LinearRegression --")
    print(f"{'term':<22} {'coef':>9}")
    for name, coef in zip(["intercept"] + feature_names, [model.intercept_] + list(model.coef_)):
        print(f"{name:<22} {coef:>9.4f}")
    yh_train, yh_test = model.predict(X_train), model.predict(X_test)
    print(f"train MSE: {mean_squared_error(y_train, yh_train):.4f}  train R^2: {r2_score(y_train, yh_train):.4f}")
    print(f"test  MSE: {mean_squared_error(y_test, yh_test):.4f}  test  R^2: {r2_score(y_test, yh_test):.4f}")

    # --- statsmodels: same fit, plus inference (std errors, p-values). ---
    A_train = sm.add_constant(pd.DataFrame(X_train, columns=feature_names))
    ols = sm.OLS(y_train, A_train).fit()

    print()
    print("-- statsmodels OLS (train split) --")
    print(f"R-squared: {ols.rsquared:.4f}")
    print(f"{'feature':<18} {'coef':>9} {'std err':>9} {'p-value':>12}")
    for name in INFERENCE_FEATURES:
        print(
            f"{name:<18} {ols.params[name]:>9.4f} {ols.bse[name]:>9.4f} "
            f"{ols.pvalues[name]:>12.4e}"
        )


if __name__ == "__main__":
    main()
