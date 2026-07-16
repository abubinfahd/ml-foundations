"""Lesson 03 — Hypothesis testing, with scipy.

Runs the same two tests as from_scratch.py — Welch's t-test (Adelie vs
Chinstrap bill length) and the chi-square test of independence (species vs
island) — using scipy.stats, which computes analytic p-values from the t and
chi-square distributions instead of permutations.

Run from the repo root:

    python src/lesson03_hypothesis_testing/with_library.py
"""

from pathlib import Path

import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"


def main() -> None:
    df = pd.read_csv(DATA)

    print("== Hypothesis testing: with scipy ==")
    print(f"rows in file: {len(df)}")
    print()

    print("-- Welch t-test: bill_length_mm, Adelie vs Chinstrap --")
    bill = df.dropna(subset=["bill_length_mm"])
    groups = {}
    for species in ["Adelie", "Chinstrap"]:
        x = bill.loc[bill["species"] == species, "bill_length_mm"]
        groups[species] = x
        print(f"  {species:<10} n={len(x):>3}  mean={x.mean():.4f}  std={x.std():.4f}")
    res = stats.ttest_ind(groups["Adelie"], groups["Chinstrap"], equal_var=False)
    print(f"Welch t statistic: {res.statistic:.4f}")
    print(f"Welch-Satterthwaite dof: {res.df:.4f}")
    print(f"two-sided p-value: {res.pvalue:.4e}")
    print()

    print("-- Chi-square test of independence: species vs island --")
    table = pd.crosstab(df["species"], df["island"]).sort_index(axis=0).sort_index(axis=1)
    chi2 = stats.chi2_contingency(table)
    print("observed counts:")
    print(f"  {'':<10}" + "".join(f"{c:>11}" for c in table.columns))
    for species, row in table.iterrows():
        print(f"  {species:<10}" + "".join(f"{v:>11}" for v in row))
    print(f"chi-square statistic: {chi2.statistic:.4f}")
    print(f"dof: {chi2.dof}")
    print(f"p-value: {chi2.pvalue:.4e}")


if __name__ == "__main__":
    main()
