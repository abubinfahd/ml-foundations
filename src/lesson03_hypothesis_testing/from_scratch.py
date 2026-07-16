"""Lesson 03 — Hypothesis testing, from scratch.

Welch's t-test and the chi-square test of independence, computed with plain
Python (csv + math from the standard library) plus NumPy — no scipy. P-values
come from permutation tests, not from t or chi-square tables. Compare against
with_library.py, which gets analytic p-values from scipy.stats.

Run from the repo root:

    python src/lesson03_hypothesis_testing/from_scratch.py
"""

import csv
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

SEED = 42
N_PERMUTATIONS = 10_000


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def variance(x: list[float]) -> float:
    """Sample variance: divides by n-1 (Bessel's correction)."""
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t statistic and Welch-Satterthwaite degrees of freedom."""
    na, nb = len(a), len(b)
    va, vb = variance(a) / na, variance(b) / nb
    t = (mean(a) - mean(b)) / math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    return t, dof


def format_p(count: int, total: int) -> str:
    """Permutation p-value, or '< 1/total' when no permutation was as extreme."""
    if count == 0:
        return f"< {1 / total}"
    return f"{count / total:.4f}"


def chi_square_stat(observed: np.ndarray) -> tuple[np.ndarray, float]:
    """Expected counts under independence and the chi-square statistic."""
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals * col_totals / observed.sum()
    return expected, float(((observed - expected) ** 2 / expected).sum())


def t_test(rows: list[dict]) -> None:
    """Do Adelie and Chinstrap penguins differ in mean bill length?"""
    groups = {}
    for species in ["Adelie", "Chinstrap"]:
        groups[species] = [
            float(r["bill_length_mm"])
            for r in rows
            if r["species"] == species and r["bill_length_mm"] not in ("", "NA")
        ]
    a, b = groups["Adelie"], groups["Chinstrap"]

    print("-- Welch t-test: bill_length_mm, Adelie vs Chinstrap --")
    for species in ["Adelie", "Chinstrap"]:
        x = groups[species]
        print(f"  {species:<10} n={len(x):>3}  mean={mean(x):.4f}  std={math.sqrt(variance(x)):.4f}")
    t, dof = welch_t(a, b)
    observed_diff = mean(a) - mean(b)
    print(f"difference in means (Adelie - Chinstrap): {observed_diff:.4f}")
    print(f"Welch t statistic: {t:.4f}")
    print(f"Welch-Satterthwaite dof: {dof:.4f}")

    # Permutation test: under H0 the group labels are arbitrary, so shuffling
    # them generates the distribution of the difference when H0 is true.
    rng = np.random.default_rng(SEED)
    pooled = np.array(a + b)
    na = len(a)
    count = 0
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(pooled)
        perm_diff = perm[:na].mean() - perm[na:].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    print(f"permutation test ({N_PERMUTATIONS} shuffles, seed {SEED}):")
    print(f"  permutations with |diff| >= |observed|: {count} of {N_PERMUTATIONS}")
    print(f"  two-sided p-value: {format_p(count, N_PERMUTATIONS)}")


def chi_square_test(rows: list[dict]) -> None:
    """Are species and island independent?"""
    species_labels = sorted({r["species"] for r in rows})
    island_labels = sorted({r["island"] for r in rows})
    s_index = {s: i for i, s in enumerate(species_labels)}
    i_index = {s: i for i, s in enumerate(island_labels)}
    s_codes = np.array([s_index[r["species"]] for r in rows])
    i_codes = np.array([i_index[r["island"]] for r in rows])
    n_s, n_i = len(species_labels), len(island_labels)

    def table(i_codes: np.ndarray) -> np.ndarray:
        return np.bincount(s_codes * n_i + i_codes, minlength=n_s * n_i).reshape(n_s, n_i)

    observed = table(i_codes)
    expected, chi2 = chi_square_stat(observed)

    print("-- Chi-square test of independence: species vs island --")
    header = f"  {'':<10}" + "".join(f"{c:>11}" for c in island_labels)
    print("observed counts:")
    print(header)
    for s, row in zip(species_labels, observed):
        print(f"  {s:<10}" + "".join(f"{v:>11}" for v in row))
    print("expected counts under independence:")
    print(header)
    for s, row in zip(species_labels, expected):
        print(f"  {s:<10}" + "".join(f"{v:>11.4f}" for v in row))
    print(f"chi-square statistic: {chi2:.4f}")
    print(f"dof: ({n_s} - 1) * ({n_i} - 1) = {(n_s - 1) * (n_i - 1)}")

    # Permutation test: under H0 (independence) the island labels carry no
    # information about species, so shuffling them generates the H0 world.
    rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(N_PERMUTATIONS):
        _, perm_chi2 = chi_square_stat(table(rng.permutation(i_codes)))
        if perm_chi2 >= chi2:
            count += 1
    print(f"permutation test ({N_PERMUTATIONS} shuffles of island labels, seed {SEED}):")
    print(f"  permutations with chi2 >= observed: {count} of {N_PERMUTATIONS}")
    print(f"  p-value: {format_p(count, N_PERMUTATIONS)}")


def main() -> None:
    with open(DATA, newline="") as f:
        rows = list(csv.DictReader(f))

    print("== Hypothesis testing: from scratch ==")
    print(f"rows in file: {len(rows)}")
    print()
    t_test(rows)
    print()
    chi_square_test(rows)


if __name__ == "__main__":
    main()
