"""Lesson 02 — Probability & distributions, with scipy.

Verifies the closed-form numbers from from_scratch.py using scipy.stats:
the binomial pmf, the standard normal pdf, and the CLT prediction for
Exponential(rate=1). Print formats match the from-scratch run so the
numbers line up.

Run from the repo root:

    python src/lesson02_probability/with_library.py
"""

import math

from scipy import stats


def binomial() -> None:
    n, p = 10, 0.3
    print(f"-- Binomial(n={n}, p={p}) pmf: scipy.stats.binom --")
    print(f"{'k':>3} {'pmf':>10}")
    for k in range(n + 1):
        print(f"{k:>3} {stats.binom.pmf(k, n, p):>10.4f}")


def normal() -> None:
    print("-- Standard normal pdf: scipy.stats.norm --")
    print(f"{'x':>3} {'pdf(x)':>10}")
    for x in [-2, -1, 0, 1, 2]:
        print(f"{x:>3} {stats.norm.pdf(x):>10.4f}")


def central_limit_theorem() -> None:
    n = 30
    mu = stats.expon.mean()  # Exponential(rate=1): mean = 1
    sigma = stats.expon.std()  # Exponential(rate=1): std = 1
    print(f"-- CLT prediction for Exponential(rate=1), n={n}: scipy.stats.expon --")
    print(f"population mean: {mu:.4f}")
    print(f"population std : {sigma:.4f}")
    print(f"{'':<22} {'':>9} {'CLT prediction':>15}")
    print(f"{'mean of sample means':<22} {'':>9} {mu:>15.4f}")
    print(f"{'std  of sample means':<22} {'':>9} {sigma / math.sqrt(n):>15.4f}")


def main() -> None:
    print("== Probability & distributions: with scipy ==")
    print()
    binomial()
    print()
    normal()
    print()
    central_limit_theorem()


if __name__ == "__main__":
    main()
