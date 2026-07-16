"""Lesson 02 — Probability & distributions, from scratch.

Every probability here is either computed from a formula (math from the
standard library) or estimated by simulation with NumPy's random number
generator — no scipy. Compare against with_library.py, which verifies the
closed-form numbers with scipy.stats.

Run from the repo root:

    python src/lesson02_probability/from_scratch.py
"""

import math

import numpy as np

SEED = 42


def binomial_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p): C(n, k) * p^k * (1-p)^(n-k)."""
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Density of Normal(mu, sigma^2) at x."""
    z = (x - mu) / sigma
    return math.exp(-(z**2) / 2) / (sigma * math.sqrt(2 * math.pi))


def law_of_large_numbers() -> None:
    """Running proportion of heads in fair coin flips converges to 0.5."""
    rng = np.random.default_rng(SEED)
    flips = rng.integers(0, 2, size=100_000)  # 1 = heads, 0 = tails
    heads = flips.cumsum()

    print("-- Law of large numbers: fair coin, true P(heads) = 0.5 --")
    print(f"{'n':>8} {'prop(heads)':>12} {'|prop - 0.5|':>13}")
    for n in [10, 100, 1_000, 10_000, 100_000]:
        prop = heads[n - 1] / n
        print(f"{n:>8} {prop:>12.4f} {abs(prop - 0.5):>13.4f}")


def binomial() -> None:
    """Binomial pmf from the formula vs estimated from simulated draws."""
    rng = np.random.default_rng(SEED)
    n, p, n_draws = 10, 0.3, 100_000
    draws = rng.binomial(n, p, size=n_draws)
    counts = np.bincount(draws, minlength=n + 1)

    print(f"-- Binomial(n={n}, p={p}) pmf: formula vs {n_draws} simulated draws --")
    print(f"{'k':>3} {'formula':>10} {'simulated':>10}")
    for k in range(n + 1):
        print(f"{k:>3} {binomial_pmf(k, n, p):>10.4f} {counts[k] / n_draws:>10.4f}")


def normal() -> None:
    """Standard normal density at a few points."""
    print("-- Standard normal pdf --")
    print(f"{'x':>3} {'pdf(x)':>10}")
    for x in [-2, -1, 0, 1, 2]:
        print(f"{x:>3} {normal_pdf(x):>10.4f}")


def central_limit_theorem() -> None:
    """Means of Exponential(rate=1) samples look normal around 1 +/- 1/sqrt(n)."""
    rng = np.random.default_rng(SEED)
    n, n_means = 30, 10_000
    sample_means = rng.exponential(scale=1.0, size=(n_means, n)).mean(axis=1)

    print(f"-- Central limit theorem: {n_means} means of n={n} draws, Exponential(rate=1) --")
    print(f"{'':<22} {'observed':>9} {'CLT prediction':>15}")
    print(f"{'mean of sample means':<22} {sample_means.mean():>9.4f} {1.0:>15.4f}")
    print(f"{'std  of sample means':<22} {sample_means.std(ddof=1):>9.4f} {1 / math.sqrt(n):>15.4f}")


def main() -> None:
    print("== Probability & distributions: from scratch ==")
    print()
    law_of_large_numbers()
    print()
    binomial()
    print()
    normal()
    print()
    central_limit_theorem()


if __name__ == "__main__":
    main()
