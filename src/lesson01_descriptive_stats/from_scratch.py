"""Lesson 01 — Descriptive statistics, from scratch.

Every statistic here is computed with plain Python (csv + math from the
standard library) so the formulas are visible as code. Compare against
with_library.py, which computes the same numbers with pandas.

Run from the repo root:

    python src/lesson01_descriptive_stats/from_scratch.py
"""

import csv
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def load_column(rows: list[dict], col: str) -> list[float]:
    """Return the non-missing values of a column as floats."""
    return [float(r[col]) for r in rows if r[col] not in ("", "NA")]


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def median(x: list[float]) -> float:
    s = sorted(x)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def variance(x: list[float]) -> float:
    """Sample variance: divides by n-1 (Bessel's correction)."""
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def std(x: list[float]) -> float:
    return math.sqrt(variance(x))


def quantile(x: list[float], q: float) -> float:
    """Quantile with linear interpolation (the NumPy/pandas default)."""
    s = sorted(x)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    frac = h - lo
    if lo + 1 < len(s):
        return s[lo] + frac * (s[lo + 1] - s[lo])
    return s[lo]


def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation: covariance scaled by both standard deviations."""
    mx, my = mean(x), mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy)


def main() -> None:
    with open(DATA, newline="") as f:
        rows = list(csv.DictReader(f))

    print("== Descriptive statistics: from scratch ==")
    print(f"rows in file: {len(rows)}")
    print()
    print(f"{'column':<20} {'n':>4} {'mean':>10} {'median':>10} {'std':>9} {'q25':>9} {'q75':>9}")
    for col in NUMERIC_COLS:
        x = load_column(rows, col)
        print(
            f"{col:<20} {len(x):>4} {mean(x):>10.4f} {median(x):>10.4f} "
            f"{std(x):>9.4f} {quantile(x, 0.25):>9.4f} {quantile(x, 0.75):>9.4f}"
        )

    # Correlation needs pairwise-complete observations: keep rows where both
    # columns are present.
    pairs = [
        (float(r["flipper_length_mm"]), float(r["body_mass_g"]))
        for r in rows
        if r["flipper_length_mm"] not in ("", "NA") and r["body_mass_g"] not in ("", "NA")
    ]
    flipper = [p[0] for p in pairs]
    mass = [p[1] for p in pairs]
    print()
    print(f"Pearson r (flipper_length_mm vs body_mass_g, n={len(pairs)}): {pearson_r(flipper, mass):.4f}")

    print()
    print("mean body_mass_g by species:")
    for species in sorted({r["species"] for r in rows}):
        x = load_column([r for r in rows if r["species"] == species], "body_mass_g")
        print(f"  {species:<10} n={len(x):>3}  mean={mean(x):.4f}")


if __name__ == "__main__":
    main()
