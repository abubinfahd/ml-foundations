"""Lesson 01 — Descriptive statistics, with pandas.

Computes exactly the same numbers as from_scratch.py, using pandas.
The two outputs must match.

Run from the repo root:

    python src/lesson01_descriptive_stats/with_library.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def main() -> None:
    df = pd.read_csv(DATA)

    print("== Descriptive statistics: with pandas ==")
    print(f"rows in file: {len(df)}")
    print()
    print(f"{'column':<20} {'n':>4} {'mean':>10} {'median':>10} {'std':>9} {'q25':>9} {'q75':>9}")
    for col in NUMERIC_COLS:
        x = df[col].dropna()
        print(
            f"{col:<20} {len(x):>4} {x.mean():>10.4f} {x.median():>10.4f} "
            f"{x.std():>9.4f} {x.quantile(0.25):>9.4f} {x.quantile(0.75):>9.4f}"
        )

    pair = df[["flipper_length_mm", "body_mass_g"]].dropna()
    r = pair["flipper_length_mm"].corr(pair["body_mass_g"])
    print()
    print(f"Pearson r (flipper_length_mm vs body_mass_g, n={len(pair)}): {r:.4f}")

    print()
    print("mean body_mass_g by species:")
    grouped = df.dropna(subset=["body_mass_g"]).groupby("species")["body_mass_g"]
    for species, x in grouped:
        print(f"  {species:<10} n={len(x):>3}  mean={x.mean():.4f}")


if __name__ == "__main__":
    main()
