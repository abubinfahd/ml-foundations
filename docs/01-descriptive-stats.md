# Lesson 01 — Descriptive Statistics

Before any modeling, you must be able to summarize data: where is its center, how spread
out is it, and how do variables move together? In this lesson you compute the core
descriptive statistics with plain Python — every formula visible as code — and then check
your numbers against pandas.

## Concept

**Central tendency.**

Some notation used throughout this lesson, spelled out before you hit the formulas:

- $n$ — the number of values in the sample (here, the non-missing entries in one column of
  the CSV).
- $x_i$ — the $i$-th individual value, where $i$ ranges from $1$ to $n$. Read it as "the
  value in row $i$."
- $\sum_{i=1}^{n}$ — "sum over all rows from 1 to $n$": add up whatever expression follows
  it, once per value.
- $\bar{x}$ (read "x-bar") — the sample mean of the $x$ values.

The *mean* of $n$ values is

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

In plain language: add up every value, then divide by how many values there are. It is the
single number that best balances the data, in the sense that the positive and negative
deviations from it cancel out exactly — a fact that makes it useful, and also makes it
sensitive to outliers (see below).

*Hand-computable example.* Let $x = [2, 4, 6, 8, 10]$, so $n = 5$.

$$\bar{x} = \frac{2+4+6+8+10}{5} = \frac{30}{5} = 6$$

Keep this same five-number list handy — the variance and quantile examples below reuse it.

The *median* is the middle value after sorting (or the average of the two middle values
when $n$ is even). For $x = [2,4,6,8,10]$ (already sorted, $n=5$, odd) the median is the
middle element, $6$ — it matches the mean here only because the data happens to be
symmetric. The median is robust to outliers; the mean is not — one extreme penguin drags
the mean, but not the median. (Try it yourself: replace the $10$ with $1000$ and recompute
both by hand — the median doesn't move, the mean jumps a lot.)

**Spread.**

- $s^2$ — the *sample variance*: a single number summarizing how far, on average, values
  sit from the mean, expressed in *squared* units.
- $(x_i - \bar{x})$ — the *deviation* of value $i$ from the mean: how far above (positive)
  or below (negative) the mean that one value sits.

The *sample variance* is

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

Why squared deviations rather than plain deviations? Plain deviations from the mean always
sum to exactly zero (the positive ones and negative ones cancel), so averaging them would
always give zero — useless as a measure of spread. Squaring makes every term positive (so
nothing cancels) and penalizes large deviations more than small ones, matching the
intuition that one wildly-off value should count for more than several mildly-off ones.

Note the $n-1$ (Bessel's correction): dividing by $n$ would systematically underestimate
the population variance, because the deviations are measured from the sample's own mean.
Concretely: once you know $n-1$ of the deviations, the last one is forced, since the
deviations always sum to zero by construction — so there are really only $n-1$
*independent* pieces of information ("degrees of freedom") in that sum. Dividing by that
smaller number, rather than by $n$, corrects the bias. With $n=1$, the formula divides by
zero: a single value carries no information about spread at all, which is exactly why
variance is undefined for one point.

The *standard deviation* $s = \sqrt{s^2}$ is in the same units as the data. (Variance is in
squared units — for a column measured in millimeters, variance is in square millimeters,
which isn't directly comparable to the original values; the square root undoes that.)

*Hand-computable example, continued.* For $x = [2,4,6,8,10]$, $\bar{x}=6$, so the
deviations are $-4,-2,0,2,4$. Squared: $16,4,0,4,16$, which sum to $40$. Then

$$s^2 = \frac{40}{5-1} = 10 \qquad s = \sqrt{10} \approx 3.1623$$

Once you've read the code below, check this against `variance([2,4,6,8,10])` and
`std([2,4,6,8,10])`.

**Quantiles.** The $q$-quantile (for $q$ between 0 and 1) is the value below which a
fraction $q$ of the data falls — the same idea as a percentile in a school test-score
report, just expressed as a fraction instead of a percentage (the 25th percentile is the
$q=0.25$ quantile). When $q(n-1)$ is not an integer, we interpolate linearly between the
two neighboring sorted values — this is the default method in NumPy and pandas, and the
one implemented here.

*Hand-computable example.* Using the sorted list $[2,4,6,8,10]$ (indices $0$ through $4$),
take $q=0.6$: $h = (n-1) \times q = 4 \times 0.6 = 2.4$. The integer part, $2$, is the lower
index; the fractional part, $0.4$, says how far to slide toward the next value:

$$\text{value} = x_{[2]} + 0.4 \times (x_{[3]} - x_{[2]}) = 6 + 0.4 \times (8-6) = 6.8$$

**Correlation.** The *Pearson correlation coefficient* measures linear association:

$$r = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_i (x_i-\bar{x})^2}\sqrt{\sum_i (y_i-\bar{y})^2}} \in [-1, 1]$$

Here $x_i$ and $y_i$ are two *paired* measurements from the same row (e.g. one penguin's
flipper length and its body mass), and $\bar x$, $\bar y$ are their respective means. The
numerator, $\sum_i (x_i-\bar x)(y_i-\bar y)$, is the (unnormalized) *covariance*: it is
positive when $x$ and $y$ tend to sit above their means together and below their means
together, negative when one tends to be above while the other is below, and close to zero
when there's no consistent pattern. The denominator — the product of the two square-root
terms — rescales that number so it always lands between $-1$ and $1$, regardless of the
units $x$ and $y$ are measured in. That's also why converting body mass from grams to
kilograms does not change $r$: rescaling $x$ scales both a numerator term and the matching
denominator term by the same factor, so the factor cancels in the ratio.

$r = 1$ is a perfect increasing line, $r = -1$ a perfect decreasing line, $r = 0$ no
*linear* relationship (there may still be a nonlinear one!).

**Missing values.** Real data has holes. Here we drop missing values per column, and for
correlation we keep only rows where *both* variables are present ("pairwise complete").
The `n` printed for each statistic tells you how many values were actually used.

## Dataset

[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) — 344 penguins from
three species measured at Palmer Station, Antarctica. License: CC0 (public domain).

```powershell
.\scripts\download_data.ps1 penguins
```

Expected output:

```text
[get ] penguins.csv
[ ok ] penguins.csv (checksum verified)
Done.
```

Take a look at the data before computing anything (always do this):

```bash
head -3 data/penguins.csv
```

```text
species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year
Adelie,Torgersen,39.1,18.7,181,3750,male,2007
Adelie,Torgersen,39.5,17.4,186,3800,female,2007
```

## From scratch

Read [`src/lesson01_descriptive_stats/from_scratch.py`](../src/lesson01_descriptive_stats/from_scratch.py)
first — it is ~100 lines, using only `csv` and `math` from the standard library. Each
function is one formula from the Concept section.

### Step-by-step: reading the code

The file has four parts, in this order: setup (imports and constants), small helper
functions that each implement one formula, a `pearson_r` function, and a `main()` function
that ties everything together and prints the report. Reading them in that order helps,
because each function only uses what came before it.

**Setup**

```python
import csv
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
```

`csv` is the standard-library module for reading comma-separated files; `math` supplies
`sqrt` and `floor`. `Path(__file__)` means "the path to this very script"; `.resolve()`
makes it absolute, and `.parents[2]` walks up two directory levels (from
`src/lesson01_descriptive_stats/` to the repo root) so the script finds `data/penguins.csv`
regardless of the folder you happen to run it from. `NUMERIC_COLS` is just the four column
names the report will summarize, defined once at the top so they're used consistently below
instead of being retyped — and possibly mistyped — in several places.

**`load_column`**

```python
def load_column(rows: list[dict], col: str) -> list[float]:
    """Return the non-missing values of a column as floats."""
    return [float(r[col]) for r in rows if r[col] not in ("", "NA")]
```

The `list[dict]` and `-> list[float]` annotations are *type hints*: documentation for
humans (and optional static-analysis tools) saying this function takes a list of
dictionaries and returns a list of floats — Python does not enforce them at runtime.

The return line is a *list comprehension*, a compact way to write a loop that builds a new
list: `[expression for item in iterable if condition]` reads as "for each `r` in `rows`, if
`r[col]` is not an empty string and not the literal text `NA`, include `float(r[col])` in
the result." CSV files represent a missing value either as nothing between the commas
(`""`) or as the text `NA` (as seen in the penguins file), so both are checked. This is
exactly the "drop missing values per column" behavior described in Concept.

**`mean`**

```python
def mean(x: list[float]) -> float:
    return sum(x) / len(x)
```

A direct translation of $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$: `sum(x)` computes
$\sum_{i=1}^n x_i$, and `len(x)` gives $n$.

**`median`**

```python
def median(x: list[float]) -> float:
    s = sorted(x)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
```

`sorted(x)` returns a new, sorted copy of the list (it does not modify `x` itself). `//` is
*integer division* (divides and rounds down to a whole number) and `%` is the *remainder*
operator, so `n % 2 == 1` tests "is `n` odd?" For odd `n`, the middle index `mid` (found
with integer division) points straight at the median. For even `n` there is no single
middle element, so the function averages the two values on either side of the midpoint —
matching the "average of the two middle values" rule from Concept.

**`variance` and `std`**

```python
def variance(x: list[float]) -> float:
    """Sample variance: divides by n-1 (Bessel's correction)."""
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def std(x: list[float]) -> float:
    return math.sqrt(variance(x))
```

`sum((v - m) ** 2 for v in x)` is a *generator expression* — like a list comprehension, but
it produces values one at a time instead of building an intermediate list, which `sum()`
then totals. It computes exactly $\sum_i (x_i - \bar x)^2$: for every value `v`, subtract
the mean `m`, square the result, and add them all up. Dividing by `len(x) - 1` is the
$n-1$ from the variance formula. `std` just wraps `variance` in `math.sqrt`, mirroring
$s = \sqrt{s^2}$.

**`quantile`**

```python
def quantile(x: list[float], q: float) -> float:
    """Quantile with linear interpolation (the NumPy/pandas default)."""
    s = sorted(x)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    frac = h - lo
    if lo + 1 < len(s):
        return s[lo] + frac * (s[lo + 1] - s[lo])
    return s[lo]
```

This is the interpolation worked by hand in the Concept section. `h = (len(s) - 1) * q` is
the (possibly fractional) index the quantile falls at. `math.floor(h)` rounds down to get
the lower neighboring index `lo`; `frac` is what's left over — the fraction of the way from
`s[lo]` to the next value. The `if lo + 1 < len(s)` guard handles the edge case $q=1$ (the
maximum): there, `lo` is already the last index and `frac` is `0`, so there is no
`s[lo + 1]` to look at, and the function returns `s[lo]` directly instead of indexing past
the end of the list.

**`pearson_r`**

```python
def pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation: covariance scaled by both standard deviations."""
    mx, my = mean(x), mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy)
```

`zip(x, y)` pairs up the two lists element-by-element — `(x[0], y[0])`, `(x[1], y[1])`, and
so on — which is exactly what "paired measurements" means in the Concept section's
correlation formula. `cov` is the numerator (unnormalized covariance); `sx` and `sy` are
the two square-root terms in the denominator. Notice that `cov`, `sx`, and `sy` never
divide by $n$ or $n-1$ anywhere — that's fine, because in the ratio
$\frac{\text{cov}}{s_x \cdot s_y}$ any such scaling factor would appear in both the top and
bottom and cancel out. Skipping it is simply less arithmetic for the same answer.

**`main`**

```python
def main() -> None:
    with open(DATA, newline="") as f:
        rows = list(csv.DictReader(f))
```

`with open(...) as f` opens the file and guarantees it gets closed afterward, even if an
error occurs — the standard, safe way to handle files in Python. `csv.DictReader` reads the
file's header row automatically and yields one `dict` per data row, keyed by column name
(so `r["species"]` fetches the species column of row `r`) — this is exactly what
`load_column` above expects. Wrapping it in `list(...)` reads the whole file into memory at
once as a list of dicts, so it can be looped over multiple times below.

```python
    print(f"{'column':<20} {'n':>4} {'mean':>10} {'median':>10} {'std':>9} {'q25':>9} {'q75':>9}")
    for col in NUMERIC_COLS:
        x = load_column(rows, col)
        print(
            f"{col:<20} {len(x):>4} {mean(x):>10.4f} {median(x):>10.4f} "
            f"{std(x):>9.4f} {quantile(x, 0.25):>9.4f} {quantile(x, 0.75):>9.4f}"
        )
```

This prints the table header, then loops over each of the four numeric columns, loads its
non-missing values, and prints one row of statistics. The `f"...{expr:<20}..."` syntax is
an *f-string* with a *format spec*: `<20` left-aligns text in a 20-character-wide field,
`>10` right-aligns in a 10-character field, and `.4f` formats a float with exactly 4 digits
after the decimal point. These are purely cosmetic — they make the columns line up in the
terminal — and have no effect on the underlying numbers.

```python
    pairs = [
        (float(r["flipper_length_mm"]), float(r["body_mass_g"]))
        for r in rows
        if r["flipper_length_mm"] not in ("", "NA") and r["body_mass_g"] not in ("", "NA")
    ]
    flipper = [p[0] for p in pairs]
    mass = [p[1] for p in pairs]
```

This builds the "pairwise complete" pairs mentioned in Concept: a list comprehension keeps
only rows where *both* flipper length and body mass are present, producing a list of
`(flipper, mass)` tuples. The next two lines split that list of pairs back into two
separate lists — `p[0]` and `p[1]` index into each tuple — because `pearson_r` expects two
separate equal-length lists, not a list of pairs.

```python
    print()
    print("mean body_mass_g by species:")
    for species in sorted({r["species"] for r in rows}):
        x = load_column([r for r in rows if r["species"] == species], "body_mass_g")
        print(f"  {species:<10} n={len(x):>3}  mean={mean(x):.4f}")
```

`{r["species"] for r in rows}` is a *set comprehension* — like a list comprehension, but a
`set` automatically drops duplicates, so this collects the distinct species names.
`sorted(...)` turns that set into a list in alphabetical order, so the report always prints
in the same order (plain sets have no guaranteed order). For each species, the inner list
comprehension `[r for r in rows if r["species"] == species]` filters `rows` down to just
that species' rows, and `load_column` then extracts that group's body-mass values, ready to
be averaged with `mean`.

**The driver block**

```python
if __name__ == "__main__":
    main()
```

`__name__` is a special variable Python sets to `"__main__"` when a file is *run directly*
(as opposed to being `import`ed by another file). This guard means `main()` only executes
when you run `python src/lesson01_descriptive_stats/from_scratch.py` directly — if some
other script imported this file to reuse `mean` or `pearson_r`, that import would not
automatically re-run the whole report. You'll see this idiom in almost every runnable
Python script in this course.

Now run it:

```bash
python src/lesson01_descriptive_stats/from_scratch.py
```

Expected output:

```text
== Descriptive statistics: from scratch ==
rows in file: 344

column                  n       mean     median       std       q25       q75
bill_length_mm        342    43.9219    44.4500    5.4596   39.2250   48.5000
bill_depth_mm         342    17.1512    17.3000    1.9748   15.6000   18.7000
flipper_length_mm     342   200.9152   197.0000   14.0617  190.0000  213.0000
body_mass_g           342  4201.7544  4050.0000  801.9545 3550.0000 4750.0000

Pearson r (flipper_length_mm vs body_mass_g, n=342): 0.8712

mean body_mass_g by species:
  Adelie     n=151  mean=3700.6623
  Chinstrap  n= 68  mean=3733.0882
  Gentoo     n=123  mean=5076.0163
```

Things to notice:

- `n=342`, not 344 — two penguins are missing all measurements. That matches the `""`/`NA`
  filtering in `load_column`: those two rows never contribute a float to any list.
- Flipper length and body mass are strongly correlated ($r = 0.87$): bigger penguins have
  bigger flippers.
- The species means differ hugely (Gentoo ≈ 5 kg vs ≈ 3.7 kg) — a *grouped* summary can
  reveal structure that a whole-column summary hides. This is exactly why the per-species
  loop exists in the code, rather than stopping at one overall mean.
- The `mean` and `median` for `body_mass_g` (4201.75 vs 4050.00) are close but not equal —
  a preview of the skewness question below.

## With a library

The same numbers, computed by pandas. Read
[`with_library.py`](../src/lesson01_descriptive_stats/with_library.py) — note how much
shorter it is, and map each pandas call back to the formula it implements.

### Step-by-step: reading the code

```python
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
```

Same path-finding trick as `from_scratch.py`, but only one extra import is needed:
`pandas` (conventionally aliased `pd`) bundles CSV reading, missing-value handling, and
every statistic below into one library.

```python
    df = pd.read_csv(DATA)
```

`pd.read_csv` reads the whole file into a `DataFrame` — pandas's table type, roughly a
collection of columns rather than `from_scratch.py`'s list of row-dicts. It also
automatically recognizes common missing-value markers, including the empty string and
`NA`, and stores them as `NaN` ("Not a Number") — the same two cases `load_column` checked
for explicitly.

```python
    for col in NUMERIC_COLS:
        x = df[col].dropna()
        print(
            f"{col:<20} {len(x):>4} {x.mean():>10.4f} {x.median():>10.4f} "
            f"{x.std():>9.4f} {x.quantile(0.25):>9.4f} {x.quantile(0.75):>9.4f}"
        )
```

`df[col]` selects one column as a `Series`; `.dropna()` removes the `NaN` entries — the
library equivalent of `load_column`'s filter. `.mean()`, `.median()`, `.std()`, and
`.quantile(q)` are the same formulas from Concept, precomputed for you. Two defaults are
worth knowing explicitly, because they're easy to get wrong: `Series.std()` divides by
$n-1$ by default (`ddof=1`), matching this lesson's sample variance — but `numpy.std` on a
plain array divides by $n$ instead unless you pass `ddof=1` yourself, which is a common
source of mismatched numbers when mixing the two libraries. `Series.quantile()` also
defaults to the same linear interpolation method implemented by hand in `quantile()`.

```python
    pair = df[["flipper_length_mm", "body_mass_g"]].dropna()
    r = pair["flipper_length_mm"].corr(pair["body_mass_g"])
```

`df[[...]]` (note the double brackets) selects *two* columns at once as a small
DataFrame; `.dropna()` on that DataFrame drops any row missing *either* column, which is
exactly the pairwise-complete rule from Concept. `.corr()` defaults to Pearson correlation
— the same formula as `pearson_r` — though it also accepts `method="spearman"` or
`method="kendall"` for rank-based alternatives, worth knowing exist even though this
lesson only uses the default.

```python
    grouped = df.dropna(subset=["body_mass_g"]).groupby("species")["body_mass_g"]
    for species, x in grouped:
        print(f"  {species:<10} n={len(x):>3}  mean={x.mean():.4f}")
```

`dropna(subset=["body_mass_g"])` drops only rows missing body mass (ignoring the other
columns), then `.groupby("species")` splits the remaining rows into one group per species
— the library equivalent of the from-scratch version's set comprehension plus manual
filtering. Iterating over a grouped object yields `(group_key, group_data)` pairs, so
`for species, x in grouped` gives the species name together with that species' subset of
`body_mass_g` values, ready for `.mean()`.

The `if __name__ == "__main__": main()` guard at the bottom of this file is the same
idiom explained in the from-scratch walkthrough above.

Now run it:

```bash
python src/lesson01_descriptive_stats/with_library.py
```

Expected output: identical to the from-scratch run except the first line, which reads
`== Descriptive statistics: with pandas ==`. If any number differs, something is wrong —
open an issue.

(Why do they match to 4 decimals? Both use sample variance with $n-1$ and linear
quantile interpolation. Defaults matter: `numpy.var` divides by $n$ unless you pass
`ddof=1`, while `pandas.Series.var` divides by $n-1$. Knowing your library's defaults is
part of knowing the statistic.)

## Check your understanding

1. Why does the variance formula divide by $n-1$ and not $n$? What would happen with $n=1$?
2. The mean body mass (4201.75 g) is larger than the median (4050 g). What does that tell
   you about the shape of the distribution?
3. If you converted body mass from grams to kilograms, what would happen to the Pearson
   correlation with flipper length? Why?
4. `numpy.std` and `pandas.Series.std` give different answers on the same data. Explain.
5. Give an example of two variables with $r \approx 0$ that are strongly (nonlinearly)
   related.
6. Using the five-number list $[2,4,6,8,10]$ from Concept, work out `quantile(x, 0.1)` by
   hand: what is `h`, what is `lo`, what is `frac`, and what is the final interpolated
   value? Check your answer by tracing through the `quantile` function line by line.
7. `pearson_r` never divides `cov`, `sx`, or `sy` by $n$ or $n-1$, yet it still returns a
   correctly normalized correlation between $-1$ and $1$. Why is that division
   unnecessary? (Hint: what would happen to the extra factor if it were added to both the
   numerator and the denominator?)

From here, you're now ready to proceed to try and reproduce
[Lesson 02 — Probability & Distributions](02-probability-distributions.md).

Before you move on, however, add an entry in the "Reproduction Log" at the bottom of this
page, following the same format: use `yyyy-mm-dd`, make sure you're using a commit id
that's on the master branch of `ml-foundations`, and use its 7-hexadecimal prefix for the
link anchor text. In the description of your pull request, please provide some details on
your setup (e.g., operating system, environment and configuration, etc.). In addition,
also provide some indication of success (e.g., everything worked) or document issues you
encountered. If you think this guide can be improved in any way (e.g., you caught a typo
or think a clarification is warranted), feel free to include it in the pull request.

## Reproduction Log

<!-- Append one line per reproduction below this comment. Keep chronological order. -->
