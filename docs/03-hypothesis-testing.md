# Lesson 03 — Hypothesis Testing

Lesson 01 showed that Adelie and Chinstrap penguins have different *sample* means. But
samples are noisy — could a difference that size appear by chance even if the species were
identical? Hypothesis testing answers exactly that question. In this lesson you build
Welch's t-test and the chi-square test of independence from their formulas, get p-values
by brute-force permutation instead of distribution tables, and then check against
`scipy.stats`.

## Concept

**Null and alternative hypotheses.** Every test starts with a *null hypothesis* $H_0$ —
the boring explanation ("the two species have the same mean bill length", "species and
island are independent") — and an *alternative* $H_1$ (they differ / they are associated).
The test asks: is the data too surprising to be plausibly explained by $H_0$?

**What a p-value actually is.** The p-value is the probability of observing data *at
least as extreme* as what you got, **given that $H_0$ is true**. It is **not** the
probability that $H_0$ is true — that quantity is not even defined in this framework.
A p-value of 0.03 means: "in a world where nothing is going on, data like this shows up
3% of the time," not "there is a 97% chance of a real effect."

**Significance level.** Before testing, you pick a threshold $\alpha$ (conventionally
0.05) and reject $H_0$ when $p < \alpha$. By construction, $\alpha$ is your false-positive
rate: if $H_0$ is true, you will wrongly reject it in a fraction $\alpha$ of experiments.

**Quick glossary, before the formulas.** A few words that get used constantly below:

- **Test statistic** — one number computed from the data (here, $t$ or $\chi^2$) that
  summarizes how far the data sits from what $H_0$ predicts. Larger magnitude means more
  surprising.
- **Standard error** — the typical amount a statistic like a sample mean would wobble if
  you re-collected the sample; it is the "noise" you divide the "signal" (the observed
  difference) by.
- **Degrees of freedom (dof)** — loosely, how much independent information went into
  estimating a statistic's spread. It selects which reference curve (a t-distribution, a
  chi-square distribution) the statistic gets compared against; more data usually means
  more dof, and more dof pulls the reference curve closer to a normal distribution.
- **Permutation test** — instead of trusting a theoretical curve, you build the $H_0$
  world yourself by shuffling labels thousands of times and recomputing the statistic each
  time. The p-value is then just the fraction of shuffles that were at least as extreme as
  the real data.

**Welch's t-test.** To compare the means of two groups without assuming equal variances,
compute

$$t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\dfrac{s_A^2}{n_A} + \dfrac{s_B^2}{n_B}}}$$

Here $\bar{x}_A$ and $\bar{x}_B$ are the two sample means, $s_A^2$ and $s_B^2$ are the two
sample *variances* (the average squared distance of each value from its own group's mean,
divided by $n-1$ rather than $n$ — "Bessel's correction," which fixes the small downward
bias you get from estimating the mean and its spread from the same data), and $n_A$, $n_B$
are the two sample sizes. The denominator is the standard error of the difference in
means — think of $t$ as "signal (the gap between the means) divided by noise (how much
that gap would jitter across repeated samples)." Under $H_0$, $t$
follows (approximately) a t-distribution with Welch–Satterthwaite degrees of freedom

$$\nu = \frac{\left(\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}\right)^2}{\frac{(s_A^2/n_A)^2}{n_A - 1} + \frac{(s_B^2/n_B)^2}{n_B - 1}}$$

$\nu$ (the Greek letter *nu*) need not be a whole number — it is an *effective* degrees of
freedom that drops below the naive $n_A + n_B - 2$ whenever the two groups have unequal
variances or unequal sizes. A smaller $\nu$ gives the t-distribution heavier tails, which
makes the test more conservative (harder to declare significance from noise alone).

**Tiny hand-computable example.** Group A = [2, 4, 6], group B = [5, 7, 9]. Means:
$\bar{x}_A = 4$, $\bar{x}_B = 7$. Variances (squared deviations summed, divided by
$n - 1 = 2$): $s_A^2 = \frac{(2-4)^2+(4-4)^2+(6-4)^2}{2} = 4$, and $s_B^2 = 4$ too (B is
just A shifted by 3, so the spread is identical). Then
$t = \frac{4 - 7}{\sqrt{4/3 + 4/3}} = \frac{-3}{\sqrt{8/3}} \approx \frac{-3}{1.633}
\approx -1.837$. That is a toy, three-point-per-group example built only to make the
arithmetic checkable by hand; the penguin data below has hundreds of rows per group and,
because the real effect is large, produces a much bigger $|t|$.

**Why permutation tests work.** If $H_0$ is true, the group labels are arbitrary — a
bill length is just a bill length, whether the tag says Adelie or Chinstrap. So we can
*generate* the world of $H_0$: pool all values, shuffle, split into groups of the original
sizes, and record the difference in means. Repeat 10,000 times and you have the
distribution of the statistic under $H_0$, with no distributional assumptions at all. The
two-sided p-value is simply the fraction of shuffles whose $|\text{difference}|$ is at
least the observed one.

**What "permutation" means, concretely.** A permutation is one random reordering of the
pooled values, re-split into two groups of the original sizes $n_A$ and $n_B$. Using the
tiny example above, the pool is [2, 4, 6, 5, 7, 9]. One permutation might reorder it to
[6, 5, 2, 9, 4, 7]; the "fake A" is the first three entries (mean 4.333) and "fake B" is
the last three (mean 6.667), giving a permuted difference of about −2.333 — one sample of
what the difference in means looks like when the labels carry no real information at all.
Do this thousands of times and you trace out the whole distribution of the statistic under
$H_0$, without assuming it follows any particular curve.

**Chi-square test of independence.** For two categorical variables, compare the observed
contingency table $O$ against the counts expected under independence,
$E_{ij} = \frac{(\text{row}_i \text{ total})(\text{col}_j \text{ total})}{N}$:

$$\chi^2 = \sum \frac{(O-E)^2}{E}$$

Here $O_{ij}$ is the observed count in row $i$, column $j$ of the table (for example, how
many Adelie penguins live on Dream island); $E_{ij}$ is what that same cell would contain
if species and island were completely unrelated, built only from the row and column
totals; $r$ and $c$ are the number of rows and columns; and $N$ is the grand total count
across the whole table. The sum runs over every cell. Each term $(O-E)^2/E$ grows when a
cell's observed count strays far from its independence-predicted value, scaled by how big
that predicted value is — so $\chi^2$ is a total "surprise score" for the table: always
zero or positive, and exactly zero only when every observed count matches its expected
count precisely. Under $H_0$, $\chi^2$ approximately follows a chi-square distribution with
$(r-1)(c-1)$ degrees of freedom — once the row and column totals are held fixed, only
$(r-1)(c-1)$ of the cells are free to vary; every other cell is then determined by the
totals. The same permutation trick applies: shuffle one variable's labels and recompute
$\chi^2$.

**Tiny hand-computable example.** A 2x2 table with row totals 10 and 10, column totals 12
and 8 ($N = 20$), and observed counts $O = \begin{pmatrix}8 & 2\\4 & 6\end{pmatrix}$.
Independence predicts $E_{11} = 10 \times 12 / 20 = 6$, and the same formula gives
$E_{12} = 4$, $E_{21} = 6$, $E_{22} = 4$. Then
$\chi^2 = \frac{(8-6)^2}{6} + \frac{(2-4)^2}{4} + \frac{(4-6)^2}{6} + \frac{(6-4)^2}{4}
\approx 0.667 + 1 + 0.667 + 1 = 3.333$, with $(2-1)(2-1) = 1$ degree of freedom.

**Warning: multiple testing.** With $\alpha = 0.05$, testing 20 true null hypotheses
yields about one "significant" result by pure chance. If you test many hypotheses
(many columns, many group pairs), you must correct for it (e.g. Bonferroni: use
$\alpha / m$ for $m$ tests) — or you are just harvesting noise. The intuition: if each of
$m$ independent tests has a false-positive rate of $\alpha$ on its own, requiring
$\alpha / m$ per test instead keeps the *overall* chance of any false positive across all
$m$ tests near $\alpha$, rather than growing with $m$. One lesson, one planned
comparison, is the honest design.

## Dataset

[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) again — you already
downloaded it in Lesson 01. The download script verifies the checksum and skips the fetch,
so it is always safe to run:

```powershell
.\scripts\download_data.ps1 penguins
```

Expected output:

```text
[skip] penguins.csv (already downloaded, checksum OK)
Done.
```

This lesson uses `species`, `island`, and `bill_length_mm`. Two `bill_length_mm` values
are missing (one Adelie, one Gentoo) — that is why the t-test sees 151 Adelies out of the
152 in the table. Every row has `species` and `island`, so the chi-square test uses all 344.

## From scratch

Read [`src/lesson03_hypothesis_testing/from_scratch.py`](../src/lesson03_hypothesis_testing/from_scratch.py)
first — `welch_t` and `chi_square_stat` are the formulas from the Concept section, and
each permutation test is a ten-line loop. Only `csv`, `math`, and NumPy; no scipy, no
t-tables.

```bash
python src/lesson03_hypothesis_testing/from_scratch.py
```

Expected output:

```text
== Hypothesis testing: from scratch ==
rows in file: 344

-- Welch t-test: bill_length_mm, Adelie vs Chinstrap --
  Adelie     n=151  mean=38.7914  std=2.6634
  Chinstrap  n= 68  mean=48.8338  std=3.3393
difference in means (Adelie - Chinstrap): -10.0424
Welch t statistic: -21.8646
Welch-Satterthwaite dof: 106.9670
permutation test (10000 shuffles, seed 42):
  permutations with |diff| >= |observed|: 0 of 10000
  two-sided p-value: < 0.0001

-- Chi-square test of independence: species vs island --
observed counts:
                 Biscoe      Dream  Torgersen
  Adelie             44         56         52
  Chinstrap           0         68          0
  Gentoo            124          0          0
expected counts under independence:
                 Biscoe      Dream  Torgersen
  Adelie        74.2326    54.7907    22.9767
  Chinstrap     33.2093    24.5116    10.2791
  Gentoo        60.5581    44.6977    18.7442
chi-square statistic: 299.5503
dof: (3 - 1) * (3 - 1) = 4
permutation test (10000 shuffles of island labels, seed 42):
  permutations with chi2 >= observed: 0 of 10000
  p-value: < 0.0001
```

Things to notice:

- The observed difference is −10.04 mm against a standard error of ≈ 0.46 mm — the t
  statistic of −21.86 says the gap is about 22 standard errors wide. Unsurprisingly,
  **none** of the 10,000 shuffles came anywhere near it, so all we can report is
  $p < 0.0001$: a permutation p-value can never be smaller than $1/N_{\text{perm}}$.
- The observed contingency table has structural zeros — Chinstrap only on Dream, Gentoo
  only on Biscoe — while independence would predict ~33 Chinstraps on Biscoe. Hence the
  enormous $\chi^2 = 299.55$.
- The expected counts are not integers, and that is fine: they are averages under $H_0$,
  not predictions of an actual table.
- Both permutation tests report a plain integer count (`0 of 10000`) before converting it
  to a p-value string — the raw count is worth reading on its own: it is literally "how
  many of the 10,000 alternate, label-shuffled worlds looked at least this extreme."

### Step-by-step: reading the code

This walks through [`from_scratch.py`](../src/lesson03_hypothesis_testing/from_scratch.py)
in the order Python actually executes it: helper functions are *defined* top to bottom but
only *run* when `main()` calls them at the bottom of the file, so we will look at each
function, then follow `main()` as the driver that ties them together.

**Imports and constants.**

```python
import csv
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

SEED = 42
N_PERMUTATIONS = 10_000
```

Only the standard library (`csv`, `math`, `pathlib`) plus NumPy — no `scipy`. `Path(__file__)`
is this script's own file path; `.resolve()` makes it absolute, and `.parents[2]` walks up
two directory levels (out of `lesson03_hypothesis_testing/`, out of `src/`) to the repo
root, then `/ "data" / "penguins.csv"` appends the data file — this is why the script runs
correctly regardless of your current working directory or operating system (the `/`
operator here is `pathlib`'s overloaded path-joining, not division, so it produces the
right kind of path separator on both Windows and Linux). `SEED = 42` fixes the random
number generator's starting state so every run reshuffles the data in exactly the same
sequence — without it, the permutation counts (and hence the "Expected output" block)
would differ slightly on every run. `N_PERMUTATIONS = 10_000` is the number of shuffles
per test; the underscore in `10_000` is just a Python readability separator, identical to
`10000`.

**`mean` and `variance`: the building blocks.**

```python
def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def variance(x: list[float]) -> float:
    """Sample variance: divides by n-1 (Bessel's correction)."""
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)
```

`mean` is the arithmetic average. `variance` computes $s^2$ from the Concept section: it
subtracts the mean from every value, squares the result (`(v - m) ** 2`), sums those
squared deviations with a *generator expression* (the `for v in x` inside the
parentheses — it streams one value at a time into `sum()` instead of building an
intermediate list, which is a common memory-saving idiom), and divides by `len(x) - 1`,
not `len(x)` — that `-1` is Bessel's correction mentioned above.

**`welch_t`: the Concept formulas, verbatim.**

```python
def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t statistic and Welch-Satterthwaite degrees of freedom."""
    na, nb = len(a), len(b)
    va, vb = variance(a) / na, variance(b) / nb
    t = (mean(a) - mean(b)) / math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    return t, dof
```

`va` and `vb` are exactly $s_A^2/n_A$ and $s_B^2/n_B$ from the Concept section — the two
terms under the square root in the $t$ formula. `t` is that formula, typed directly:
difference of means over the square root of `va + vb`. `dof` is the Welch–Satterthwaite
$\nu$ formula, also typed directly. Returning `t, dof` as a tuple lets the caller unpack
both with one call: `t, dof = welch_t(a, b)`.

**`format_p`: why the output says `< 0.0001`, not `0.0000`.**

```python
def format_p(count: int, total: int) -> str:
    """Permutation p-value, or '< 1/total' when no permutation was as extreme."""
    if count == 0:
        return f"< {1 / total}"
    return f"{count / total:.4f}"
```

A permutation p-value is `count / total`. If `count` happens to be `0` (no shuffle was as
extreme as the real data), reporting `p = 0.0000` would falsely claim you *proved* the
event is impossible — you only ran 10,000 trials, so all you actually know is that the true
p-value is somewhere below the smallest value you could have detected, $1/10{,}000 =
0.0001$. Hence the special case.

**`chi_square_stat`: the Concept chi-square formula, vectorized.**

```python
def chi_square_stat(observed: np.ndarray) -> tuple[np.ndarray, float]:
    """Expected counts under independence and the chi-square statistic."""
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    expected = row_totals * col_totals / observed.sum()
    return expected, float(((observed - expected) ** 2 / expected).sum())
```

`observed` is the $r \times c$ table $O$ as a 2-D NumPy array. `axis=1` sums *across*
columns, collapsing each row to its total (`row_totals` has shape `(r, 1)`); `axis=0` sums
*across* rows, collapsing each column to its total (`col_totals` has shape `(1, c)`).
`keepdims=True` keeps those results 2-D instead of flattening them to plain 1-D lists —
that matters for the next line, `row_totals * col_totals`, which relies on NumPy
*broadcasting*: multiplying a `(r, 1)` array by a `(1, c)` array stretches each to
`(r, c)` and multiplies element-by-element, producing every $(\text{row}_i)(\text{col}_j)$
product in one call — precisely the numerator of the $E_{ij}$ formula from the Concept
section. Dividing by `observed.sum()` (the grand total $N$) finishes it. The last line is
the $\chi^2 = \sum (O-E)^2/E$ formula, computed over the whole array at once and reduced
to a single number with `.sum()`.

**`t_test`: filtering the data, then the permutation loop.**

```python
groups = {}
for species in ["Adelie", "Chinstrap"]:
    groups[species] = [
        float(r["bill_length_mm"])
        for r in rows
        if r["species"] == species and r["bill_length_mm"] not in ("", "NA")
    ]
a, b = groups["Adelie"], groups["Chinstrap"]
```

This builds one list of bill lengths per species, skipping rows where
`bill_length_mm` is blank or the literal string `"NA"` — this is exactly why the printed
`n=151` for Adelie is one short of the 152 Adelie rows in the file (see the Dataset
section above). Everything downstream (`mean`, `variance`, `welch_t`) works on these plain
Python lists.

```python
rng = np.random.default_rng(SEED)
pooled = np.array(a + b)
na = len(a)
count = 0
for _ in range(N_PERMUTATIONS):
    perm = rng.permutation(pooled)
    perm_diff = perm[:na].mean() - perm[na:].mean()
    if abs(perm_diff) >= abs(observed_diff):
        count += 1
```

`np.random.default_rng(SEED)` creates a seeded random *generator* object — this is the
modern NumPy idiom for reproducible randomness (it replaces the older, global
`np.random.seed(...)` style); because `rng` is a local object, two different `rng`s never
interfere with each other, which matters once both `t_test` and `chi_square_test` create
their own. `a + b` on plain Python lists is concatenation (not addition), so `pooled` holds
every bill length from both species with no group information attached — exactly the
"pool everything" step from the Concept section's permutation-test explanation.
`rng.permutation(pooled)` returns a new array with the same values in a random order (a
*shuffle*, not a resample-with-replacement — every original value appears exactly once).
Slicing the first `na` entries as "fake Adelie" and the rest as "fake Chinstrap"
(`perm[:na]`, `perm[na:]`) recreates two groups of the original sizes from the shuffled
pool. Comparing `abs(perm_diff) >= abs(observed_diff)` implements the two-sided p-value
definition directly: count every shuffle whose difference is at least as far from zero as
the real one, in either direction.

**`chi_square_test`: encoding categories as integers, then the same permutation idea.**

```python
species_labels = sorted({r["species"] for r in rows})
island_labels = sorted({r["island"] for r in rows})
s_index = {s: i for i, s in enumerate(species_labels)}
i_index = {s: i for i, s in enumerate(island_labels)}
s_codes = np.array([s_index[r["species"]] for r in rows])
i_codes = np.array([i_index[r["island"]] for r in rows])
n_s, n_i = len(species_labels), len(island_labels)
```

NumPy arrays hold numbers, not strings efficiently, so each species and island name is
first mapped to a small integer code. `{r["species"] for r in rows}` is a *set*
comprehension — it collects the distinct species names, discarding duplicates — and
`sorted(...)` puts them in alphabetical order (this is what fixes the row/column order you
see in "Expected output": Adelie, Chinstrap, Gentoo and Biscoe, Dream, Torgersen). The
dict comprehensions `{s: i for i, s in enumerate(...)}` then assign each label an integer
position (Adelie → 0, Chinstrap → 1, Gentoo → 2, and similarly for islands). `s_codes` and
`i_codes` are then the whole dataset's species and island, each recoded as an integer
array of the same length.

```python
def table(i_codes: np.ndarray) -> np.ndarray:
    return np.bincount(s_codes * n_i + i_codes, minlength=n_s * n_i).reshape(n_s, n_i)
```

This builds the contingency table without any explicit nested loop over species and
island. `s_codes * n_i + i_codes` combines each row's two small codes into one unique
integer (a standard trick: if `n_i` islands exist, then `species_code * n_i + island_code`
never collides between two different `(species, island)` pairs). `np.bincount` is a fast
histogram: it counts how many times each integer `0, 1, 2, ...` appears in its input;
`minlength=n_s * n_i` pads the result so every possible combined code has a slot even if
that combination never occurs (this is exactly why the Chinstrap/Biscoe cell can show `0`
rather than being missing). `.reshape(n_s, n_i)` folds the flat list of counts back into
an `(n_s, n_i)` matrix — the observed table $O$. Notice `table` takes `i_codes` as a
parameter rather than reading the outer variable directly: that is deliberate, so the
permutation loop below can call `table(shuffled_i_codes)` to build a fresh fake table on
every iteration while `s_codes` (species) never changes.

```python
rng = np.random.default_rng(SEED)
count = 0
for _ in range(N_PERMUTATIONS):
    _, perm_chi2 = chi_square_stat(table(rng.permutation(i_codes)))
    if perm_chi2 >= chi2:
        count += 1
```

Same idea as the t-test's permutation loop, adapted to this test: shuffle only the island
codes (species stays fixed, so this asks "if island were reassigned at random, independent
of species, how big would $\chi^2$ typically get?"), rebuild the table, and recompute
$\chi^2$. The comparison is `perm_chi2 >= chi2` with **no** `abs(...)`, unlike the t-test's
`abs(perm_diff) >= abs(observed_diff)` — $\chi^2$ is already a sum of squares, so it can
never be negative, and "more extreme" only ever means "larger." This mirrors the fact that
a chi-square test has no two-sided/one-sided distinction the way a difference-in-means
test does.

**`main`: the driver.**

```python
def main() -> None:
    with open(DATA, newline="") as f:
        rows = list(csv.DictReader(f))

    print("== Hypothesis testing: from scratch ==")
    print(f"rows in file: {len(rows)}")
    print()
    t_test(rows)
    print()
    chi_square_test(rows)
```

`csv.DictReader` reads the header line of `penguins.csv` automatically and turns every
subsequent row into a dictionary keyed by column name (`r["species"]`, `r["island"]`,
`r["bill_length_mm"]`, and so on) — this is why every helper function above indexes rows
by column name rather than by position. `list(...)` materializes the reader's output into
an ordinary list so it can be scanned twice: once inside `t_test` (filtering to Adelie and
Chinstrap) and once inside `chi_square_test` (using every row, since species/island have
no missing values). The two calls, `t_test(rows)` then `chi_square_test(rows)`, are why the
Welch section of "Expected output" always appears before the chi-square section.

## With a library

The same two tests with analytic p-values from `scipy.stats`. Read
[`with_library.py`](../src/lesson03_hypothesis_testing/with_library.py) — the entire
Welch test is one call, `stats.ttest_ind(a, b, equal_var=False)`, and the chi-square test
is `stats.chi2_contingency(table)`.

```bash
python src/lesson03_hypothesis_testing/with_library.py
```

Expected output:

```text
== Hypothesis testing: with scipy ==
rows in file: 344

-- Welch t-test: bill_length_mm, Adelie vs Chinstrap --
  Adelie     n=151  mean=38.7914  std=2.6634
  Chinstrap  n= 68  mean=48.8338  std=3.3393
Welch t statistic: -21.8646
Welch-Satterthwaite dof: 106.9670
two-sided p-value: 2.9001e-41

-- Chi-square test of independence: species vs island --
observed counts:
                 Biscoe      Dream  Torgersen
  Adelie             44         56         52
  Chinstrap           0         68          0
  Gentoo            124          0          0
chi-square statistic: 299.5503
dof: 4
p-value: 1.3546e-63
```

### Step-by-step: reading the code

[`with_library.py`](../src/lesson03_hypothesis_testing/with_library.py) is much shorter
than `from_scratch.py` because `pandas` and `scipy.stats` absorb every formula from the
Concept section into single function calls.

```python
import pandas as pd
from scipy import stats
```

`pandas` loads and slices the tabular data; `scipy.stats` supplies the hypothesis tests
themselves. There is no hand-written `mean`, `variance`, `welch_t`, or `chi_square_stat`
anywhere in this file.

```python
df = pd.read_csv(DATA)
```

One call replaces `from_scratch.py`'s `open(...)` plus `csv.DictReader` plus `list(...)`:
`pd.read_csv` reads the whole file into a `DataFrame` (a table object with named,
type-inferred columns), so `df["bill_length_mm"]` is already numeric — no `float(...)`
conversion needed.

```python
bill = df.dropna(subset=["bill_length_mm"])
groups = {}
for species in ["Adelie", "Chinstrap"]:
    x = bill.loc[bill["species"] == species, "bill_length_mm"]
    groups[species] = x
    print(f"  {species:<10} n={len(x):>3}  mean={x.mean():.4f}  std={x.std():.4f}")
```

`df.dropna(subset=["bill_length_mm"])` drops every row with a missing `bill_length_mm`
value in one call — the pandas equivalent of `from_scratch.py`'s
`if r["bill_length_mm"] not in ("", "NA")` filter, and the reason `n=151` matches between
the two scripts. `bill.loc[bill["species"] == species, "bill_length_mm"]` is pandas'
boolean-mask indexing: `bill["species"] == species` produces a column of `True`/`False`,
and `.loc[mask, column]` keeps only the matching rows of that one column. **Gotcha**:
`x.std()` uses pandas' default `ddof=1` (divide by $n-1$), the same Bessel's correction as
`from_scratch.py`'s `variance` — that default is why the printed `mean`/`std` lines match
the from-scratch output exactly; NumPy's own `.std()` defaults to `ddof=0` instead, and
would print a slightly different number here.

```python
res = stats.ttest_ind(groups["Adelie"], groups["Chinstrap"], equal_var=False)
print(f"Welch t statistic: {res.statistic:.4f}")
print(f"Welch-Satterthwaite dof: {res.df:.4f}")
print(f"two-sided p-value: {res.pvalue:.4e}")
```

`stats.ttest_ind` is scipy's general two-independent-samples t-test — it implements both
the classic Student's t-test and Welch's t-test from the Concept section, selected by one
argument. **Gotcha**: `equal_var` defaults to `True`, which runs *Student's* t-test (it
assumes both groups share one population variance) — silently dropping
`equal_var=False` from this call would switch to a different statistical test with a
different formula and a different dof, not just a different-looking function call. Passing
`equal_var=False` requests exactly the Welch statistic and Welch–Satterthwaite dof this
lesson builds by hand. The result object's `.statistic`, `.df`, and `.pvalue` correspond to
$t$, $\nu$, and the p-value. **Gotcha**: `ttest_ind` also defaults to
`alternative="two-sided"`, matching the from-scratch permutation test's `abs(...) >=
abs(...)` two-sided comparison; passing `alternative="less"` or `"greater"` would instead
test a one-sided hypothesis (e.g., "Adelie bills are shorter than Chinstrap's") and roughly
halve the reported p-value for the same data.

```python
table = pd.crosstab(df["species"], df["island"]).sort_index(axis=0).sort_index(axis=1)
chi2 = stats.chi2_contingency(table)
```

`pd.crosstab(df["species"], df["island"])` builds the observed contingency table in one
call — the pandas equivalent of `from_scratch.py`'s `s_codes * n_i + i_codes` plus
`np.bincount` trick. The two `.sort_index(...)` calls alphabetize the rows and columns so
they line up with the from-scratch script's `sorted(...)` label order.
`stats.chi2_contingency(table)` computes the chi-square statistic, its dof, its p-value,
and the expected-counts table all at once, returned together on `chi2`. **Gotcha**: by
default `chi2_contingency` applies Yates' continuity correction, but only when the table is
2x2 (`dof == 1`); this table is 3x3, so the correction has no effect here — on a 2x2 table,
the default would shrink $\chi^2$ slightly relative to the uncorrected formula from the
Concept section, and you would need `correction=False` to match it exactly. Note also that
this script never prints `chi2.expected_freq` — unlike `from_scratch.py`, it does not
display the expected-counts table, even though scipy computed one internally.

```python
print(f"chi-square statistic: {chi2.statistic:.4f}")
print(f"dof: {chi2.dof}")
print(f"p-value: {chi2.pvalue:.4e}")
```

`chi2.dof` is printed as a plain integer, computed internally by scipy as $(r-1)(c-1)$ —
compare this to `from_scratch.py`, which prints the formula itself,
`"(3 - 1) * (3 - 1) = 4"`, so you can see the arithmetic rather than trust the library.

The statistics match the from-scratch run exactly — same $t$, same dof, same $\chi^2$ —
because those are deterministic formulas. The **p-values** agree in verdict but not in
value, and they never will exactly:

- The **analytic** p-value assumes the statistic follows a known curve under $H_0$ (the
  t-distribution with 106.97 dof; the chi-square with 4 dof) and integrates its tail. That
  curve is itself an approximation — exact only under normality for the t-test, and
  asymptotic for the chi-square — but it can resolve absurdly small tails like $10^{-41}$.
- The **permutation** p-value assumes only that labels are exchangeable under $H_0$, and
  estimates the tail by Monte Carlo. With 10,000 shuffles its resolution floor is
  $1/10{,}000$, and even for moderate p-values it carries sampling noise of order
  $\sqrt{p(1-p)/N_{\text{perm}}}$.

Here the truth is so far into the tail that the permutation test saturates at
$p < 0.0001$ while scipy reports $10^{-41}$ — same conclusion, different resolution. On a
borderline dataset ($p \approx 0.05$) you would instead see the two values differ in the
third decimal or so.

## Check your understanding

1. A colleague reads $p = 0.03$ and says "there is a 97% probability the species really
   differ." Rewrite their sentence so it states what the p-value actually measures.
2. The permutation test reported `0 of 10000`. Why is it wrong to report $p = 0$? What is
   the smallest p-value 10,000 permutations can resolve?
3. Welch's test computes $\nu \approx 106.97$ from $n_A = 151$ and $n_B = 68$. What
   property of the two groups pushes $\nu$ below $n_A + n_B - 2 = 217$?
4. In the chi-square permutation test we shuffled island labels but kept the row and
   column totals fixed. Why must the expected counts stay the same across permutations?
5. You test all three species pairs and all four numeric columns — 18 t-tests at
   $\alpha = 0.05$. About how many false positives do you expect if no species differ at
   all, and what threshold would Bonferroni suggest?
6. `chi_square_test`'s permutation loop compares `perm_chi2 >= chi2` with no `abs(...)`,
   while `t_test`'s loop compares `abs(perm_diff) >= abs(observed_diff)`. Why does the
   chi-square comparison not need an absolute value?
7. If `with_library.py` called `stats.ttest_ind(a, b)` without `equal_var=False`, which
   test would scipy actually run, what assumption would it silently add, and would you
   expect the reported dof to still be 106.97?

From here, you're now ready to proceed to try and reproduce
[Lesson 04 — Linear Regression](04-linear-regression.md).

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
