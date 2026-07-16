# Lesson 10 — PCA

Eleven correlated chemistry measurements per wine is more than you can plot and more than
some models can comfortably digest. Principal Component Analysis finds the few directions
along which the data actually varies and lets you project onto them. In this lesson you
build PCA from its linear-algebra core — a covariance matrix and one call to `eigh` — and
then make scikit-learn reproduce every number to 4 decimals.

## Concept

**Motivation: the curse of dimensionality.** In high dimensions, data gets lonely —
points spread so thin that distances lose contrast and models need exponentially more
samples. But measured features are rarely independent (fixed acidity, citric acid, and pH
all echo the same underlying acidity), so the data often lives near a much
lower-dimensional subspace. PCA finds that subspace.

If that summary feels abstract, hold onto one picture for the rest of this lesson: think
of the 11-dimensional wine data as a cloud of points. It is not a uniform blob — it is
stretched out like a cigar, long in some directions and thin in others. PCA's entire job
is to find the long axes of that cloud, ranked from longest to shortest, and hand you a
new coordinate system lined up with them.

### Geometric intuition: rotating to face the spread

Picture a simple 2-D scatter plot of two correlated measurements — say height and weight.
Plotted on the raw x/y axes, the cloud of points looks like a tilted ellipse: most of the
"action" happens along a diagonal (taller people tend to weigh more), not purely
horizontally or vertically. **PCA finds that diagonal direction of maximum spread and
rotates the data to align with it.** The new first axis (PC1) points along the long axis
of the ellipse; the new second axis (PC2) is forced to be perpendicular to it (in 2-D
there is no other choice) and picks up whatever scatter is left over — usually much less.

In $d$ dimensions the same recipe repeats: find the single direction of maximum spread
(PC1); then, restricted to directions perpendicular to PC1, find the next direction of
maximum spread (PC2); then the next, perpendicular to both, and so on. Because every new
axis is forced to be perpendicular ("orthogonal") to all the earlier ones, no two
components ever describe the same spread twice.

**Variance maximization.** PCA asks: which *unit vector* $w$ — a vector of length exactly
1, so it encodes a pure direction with no scale of its own — maximizes the variance of the
projected data $Xw$? "Projecting" a data point onto $w$ means taking its dot product with
$w$: geometrically, dropping a perpendicular from the point onto the line through $w$ and
reading off the signed distance along that line from the center. Do this for every row of
$X$ and $Xw$ is the resulting column of numbers, one per sample; its variance measures how
spread out those projected points are. That first *principal component* is the line the
cloud is most spread along; the second maximizes variance among directions orthogonal to
the first (at a right angle, sharing no direction with it), and so on.

**The covariance matrix is all you need.** For centered data $X$ ($n \times d$ — $n$ rows,
one per sample, and $d$ columns, one per feature, with every column already shifted so its
mean is zero), the sample covariance matrix is

$$C = \frac{1}{n-1} X^\top X, \qquad \mathrm{Var}(Xw) = w^\top C w$$

$X^\top$ is $X$ transposed (rows and columns swapped), so $X^\top X$ comes out $d \times
d$ — one row and one column per feature, not per sample. Entry $(i, j)$ of $C$ is the
covariance between feature $i$ and feature $j$: positive if the two tend to rise and fall
together, negative if one rises as the other falls, near zero if they barely relate. The
diagonal entries $C_{ii}$ are the ordinary variances of each feature. $w^\top C w$ (read
"$w$-transpose times $C$ times $w$") is a compact way of writing the variance of the
projected data $Xw$ purely in terms of the direction $w$ and the covariance matrix $C$,
without ever needing the raw data $X$ again — that is the sense in which "the covariance
matrix is all you need."

Maximizing $w^\top C w$ subject to $\lVert w \rVert = 1$ (the length of $w$ must stay
exactly 1 — otherwise you could inflate the "variance" simply by making $w$ longer) is a
classic eigenproblem: the solutions are the **eigenvectors** of $C$ (the principal
directions), and each **eigenvalue** is exactly the variance captured along its
eigenvector. An eigenvector $v$ of $C$ is a special direction that $C$ does not rotate —
it only stretches or shrinks it, $Cv = \lambda v$, and the stretch factor $\lambda$ is its
eigenvalue. Because $C$ is symmetric ($C_{ij} = C_{ji}$, true of every covariance matrix),
linear algebra guarantees $d$ real eigenvalues and $d$ eigenvectors that are all mutually
orthogonal — together they form exactly the perpendicular coordinate system PCA is
looking for. `np.linalg.eigh` is the NumPy routine specialized for computing eigenvalues
and eigenvectors of such symmetric (or, in the complex case, Hermitian — the "h" in
`eigh`) matrices stably.

### A tiny worked example

Suppose you had five 2-D points laid out roughly along a diagonal:

```text
x = [1, 2, 3, 4, 5]
y = [1, 2.2, 2.8, 4.3, 4.9]
```

Sketch them and you would see a cloud stretched from bottom-left to top-right, hugging the
line $y = x$. Running PCA on just these two columns would return a PC1 close to the
direction $(1/\sqrt2,\ 1/\sqrt2)$ — the 45-degree diagonal — because that is the direction
these five points are most spread along. PC2 would be the perpendicular direction,
$(1/\sqrt2,\ -1/\sqrt2)$ (or its sign-flip), capturing the small amount of scatter off the
diagonal, such as point 4 sitting a touch above the line. Projecting the points onto PC1
alone would preserve almost all of the spread using one number per point instead of two —
the same compression PCA performs on the 11 wine features, just easy to sketch here
because $d = 2$. The wine dataset works exactly the same way with 11 axes instead of 2 —
you cannot sketch an 11-dimensional ellipse, but its eigenvectors are still nothing more
exotic than "the long axes of the cloud."

**Explained variance ratio.** $\lambda_j / \sum_k \lambda_k$ — the $j$-th eigenvalue
divided by the sum of all of them — is the fraction of total variance on component $j$;
the cumulative sum tells you how many components you need ("2 PCs keep 46% of the
variance, 7 keep 91%").

**Standardize first.** PCA chases variance, and variance has units. Raw total sulfur
dioxide (std ≈ 33 mg/L) would out-vary citric acid (std ≈ 0.19 g/L) a thousandfold, and
PC1 would just be "sulfur dioxide, rescaled". z-scoring every feature first — subtracting
each column's mean, then dividing by its standard deviation, so every column ends up with
mean 0 and variance 1 — gives each feature equal footing, so components reflect
*correlation structure* rather than measurement units.

**Sign ambiguity.** If $Cv = \lambda v$ then $C(-v) = \lambda(-v)$ (flipping every sign in
$v$ leaves the equation true, since $\lambda$ multiplies straight through the flip): an
eigenvector's sign is arbitrary, and different libraries legitimately disagree. For
reproducible output both scripts apply the same convention — flip each eigenvector so its
largest-absolute-value component is positive.

**PCA is linear.** It can only rotate and truncate axes — it can never bend or fold the
coordinate system. Structure that lives on a curve — a spiral, a swiss roll — is invisible
to it; you'd reach for kernel PCA, t-SNE, or UMAP. And maximal variance is not the same
thing as maximal *relevance* to any particular prediction task (Check your understanding
question 5, below, asks you to construct a case where the two flatly disagree).

## Dataset

[UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (red) again
(License: CC BY 4.0) — you already downloaded it in Lesson 04, so the script just verifies
the checksum:

```powershell
.\scripts\download_data.ps1 wine
```

Expected output:

```text
[skip] winequality-red.csv (already downloaded, checksum OK)
Done.
```

Remember from Lesson 04: the file is **semicolon**-separated. We use the 11
physico-chemical features (quality excluded) and standardize all of them.

## From scratch

Read [`from_scratch.py`](../src/lesson10_pca/from_scratch.py) — the math is four lines:
standardize, `np.cov`, `np.linalg.eigh`, sort eigenvalues descending. `fix_signs`
implements the sign convention, and projection is a single matrix product `X @ V[:, :2]`.

### Step-by-step: reading the code

Walk through the file in the order Python actually executes it — imports and constants
first, then each function definition, then the `main` driver at the bottom.

```python
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "winequality-red.csv"

SEED = 42  # no randomness here, kept for convention
```

`csv` reads the semicolon-separated data file; `Path(__file__).resolve().parents[2]`
climbs two directories up from this script (`src/lesson10_pca/` → `src/` → the repo root)
and then into `data/`, so the script finds the CSV no matter what directory you run it
from. `SEED` is never used — PCA on a fixed dataset is entirely deterministic, there is no
random split or initialization to seed — but it is kept anyway so every lesson script in
the repo has the same visible shape.

```python
def load_data() -> tuple[np.ndarray, list[str]]:
    """Return (X standardized, feature_names). The file is SEMICOLON-separated;
    the last column (quality) is excluded. z-scoring uses ddof=0, the same
    convention as sklearn's StandardScaler."""
    with open(DATA, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        names = next(reader)[:-1]
        X = np.array([[float(v) for v in row[:-1]] for row in reader if row])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, names
```

`csv.reader(f, delimiter=";")` tells the `csv` module to split on semicolons rather than
the default comma. `next(reader)` consumes the header row once; `[:-1]` drops its last
entry, `"quality"`, because this lesson only analyzes the 11 physico-chemical features.
The list comprehension does the same trick per row — `row[:-1]` skips each row's quality
value — and builds a plain list of lists of `float`s, skipping any blank trailing row
(`if row`); wrapping that in `np.array(...)` turns it into one `(1599, 11)` numeric
matrix. The last line is the standardization step the Concept section calls "standardize
first": `X.mean(axis=0)` and `X.std(axis=0)` compute one mean and one standard deviation
per *column* (`axis=0` means "collapse down the rows, keep one result per column"), so
subtracting and dividing z-scores every feature independently. This also produces the
*centered* data ($X$ with mean 0 per column) that the covariance formula $C = \frac{1}{n-1}
X^\top X$ in the Concept section assumes.

```python
def fix_signs(V: np.ndarray) -> np.ndarray:
    """Eigenvectors are defined only up to sign: flip each column so that its
    largest-absolute-value component is positive (first index wins ties)."""
    V = V.copy()
    for j in range(V.shape[1]):
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] = -V[:, j]
    return V
```

`V` is a matrix whose *columns* are eigenvectors — one column per principal component.
`np.argmax(np.abs(V[:, j]))` finds the row index of the largest-magnitude entry in column
`j`; if the raw value there happens to be negative, the whole column is negated with
`V[:, j] = -V[:, j]`. This is exactly the sign convention described in the Concept section
("flip each eigenvector so its largest-absolute-value component is positive"), needed
because $Cv = \lambda v$ and $C(-v) = \lambda(-v)$ are equally valid — nothing about the
math picks a sign for you. `V.copy()` at the top matters: without it, the flips would
mutate the caller's array in place, silently changing `eigvecs` too.

```python
def main() -> None:
    X, names = load_data()

    print("== PCA (covariance eigendecomposition): from scratch ==")
    print(f"rows: {len(X)}, features: {len(names)} (quality excluded), standardized")
    print()
```

`main` loads the standardized data and prints a small header so the two scripts' outputs
are easy to tell apart while diffing (`from scratch` here vs. `with scikit-learn` in
`with_library.py`).

```python
    C = np.cov(X, rowvar=False)  # divides by n-1
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending order
    order = np.argsort(-eigvals, kind="stable")  # descending, first wins ties
    eigvals = eigvals[order]
    V = fix_signs(eigvecs[:, order])
```

`np.cov(X, rowvar=False)` builds the $d \times d$ covariance matrix $C$ from the Concept
formula directly; `rowvar=False` tells NumPy that *columns* are the variables/features
and rows are observations — the opposite of NumPy's own default assumption, so it is easy
to get backwards. `np.linalg.eigh(C)` is the stable symmetric eigensolver from the Concept
section: it returns `eigvals` in ascending order and `eigvecs`, whose column `j` is the
eigenvector paired with `eigvals[j]`. Since PCA wants the *largest*-variance direction
first, `np.argsort(-eigvals, kind="stable")` sorts descending — negating before an
ascending sort reverses the order — and `kind="stable"` guarantees that exact ties keep
their original relative order, which matters for reproducibility. `eigvals[order]` and
`eigvecs[:, order]` then reindex the eigenvalues and the *columns* of the eigenvector
matrix into that descending order, before `fix_signs` normalizes their signs.

```python
    ratio = eigvals / eigvals.sum()
    print(f"  {'PC':>4}  {'eigenvalue':>10}  {'explained':>9}  {'cumulative':>10}")
    for j in range(len(eigvals)):
        print(
            f"  {f'PC{j + 1}':>4}  {eigvals[j]:>10.4f}  {ratio[j]:>9.4f}  "
            f"{ratio[: j + 1].sum():>10.4f}"
        )
    print()
```

`ratio = eigvals / eigvals.sum()` is $\lambda_j / \sum_k \lambda_k$ from the Concept
section, computed for every component at once. The loop prints one table row per
component; `ratio[: j + 1].sum()` sums the ratios from PC1 through PC`j+1`, giving the
running "cumulative" column. The `:>4`, `:>10.4f`, and similar format specs just right-align
each column to a fixed width so the table lines up — they don't affect the numbers, only
how they're displayed.

```python
    for j in range(2):
        top = sorted(range(len(names)), key=lambda i: (-abs(V[i, j]), i))[:3]
        loads = ", ".join(f"{names[i]} {V[i, j]:+.4f}" for i in top)
        print(f"PC{j + 1} top-3 loadings by |value|: {loads}")
    print()
```

A "loading" is one entry of an eigenvector — it says how much a single original feature
contributes to that principal component. For PC1 and PC2 (`range(2)`), this sorts the
feature indices by descending absolute loading (`-abs(V[i, j])` negates so an ascending
sort behaves like a descending one; `i` breaks ties by original column order) and keeps
the top 3. `loads` then formats each as `name +value`, joined into the single readable
line you see in the Expected output below.

```python
    Z = X @ V[:, :2]
    print("projection of the first 3 samples onto (PC1, PC2):")
    for i in range(3):
        print(f"  sample {i + 1}: ({Z[i, 0]:+.4f}, {Z[i, 1]:+.4f})")
```

`V[:, :2]` slices out just the first two columns of `V` — the PC1 and PC2 directions. `@`
is NumPy's matrix-multiplication operator, so `X @ V[:, :2]` projects *every* standardized
sample onto both directions in one shot, producing `Z`, an `(n, 2)` matrix. This is
literally $Xw$ from the Concept section's variance-maximization formula, evaluated for
two different $w$'s at once; `Z[i, 0]` and `Z[i, 1]` are sample `i`'s coordinates in the
new 2-D principal-component space, which is what the last block of Expected output prints.

```python
if __name__ == "__main__":
    main()
```

The standard Python entry-point guard: `main()` only runs when the file is executed
directly (`python src/lesson10_pca/from_scratch.py`), not when it's imported elsewhere —
for instance, by a test that wants to call `load_data()` without also printing a report.

```bash
python src/lesson10_pca/from_scratch.py
```

Expected output:

```text
== PCA (covariance eigendecomposition): from scratch ==
rows: 1599, features: 11 (quality excluded), standardized

    PC  eigenvalue  explained  cumulative
   PC1      3.1011     0.2817      0.2817
   PC2      1.9271     0.1751      0.4568
   PC3      1.5515     0.1410      0.5978
   PC4      1.2140     0.1103      0.7081
   PC5      0.9599     0.0872      0.7953
   PC6      0.6600     0.0600      0.8552
   PC7      0.5842     0.0531      0.9083
   PC8      0.4232     0.0385      0.9468
   PC9      0.3449     0.0313      0.9781
  PC10      0.1814     0.0165      0.9946
  PC11      0.0596     0.0054      1.0000

PC1 top-3 loadings by |value|: fixed acidity +0.4893, citric acid +0.4636, pH -0.4385
PC2 top-3 loadings by |value|: total sulfur dioxide +0.5695, free sulfur dioxide +0.5136, alcohol -0.3862

projection of the first 3 samples onto (PC1, PC2):
  sample 1: (-1.6195, +0.4510)
  sample 2: (-0.7992, +1.8566)
  sample 3: (-0.7485, +0.8820)
```

Things to notice:

- The loadings are readable chemistry: PC1 is an *acidity axis* (fixed acidity and citric
  acid load positive, pH negative — more acid, lower pH, moving together), and PC2 is a
  *sulfur dioxide axis* (both sulfur dioxide measurements load positive, alcohol negative).
  PCA rediscovered these groupings purely from correlations in the numbers — it was never
  told which features were "chemically related."
- The spectrum decays gently: 2 PCs keep only 46% of the variance, and 95% needs 9 of 11
  components. Not every dataset hides a tiny subspace — PCA also tells you when it doesn't,
  and here the 11 wine measurements are only mildly redundant, not massively so.
- A ddof footnote (echoing Lesson 01): the eigenvalues sum to $11 \cdot \frac{n}{n-1}
  \approx 11.0069$, not exactly 11, because standardization divides by $n$ (like
  `StandardScaler`) while `np.cov` divides by $n-1$. The explained variance *ratios* are
  unaffected, because the same $\frac{n}{n-1}$ factor multiplies every eigenvalue equally
  and cancels out of the ratio $\lambda_j / \sum_k \lambda_k$ (see Check your understanding
  question 4).

## With a library

Read [`with_library.py`](../src/lesson10_pca/with_library.py) — `StandardScaler` +
`PCA(n_components=None)`, then the *same* sign convention applied to
`pca.components_` before printing, because sklearn's own sign choice (inherited from its
SVD routine) differs from raw `eigh` output.

### Step-by-step: reading the code

`with_library.py` mirrors `from_scratch.py` almost line for line; only the middle of
`main` changes, replacing `np.cov` + `np.linalg.eigh` with scikit-learn's `PCA`.

```python
    X = StandardScaler().fit_transform(X)
```

`StandardScaler().fit_transform(X)` is the library equivalent of the manual
`(X - X.mean(axis=0)) / X.std(axis=0)` in `from_scratch.py`: it computes each column's
mean and standard deviation (also with `ddof=0`, matching the from-scratch script) and
returns the z-scored matrix in one call — the same "standardize first" step from the
Concept section.

```python
    pca = PCA(n_components=None)
    pca.fit(X)
    eigvals = pca.explained_variance_  # eigenvalues of the covariance matrix
    V = fix_signs(pca.components_.T)  # columns = principal directions
```

`PCA(n_components=None)` builds an estimator that keeps *all* $d$ components rather than
reducing dimensionality (that argument is where you'd normally pass, say, `2`, to keep
only the top two). `pca.fit(X)` fits it — internally, scikit-learn computes this via the
singular value decomposition (SVD) of the centered data rather than an explicit
eigendecomposition of the covariance matrix, but the two routes agree mathematically: the
squared singular values, divided by $n-1$, are exactly the eigenvalues from the Concept
section, and the right singular vectors are the same eigenvectors `eigh` would return
(the existing note under Expected output below expands on this). `pca.explained_variance_`
exposes those eigenvalues directly — the equivalent of `eigvals` in `from_scratch.py`.
`pca.components_` holds the eigenvectors as *rows* (one row per component), the opposite
layout from `from_scratch.py`'s `eigvecs`, which holds them as *columns* — hence the
`.T` transpose, so that `fix_signs`, written to flip columns, still flips the right thing.

```python
    ratio = pca.explained_variance_ratio_
```

`pca.explained_variance_ratio_` is scikit-learn's precomputed version of
`eigvals / eigvals.sum()` — the same $\lambda_j / \sum_k \lambda_k$ ratio, so this line
replaces a whole computation from `from_scratch.py` with an attribute lookup.

The two loops that print the top-3 loadings and the ratio/cumulative table are copied
verbatim from `from_scratch.py` — proof that once `X`, `eigvals`, and `V` are in hand, the
reporting code doesn't care which library produced them.

```python
    Z = X @ V[:, :2]  # X is already centered, so this is pca.transform + sign fix
```

You might expect `pca.transform(X)` here instead of a manual matrix product. The comment
explains why it's written this way: `pca.transform` would apply scikit-learn's own
(unflipped) component signs, whereas this script needs the *sign-fixed* `V` for its output
to match `from_scratch.py` to 4 decimals. Since `X` is already centered, projecting by
hand with `X @ V[:, :2]` — the same $Xw$ from the Concept section, and the same line used
in `from_scratch.py` — gives an identical result with the right signs.

```bash
python src/lesson10_pca/with_library.py
```

Expected output: identical to the from-scratch run except the first line, which reads
`== PCA: with scikit-learn ==`. Every eigenvalue, ratio, loading, and projected coordinate
matches to 4 decimals — `pca.explained_variance_` *is* the eigenvalue spectrum of the
covariance matrix, computed via SVD instead of `eigh`. If any number differs, something is
wrong — open an issue.

## Check your understanding

1. Why does maximizing $w^\top C w$ over unit vectors $w$ yield the eigenvector of $C$
   with the largest eigenvalue? (Hint: write $w$ in the eigenvector basis.)
2. What would PC1 look like if we skipped standardization? Which feature would dominate
   and why? (Peek at the stds in Lesson 04 if you need them.)
3. Your colleague runs PCA on the same data and gets your PC1 with every sign flipped. Is
   their code wrong? What downstream quantities are affected, and which are not?
4. The covariance matrix of standardized (ddof=0) data has $n/(n-1)$ on its diagonal
   instead of 1. Why does this factor cancel in the explained variance ratio?
5. Give an example of a 2-D dataset where the direction of maximum variance is the *worst*
   possible axis for separating two classes.
6. `fix_signs` breaks ties in `np.argmax` by keeping the first index (`(-abs(V[i, j]), i)`
   in the loading-sort, and simple first-match in `fix_signs` itself). Construct a
   (contrived) eigenvector where a tie could occur, and explain why the convention still
   produces reproducible output even though "first index wins" is an arbitrary rule.
7. `with_library.py` transposes `pca.components_` (`pca.components_.T`) before calling
   `fix_signs`, but `from_scratch.py` never transposes `eigvecs`. Look at how
   `np.linalg.eigh` and `sklearn.decomposition.PCA` each lay out eigenvectors/components
   in memory — rows or columns — and explain why the transpose is necessary in one script
   and not the other.

From here, you're now ready to proceed to try and reproduce
[Lesson 11 — Neural Networks from Scratch](11-neural-networks.md).

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
