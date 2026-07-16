# Lesson 04 — Linear Regression

Your first predictive model. Linear regression fits a weighted sum of features to a
numeric target — and it has a rare luxury: an exact closed-form solution. In this lesson
you derive that solution (the normal equation), then reach the same answer iteratively
with gradient descent — the workhorse that trains nearly every model in later lessons —
and finally verify both against scikit-learn and statsmodels.

## Concept

**Notation first — every symbol below is something you can point to in the code.**

- $n$ — the number of rows (wines) in a split. In this lesson $n = 1279$ for train,
  $320$ for test.
- $d$ — the number of features. Here $d = 11$ (fixed acidity, volatile acidity, ...,
  alcohol).
- $X$ — the **design matrix**: one row per wine, one column per feature, plus a column
  of all 1s glued on the left so the model gets an intercept "for free." Shape
  $n \times (d+1)$. In the code this is the array called `A_train`/`A_test`.
- $y$ — the vector of true targets (the `quality` column), length $n$.
- $\theta$ (theta) — the vector of parameters you are solving for: index 0 is the
  intercept, indices 1..d are the per-feature coefficients. Length $d+1$. In the code
  this is `theta_ne` (from the normal equation) and `theta_gd` (from gradient descent).
- $\hat{y}$ ("y hat") — the model's predictions: one number per row, always computed as
  $X\theta$.

**The model.** With $n$ rows and $d$ features, stack the data into a matrix $X$ (with a
leading column of ones for the intercept) and predict

$$\hat{y} = X\theta$$

where $\theta \in \mathbb{R}^{d+1}$ holds the intercept and one coefficient per feature.
Written out per row, $\hat y_i = \theta_0 + \theta_1 x_{i1} + \theta_2 x_{i2} + \dots +
\theta_d x_{id}$: each prediction is a constant offset plus a weighted sum of that row's
features. "Linear" refers to this — a straight-line combination of the inputs, not a
curve.

**A hand-computable example.** Before the wine data, fit a line through four points you
can check with a calculator: $(x, y) \in \{(1,1), (2,2), (3,2), (4,3)\}$ — $y$ roughly
grows with $x$. With one feature, $X$ has a column of 1s (for the intercept) and a
column of $x$:

$$X=\begin{bmatrix}1&1\\1&2\\1&3\\1&4\end{bmatrix},\qquad y=\begin{bmatrix}1\\2\\2\\3\end{bmatrix}$$

The normal equation (derived below) needs only two small matrices built from $X$ and
$y$:

$$X^TX=\begin{bmatrix}4&10\\10&30\end{bmatrix},\qquad X^Ty=\begin{bmatrix}8\\23\end{bmatrix}$$

Every entry is a sum you can do by hand: 4 rows; $\sum x_i = 1+2+3+4=10$;
$\sum x_i^2 = 1+4+9+16=30$; $\sum y_i = 1+2+2+3=8$; $\sum x_iy_i = 1+4+6+12=23$. Solving
$X^TX\,\theta = X^Ty$ — that is, $4\theta_0+10\theta_1=8$ and
$10\theta_0+30\theta_1=23$ — gives $\theta_0=0.5$, $\theta_1=0.6$: intercept $0.5$,
slope $0.6$. Sanity-check one point: at $x=2$, $\hat y = 0.5+0.6\times2=1.7$ against an
actual $y=2$ (residual $0.3$); add up the residual at all four points and they cancel to
exactly zero — a property of least squares whenever an intercept is included. Nothing in
`from_scratch.py` is conceptually different from this — it is the same two-matrix
computation with 12 columns (11 features plus intercept) instead of 2, and 1279 rows
instead of 4.

**The objective.** Least squares picks the $\theta$ minimizing the mean squared error

$$J(\theta) = \frac{1}{n}\lVert X\theta - y\rVert^2 = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

$\lVert v \rVert^2$ denotes the squared length of a vector: square every entry and add
them up. So $\lVert X\theta - y\rVert^2$ is shorthand for "take every residual
$\hat y_i - y_i$, square it, and sum" — dividing by $n$ then turns that sum into an
average, the *mean* squared error. Squaring (rather than, say, taking absolute values)
is what makes $J$ smooth and bowl-shaped, and it penalizes large errors more than small
ones.

**Normal equation.** $J$ is a convex quadratic bowl (picture a single valley with one
lowest point, never a wavy surface with multiple dips), so set its gradient to zero:
$\nabla J = \frac{2}{n}X^T(X\theta - y) = 0$ gives $X^TX\,\theta = X^Ty$, hence

$$\theta^* = (X^TX)^{-1}X^Ty$$

$\nabla J$ ("gradient of $J$") is the vector of partial derivatives of $J$ with respect
to every entry of $\theta$ — it points toward the direction in which $J$ increases
fastest, so the one place it equals zero, on a convex bowl, is the single lowest point.
$X^T$ is the transpose of $X$ (its rows and columns swapped), and $(X^TX)^{-1}$ is a
*matrix inverse* — the matrix analogue of dividing by a number, the thing that undoes
multiplying by $X^TX$. Read the whole formula as "the one point where the bowl bottoms
out, reached in a single algebraic step" — this is exactly the two-line computation done
by hand above, just at $12\times12$ scale instead of $2\times2$.

In code we never form the inverse — we call `solve(X^T X, X^T y)`, which is faster and
numerically safer. `np.linalg.solve(M, b)` finds the $\theta$ satisfying $M\theta = b$
directly, skipping the extra arithmetic (and extra rounding error) of computing
$(X^TX)^{-1}$ and then multiplying it by $X^Ty$.

**Gradient descent.** Instead of solving exactly, start anywhere and repeatedly step
downhill:

$$\theta \leftarrow \theta - \alpha \nabla J(\theta) = \theta - \alpha\,\frac{2}{n}X^T(X\theta - y)$$

with learning rate $\alpha$ (alpha) — a small positive number you choose, $0.1$ in the
script — controlling how big a step to take each time. Read the right-hand side
right-to-left: $X\theta - y$ is the current residual vector (how wrong each prediction
is right now); multiplying by $X^T$ folds those $n$ residuals back down into a
$(d+1)$-length vector of "how much to blame each parameter"; scaling by $2/n$ matches
the derivative of $J$; and $\theta \leftarrow \theta - \alpha(\cdot)$ nudges every
parameter a small distance in the direction that reduces the loss. Too large and it
diverges; too small and it crawls. Because $J$ is convex, gradient descent converges to
the *same* $\theta^*$ as the normal equation — the script proves it by printing the max
absolute difference between the two.

**Why standardize?** Features on wildly different scales (total sulfur dioxide ≈ 46,
density ≈ 0.997) make the loss surface a stretched valley: the step size that is safe for
the steep direction is glacial for the shallow one. Rescaling every feature to mean 0 and
standard deviation 1 rounds out the bowl, so one learning rate works for all coordinates.

**Data leakage.** The mean and std used for standardization are computed on the *training
rows only*, then applied to the test rows. Computing them on all rows would let facts
about the test set influence training — a subtle form of cheating called *leakage*. Test
error must estimate performance on data the whole pipeline has never touched.

**Metrics.** $\text{MSE} = \frac{1}{n}\sum_i(\hat{y}_i - y_i)^2$ — the same quantity as
$J(\theta)$ above, just given its usual evaluation-time name — and

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

where $\bar y$ ("y bar") is the mean of $y$: the numerator is your model's total squared
error, the denominator is the total squared error of the naive baseline that always
predicts the mean. $R^2$ compares the model against the dumbest baseline — always
predicting the mean. $R^2 = 1$ is perfect, $0$ is no better than the mean, and on a
*test* set it can even go negative. Train error is always the flattering number; test
error is the honest one.

## Dataset

[UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) (red wine) —
1599 Portuguese "Vinho Verde" red wines with 11 physicochemical measurements (acidity,
sugar, alcohol, ...) and a `quality` score (0–10) from blind taste tests. License:
CC BY 4.0. Please cite: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis,
*Modeling wine preferences by data mining from physicochemical properties*, Decision
Support Systems 47(4):547–553, 2009.

```powershell
.\scripts\download_data.ps1 wine
```

Expected output:

```text
[get ] winequality-red.csv
[ ok ] winequality-red.csv (checksum verified)
Done.
```

Take a look at the data before modeling anything (always do this):

```bash
head -2 data/winequality-red.csv
```

```text
"fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
```

Note the *semicolons*: this file is `;`-separated, not comma-separated. A default
`read_csv` call would give you one useless column — always check the delimiter.

## From scratch

Read [`from_scratch.py`](../src/lesson04_linear_regression/from_scratch.py) first. NumPy
does the linear algebra, but every modeling step is explicit: the shuffled 80/20 split,
train-only standardization, `solve(X^T X, X^T y)`, and the gradient descent loop.

### Step-by-step: reading the code

Follow the file top to bottom — this is the order Python actually executes it in.

**1. Constants set the experiment up front.**

```python
DATA = Path(__file__).resolve().parents[2] / "data" / "winequality-red.csv"
SEED = 42
LR = 0.1  # gradient descent learning rate
ITERS = 2000  # gradient descent iterations
CHECKPOINTS = (0, 100, 500, 2000)
```

`SEED` fixes the random shuffle so the split — and every downstream number — is
reproducible run to run. `LR` and `ITERS` are the learning rate $\alpha$ and iteration
count from Concept above. `CHECKPOINTS` only controls which iterations print a progress
line; it has no effect on the math.

**2. `load_data` turns the CSV into plain NumPy arrays.**

```python
def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the semicolon-separated wine CSV: 11 features, target = quality."""
    with open(DATA) as f:
        header = [h.strip().strip('"') for h in f.readline().strip().split(";")]
    raw = np.genfromtxt(DATA, delimiter=";", skip_header=1)
    return raw[:, :-1], raw[:, -1], header[:-1]
```

The header line is read separately (and its surrounding quote characters stripped) only
to get human-readable feature names for later printouts. `np.genfromtxt` loads the
numeric rows in one call; `skip_header=1` tells it to skip the header line rather than
try to parse it as numbers. `raw[:, :-1]` means "every row, every column except the
last" — the 11 features, i.e. $X$ — and `raw[:, -1]` means "every row, only the last
column" — `quality`, i.e. $y$. This `[:, k]` slicing (rows, then columns) is the
standard way to index a 2-D NumPy array.

**3. `mse` and `r2` implement the two formulas from Concept directly.**

```python
def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error: average squared residual."""
    return float(np.mean((y - y_hat) ** 2))


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot: fraction of variance explained,
    relative to the always-predict-the-mean baseline."""
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot
```

`y - y_hat` subtracts two equal-length arrays element by element — no loop needed,
NumPy does it for every row at once. `** 2` squares every entry, and `np.mean`/`np.sum`
collapse the whole array down to a single number. That is precisely
$\frac{1}{n}\sum_i(\hat y_i-y_i)^2$ and $\sum_i(\hat y_i - y_i)^2$, computed without
writing a `for` loop over rows yourself.

**4. `print_metrics` is shared by both the normal equation and gradient descent, so
both blocks are scored identically.**

```python
def print_metrics(A_train, y_train, A_test, y_test, theta) -> None:
    yh_train, yh_test = A_train @ theta, A_test @ theta
    print(f"train MSE: {mse(y_train, yh_train):.4f}  train R^2: {r2(y_train, yh_train):.4f}")
    print(f"test  MSE: {mse(y_test, yh_test):.4f}  test  R^2: {r2(y_test, yh_test):.4f}")
```

`A_train @ theta` is matrix-vector multiplication — `@` is NumPy's operator for
`np.matmul`: for every row of `A_train`, multiply it element-wise against `theta` and
sum, producing one number (one prediction) per row. This is exactly $X\theta$ from the
model equation. `yh` is this file's shorthand for $\hat y$ throughout.

**5. `main` loads the data, then builds the shuffled 80/20 split.**

```python
    # Deterministic 80/20 split: shuffle all row indices once, first 80% train.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
```

`rng.permutation(n)` returns a random reordering of the integers $0, 1, \dots, n-1$ — a
shuffled list of row indices, not the data itself. Taking the first 80% of that shuffled
list as `train_idx` and the remainder as `test_idx`, then writing `X[train_idx]`
(NumPy's "fancy indexing": pull out exactly these rows, in this order), is equivalent to
shuffling the deck and cutting it, without ever physically reordering the original
arrays.

**6. Standardization uses train statistics only — the code-level version of "Data
leakage" in Concept.**

```python
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
```

`axis=0` means "collapse down the rows, once per column," so `mu` and `sigma` are each
length-11 vectors — one mean and one standard deviation per feature. `X_train - mu`
subtracts a length-11 vector from each of the 1279 rows of `X_train`: this is
*broadcasting*, NumPy's rule for stretching a smaller array across a bigger one without
writing an explicit loop or copying data. Notice `X_test` is transformed using `mu` and
`sigma` computed from the *training* rows — they are never recomputed on test data.

**7. A column of ones is glued onto both matrices so the model gets an intercept.**

```python
    A_train = np.column_stack([np.ones(len(X_train)), X_train])
    A_test = np.column_stack([np.ones(len(X_test)), X_test])
```

`np.column_stack` places its arguments side by side as columns of one matrix. `A_train`
and `A_test` are the design matrix $X$ from Concept — column 0 is all 1s, exactly like
the hand-worked example above, just with 11 feature columns after it instead of 1.

**8. The normal equation, solved directly.**

```python
    # --- Normal equation: solve (X^T X) theta = X^T y exactly. ---
    theta_ne = np.linalg.solve(A_train.T @ A_train, A_train.T @ y_train)
```

`A_train.T` is the transpose, $X^T$. `A_train.T @ A_train` computes $X^TX$ and
`A_train.T @ y_train` computes $X^Ty$ — two matrix products via `@`. Passing both into
`np.linalg.solve` is precisely $\theta^* = (X^TX)^{-1}X^Ty$ from Concept, computed
without ever forming the inverse explicitly. The `for name, coef in zip(...)` loop right
after this line just pairs each of the 12 entries of `theta_ne` with its feature name
for printing — it has no effect on the fitted values.

**9. Gradient descent: the same update rule as Concept, applied as one vectorized
expression per iteration.**

```python
    theta_gd = np.zeros(A_train.shape[1])
    print(f"iter {0:>5}  train MSE: {mse(y_train, A_train @ theta_gd):.4f}")
    for it in range(1, ITERS + 1):
        grad = (2 / len(y_train)) * A_train.T @ (A_train @ theta_gd - y_train)
        theta_gd -= LR * grad
        if it in CHECKPOINTS:
            print(f"iter {it:>5}  train MSE: {mse(y_train, A_train @ theta_gd):.4f}")
```

`theta_gd` starts at all zeros — the "start anywhere" of Concept, deliberately naive.
Inside the loop, `A_train @ theta_gd - y_train` is the residual vector $X\theta - y$;
multiplying by `A_train.T` and scaling by `2 / len(y_train)` reproduces
$\frac{2}{n}X^T(X\theta-y)$ — the full gradient over all 1279 rows and 12 parameters,
computed in one line with no explicit loop over rows. `theta_gd -= LR * grad` is
$\theta \leftarrow \theta - \alpha\nabla J(\theta)$ applied in place.

**10. Finally, both solutions are scored and compared.**

```python
    print_metrics(A_train, y_train, A_test, y_test, theta_gd)
    print(f"max |theta_gd - theta_ne|: {np.max(np.abs(theta_gd - theta_ne)):.4e}")
```

This last line produces the `9.6043e-13` you'll see in Expected output below — proof
that 2000 steps of gradient descent land on essentially the same point the normal
equation reaches in a single algebraic step.

```bash
python src/lesson04_linear_regression/from_scratch.py
```

Expected output:

```text
== Linear regression: from scratch ==
rows: 1599  train: 1279  test: 320

-- Normal equation --
term                        coef
intercept                 5.6341
fixed acidity             0.0353
volatile acidity         -0.2046
citric acid              -0.0522
residual sugar            0.0222
chlorides                -0.0916
free sulfur dioxide       0.0364
total sulfur dioxide     -0.1057
density                  -0.0356
pH                       -0.0727
sulphates                 0.1591
alcohol                   0.2944
train MSE: 0.4317  train R^2: 0.3570
test  MSE: 0.3595  test  R^2: 0.3728

-- Batch gradient descent (lr=0.1, 2000 iterations) --
iter     0  train MSE: 32.4144
iter   100  train MSE: 0.4318
iter   500  train MSE: 0.4317
iter  2000  train MSE: 0.4317
train MSE: 0.4317  train R^2: 0.3570
test  MSE: 0.3595  test  R^2: 0.3728
max |theta_gd - theta_ne|: 9.6043e-13
```

Things to notice:

- Because the features are standardized, coefficients are comparable: alcohol (+0.29)
  and volatile acidity (−0.20) dominate. More alcohol → higher rated wine; more volatile
  acidity (vinegar taint) → lower. Comparing *raw* (non-standardized) coefficients would
  be misleading, since a feature's scale alone can shrink or inflate its coefficient.
- Gradient descent starts at $\theta = 0$ (MSE 32.41 — the mean quality is ≈ 5.64, and
  $5.64^2 \approx 32$) and is within a hair of the optimum by iteration 100. That first
  MSE is exactly what you'd get from `mse(y_train, np.zeros_like(y_train))`: predicting
  0 for every wine.
- After 2000 iterations the GD coefficients agree with the closed-form solution to
  ~$10^{-13}$ — two completely different algorithms, one answer, because the loss is
  convex. (This is the one line printed in scientific notation, precisely because it is
  indistinguishable from zero at 4 decimals.)
- $R^2 \approx 0.36$: taste is noisy, and a linear model explains only a third of the
  variance in quality. That is a finding, not a failure.

## With a library

The same numbers from scikit-learn, then statsmodels for what scikit-learn doesn't give
you: standard errors and p-values. Read
[`with_library.py`](../src/lesson04_linear_regression/with_library.py) — note that it
replicates the *identical* NumPy shuffle rather than calling `train_test_split`, so that
every row lands in the same set and every number matches from-scratch exactly.

### Step-by-step: reading the code

`with_library.py` mirrors `from_scratch.py`'s split and standardization exactly, then
swaps the hand-written linear algebra for library calls.

**Same split, loaded via pandas instead of `np.genfromtxt`.**

```python
    df = pd.read_csv(DATA, sep=";")
    feature_names = [c for c in df.columns if c != "quality"]
    X = df[feature_names].to_numpy(dtype=float)
    y = df["quality"].to_numpy(dtype=float)
```

`pd.read_csv(..., sep=";")` handles the semicolon delimiter directly, no manual header
parsing needed. `.to_numpy(dtype=float)` converts the pandas DataFrame/Series columns
back into plain NumPy arrays, so the split code that follows — same `SEED`, same
`rng.permutation(n)` — is copy-pasted from `from_scratch.py` rather than replaced with
`sklearn.model_selection.train_test_split`. That is deliberate: it guarantees both
scripts put the same rows in train and test, so every printed number matches to the
digit.

**scikit-learn solves the same least-squares problem as the normal equation.**

```python
    model = LinearRegression()
    model.fit(X_train, y_train)
```

`LinearRegression` (with its default `fit_intercept=True`) fits the same
$\theta^*=(X^TX)^{-1}X^Ty$ under the hood — but it does not need the "glue a column of
ones onto $X$" trick from `from_scratch.py`, because it fits the intercept
(`model.intercept_`) separately from the per-feature coefficients (`model.coef_`).
`model.predict(X_train)` computes $\hat y = X\theta$, and
`mean_squared_error`/`r2_score` from `sklearn.metrics` are library equivalents of the
`mse`/`r2` functions you already read.

**statsmodels fits the identical model again, to get inference statistics scikit-learn
does not provide.**

```python
    A_train = sm.add_constant(pd.DataFrame(X_train, columns=feature_names))
    ols = sm.OLS(y_train, A_train).fit()
```

`sm.add_constant` is statsmodels' equivalent of prepending the ones column — unlike
scikit-learn, statsmodels does not add an intercept automatically. `sm.OLS(y_train,
A_train).fit()` solves the same normal equation (`ols.params` matches scikit-learn's
`coef_`/`intercept_`), but additionally computes `ols.bse` (standard errors) and
`ols.pvalues` — quantities that describe how much $\theta$ would vary across
hypothetical resamples of the data, something neither the closed-form solve nor
gradient descent computes, because it isn't needed to find $\theta$ itself.

```bash
python src/lesson04_linear_regression/with_library.py
```

Expected output:

```text
== Linear regression: with scikit-learn & statsmodels ==
rows: 1599  train: 1279  test: 320

-- scikit-learn LinearRegression --
term                        coef
intercept                 5.6341
fixed acidity             0.0353
volatile acidity         -0.2046
citric acid              -0.0522
residual sugar            0.0222
chlorides                -0.0916
free sulfur dioxide       0.0364
total sulfur dioxide     -0.1057
density                  -0.0356
pH                       -0.0727
sulphates                 0.1591
alcohol                   0.2944
train MSE: 0.4317  train R^2: 0.3570
test  MSE: 0.3595  test  R^2: 0.3728

-- statsmodels OLS (train split) --
R-squared: 0.3570
feature                 coef   std err      p-value
alcohol               0.2944    0.0324   3.7314e-19
volatile acidity     -0.2046    0.0251   9.0296e-16
sulphates             0.1591    0.0220   8.6946e-13
```

The scikit-learn block is character-for-character identical to the normal-equation block
above — `LinearRegression` solves the same least-squares problem. The statsmodels block
adds *inference*: each standard error measures how much the coefficient would wobble
across resamples, and the p-value tests $H_0$: coefficient $= 0$ (printed in scientific
notation because all three are far below 0.0001). All three features are overwhelmingly
significant — with 1279 training rows, even modest effects are detected with confidence.

## Check your understanding

1. The normal equation needs $X^TX$ to be invertible. Name two situations with real data
   where it isn't (hint: duplicate a column; or $d > n$). What happens to gradient
   descent in those cases?
2. Why must the test set be standardized with the *train* mean and std? What could go
   wrong if you standardized before splitting?
3. Gradient descent took ~100 iterations to converge here. What would happen to the
   learning rate you're allowed to use if the features were *not* standardized?
4. Train $R^2$ is 0.3570 but test $R^2$ is 0.3728 — the model looks *better* on unseen
   data. Is that a bug? What does it tell you about the variance of a 320-row test
   estimate?
5. The alcohol coefficient is 0.2944 on standardized features. Restate its meaning in
   original units (the train std of alcohol is ≈ 1.08 %vol).
6. `A_train` has shape `(1279, 12)`. What is the shape of `A_train.T @ A_train`, and why
   does that shape depend on the number of features but not on the number of training
   rows? What would change if you had 100,000 rows instead of 1279?
7. The gradient in `from_scratch.py` is computed as one vectorized NumPy expression over
   all rows at once, rather than an explicit Python `for` loop over rows. What would
   happen to the printed numbers — not just the speed — if that line accidentally used
   `A_test`/`y_test` in place of `A_train`/`y_train` inside the gradient descent loop?

From here, you're now ready to proceed to try and reproduce
[Lesson 05 — Logistic Regression](05-logistic-regression.md).

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
