# Lesson 05 — Logistic Regression

Lesson 04 predicted a number; now the target is binary — did this passenger survive the
Titanic or not? You will see why fitting a straight line to 0/1 labels is the wrong tool,
fix it with the sigmoid, derive the right loss from the Bernoulli likelihood, and train
the whole thing with the same gradient descent loop you already know.

## Concept

Lesson 04 introduced the coefficient vector $\theta$ (theta) and design matrix $X$; this lesson reuses
both, plus a few new symbols. Before the formulas, here is what each one means:

- $x$ — one passenger's feature vector (the six standardized numbers: Pclass, Sex, Age, SibSp, Parch,
  Fare).
- $\theta$ (theta) — the parameter vector the model learns: one weight per feature, plus an intercept.
  This is the "coef" column printed by both scripts.
- $x^T\theta$ — the dot product of a passenger's features with the weights: multiply each feature by its
  weight and add the results (plus the intercept). A single number.
- $z$ — shorthand for that dot product, the *linear score* (also called the "logit").
- $y_i$ — the true label for passenger $i$: 1 if they survived, 0 if they did not.
- $p_i$ (or just $p$) — the model's predicted *probability* that $y_i = 1$.
- $n$ — the number of training examples (rows) a formula averages over.

**Why not linear regression?** A linear model $\hat{y} = x^T\theta$ happily predicts
$-0.3$ or $1.7$, which are nonsense as probabilities. Worse, squared error punishes a
confidently-correct prediction (say 1.7 for a true 1) as if it were an error. We need a
model whose output *is* a probability.

**Sigmoid.** Keep the linear score $z = x^T\theta$, but squash it:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

$\sigma$ (sigma) names this *sigmoid function*; $e$ is Euler's number ($\approx 2.71828$). Feed it any
real-valued score $z$ — however large or negative — and it returns a number strictly between 0 and 1, a
valid probability.

$\sigma(0) = 0.5$, large positive $z$ → near 1, large negative $z$ → near 0.

**Odds and log-odds.** Inverting the sigmoid gives $\log\frac{p}{1-p} = x^T\theta$: the
model is *linear in the log-odds*. Each coefficient is the change in log-odds per unit of
its feature (per one standard deviation here, since we standardize); $e^{\theta_j}$ is
the multiplicative change in the odds. That is what makes logistic regression a favorite
in medicine and the social sciences — the coefficients mean something. Concretely, the *odds* of an
event are the ratio $\frac{p}{1-p}$ ("3-to-1 odds" means $p = 0.75$), and the *log-odds* (or "logit") is
the natural log of that ratio — the exact quantity the model treats as a plain weighted sum of features.

**From likelihood to loss.** Model each label as a Bernoulli draw:
$P(y_i \mid x_i) = p_i^{y_i}(1-p_i)^{1-y_i}$. Maximizing the log-likelihood over the
dataset is the same as minimizing its negative mean — the *binary cross-entropy*:

$$J(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]$$

$J(\theta)$ is the number training tries to minimize (the *cost*, or *loss*); the sum runs over every
training row $i$, and $\log$ is the natural logarithm. Notice the "if/else" trick baked into the
formula: since $y_i$ is always exactly 0 or 1, only one of the two bracketed terms survives per row —
$y_i \log p_i$ when the true label is 1, or $(1-y_i)\log(1-p_i)$ when it is 0. So, row by row, the loss
is simply $-\log(\text{probability the model assigned to the correct answer})$: confident and correct is
cheap, confident and wrong is expensive.

This loss is convex in $\theta$ (unlike MSE-through-a-sigmoid), so gradient descent finds
the global optimum. And its gradient collapses to something you have seen before:

$$\nabla J = \frac{1}{n}X^T(p - y)$$

$\nabla J$ (read "gradient of J", nabla J) is a vector with one entry per parameter, pointing in the
direction that increases the loss fastest — gradient descent steps *against* it. $X$ is the whole
training design matrix (one row per passenger), and $p - y$ is the vector of prediction errors
(predicted probability minus true label, one per passenger). The formula above is
identical in form to linear regression's gradient, with the sigmoid output in place of
the linear one. This is not luck: sigmoid and cross-entropy are a matched pair (the
canonical link and loss of the Bernoulli model), and the messy $\sigma'$ term cancels
exactly.

### Why cross-entropy, not squared error

Swap in the sigmoid but keep squared error ($\frac{1}{n}\sum_i (p_i - y_i)^2$) instead of cross-entropy,
and two problems appear:

1. **The loss stops being convex.** Composing squared error with the sigmoid produces a wavy surface
   with flat spots and local minima, so gradient descent can get stuck short of the best fit.
2. **Confident, wrong predictions barely get punished.** Suppose the true label is 1 and the model
   outputs $p = 0.01$ — badly wrong. Out at the extremes the sigmoid is nearly flat ($\sigma'(z) \approx
   0$), so a squared-error gradient shrinks toward zero exactly where it should be largest — the model
   would learn almost nothing from its worst mistakes. Cross-entropy avoids this: its $\sigma'$ term
   cancels algebraically (as shown above), leaving a gradient directly proportional to the plain error
   $p - y$. A wildly wrong prediction produces a correspondingly large, useful gradient.

Cross-entropy also has a direct probabilistic justification — it falls out of maximizing the likelihood
of the labels you actually observed — whereas squared error on probabilities has no such story.

### A tiny hand-worked example

The real run below uses 712 rows and 6 standardized features, too much to trace by hand. Here is a
miniature, one-example, one-feature version of the same update rule, just to see the arithmetic.

Say one passenger has standardized feature $x = 1$, true label $y = 1$ (survived), and the model starts
at $\theta_0 = 0$ (intercept) and $\theta_1 = 0$ (weight) — exactly how `from_scratch.py` initializes
`theta`.

- Score: $z = \theta_0 + \theta_1 x = 0$.
- Prediction: $p = \sigma(0) = 0.5$.
- Loss for this row: $-[y\log p + (1-y)\log(1-p)] = -\log(0.5) \approx 0.6931$. That is exactly the
  $\ln 2$ starting loss printed as `iter 0` below — with all-zero parameters every passenger gets
  $p = 0.5$, so the mean loss is $\ln 2$ regardless of dataset size.
- Gradient (dropping the $\frac{1}{n}$ since there is only one row here): both
  $\partial J/\partial\theta_0$ and $\partial J/\partial\theta_1$ equal $(p - y) \cdot x =
  (0.5 - 1)\times 1 = -0.5$.
- One gradient descent step with learning rate 0.5: $\theta \leftarrow \theta - 0.5\times(-0.5) = 0.25$
  for both parameters.
- New score: $z = 0.25 + 0.25 \times 1 = 0.5$; new prediction: $p = \sigma(0.5) \approx 0.6225$ — closer
  to the true label 1 than before. One step of gradient descent moved the prediction in the right
  direction, exactly as the formulas promise.

**Decision threshold.** The model outputs probabilities; to get a yes/no answer we
predict "survived" when $p \ge 0.5$ — the point where the odds pass 1:1. The threshold is
a *choice*, not a law; lesson 06 explores what happens when you move it.

## Dataset

The Titanic passenger manifest — 891 passengers with class, sex, age, family counts,
fare, and survival. A classic because the signal is strong and human-readable ("women
and children first" is visible in the coefficients). Historical public-domain data;
we download the CSV mirror published by
[Data Science Dojo](https://github.com/datasciencedojo/datasets).

```powershell
.\scripts\download_data.ps1 titanic
```

Expected output:

```text
[get ] titanic.csv
[ ok ] titanic.csv (checksum verified)
Done.
```

Take a look at the data before modeling anything (always do this):

```bash
head -3 data/titanic.csv
```

```text
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
```

We use six features: `Pclass`, `Sex` (encoded male=1, female=0), `Age`, `SibSp`
(siblings/spouses aboard), `Parch` (parents/children aboard), `Fare`. Target:
`Survived`. `Age` is missing for 177 passengers — we impute the *train* median (leakage
rule from lesson 04: every statistic used in preprocessing comes from the training rows
only). Note the quoted commas inside names: this is why we parse with a real CSV reader,
never `line.split(",")`.

## From scratch

Read [`from_scratch.py`](../src/lesson05_logistic_regression/from_scratch.py) first.
The pipeline (split → impute → standardize) mirrors lesson 04; the new parts are
`sigmoid`, `log_loss`, and a gradient descent loop whose gradient line is almost
identical to lesson 04's — see the comment in the code explaining why.

### Step-by-step: reading the code

Follow the file top to bottom — this is the order Python actually executes it.

**Imports and constants**

```python
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "titanic.csv"
SEED = 42
LR = 0.5  # gradient descent learning rate
ITERS = 20000  # gradient descent iterations
CHECKPOINTS = (0, 10, 100, 1000, 20000)

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
```

`csv` is the standard-library CSV reader, used instead of `str.split(",")` because — as the Dataset
section notes — some passenger names contain commas inside quotes. `Path(__file__).resolve().parents[2]`
walks up two directories from this file (`src/lesson05_logistic_regression/` → `src/` → repo root) to
find `data/titanic.csv` no matter what directory you run the script from. `SEED` fixes the random number
generator so the train/test split is identical every run, and identical to `with_library.py`'s split.
`LR` (learning rate) and `ITERS` are the two knobs of gradient descent; `CHECKPOINTS` just lists the
iteration numbers to print progress at, matching the "iter" lines in the expected output below.

**`load_data`: CSV rows to NumPy arrays**

```python
def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Features: Pclass, Sex (male=1, female=0), Age (NaN if missing),
    SibSp, Parch, Fare. Target: Survived."""
    with open(DATA, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array(
        [
            [
                float(r["Pclass"]),
                1.0 if r["Sex"] == "male" else 0.0,
                float(r["Age"]) if r["Age"] != "" else np.nan,
                float(r["SibSp"]),
                float(r["Parch"]),
                float(r["Fare"]),
            ]
            for r in rows
        ]
    )
    y = np.array([float(r["Survived"]) for r in rows])
    return X, y
```

`csv.DictReader` turns each CSV row into a dict keyed by column header, so `r["Pclass"]` is readable
instead of a magic index like `r[2]`. The nested list comprehension builds one Python list per passenger,
and wrapping it in `np.array(...)` converts the whole thing into a 2-D `float` array `X` in a single
call. Two encoding decisions happen inline: `Sex` (text `"male"`/`"female"`) becomes `1.0`/`0.0` because
the model only understands numbers, and a missing `Age` field (an empty string in the CSV) becomes
`np.nan` rather than `0.0` — a real missing value must never be silently treated as age zero. `y` is a
separate 1-D array, one label per passenger, built the same way.

**`sigmoid`: the squashing function from Concept**

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Map any real score to a probability in (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))
```

This is $p = \sigma(z) = \frac{1}{1+e^{-z}}$ from Concept, typed almost verbatim. `np.exp(-z)` computes
$e^{-z}$ element-wise, so `z` can be a single number or an entire array of scores (one per passenger) —
the function is called both ways later in the file.

**`log_loss`: binary cross-entropy**

```python
def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Binary cross-entropy: the negative mean Bernoulli log-likelihood."""
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
```

This is $J(\theta) = -\frac{1}{n}\sum_i\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$. `y * np.log(p)` and
`(1 - y) * np.log(1 - p)` are element-wise NumPy operations that produce one per-row loss term each;
`np.mean` performs the $\frac{1}{n}\sum$ in one call, and the leading minus sign matches the formula.
Because `y` only ever holds 0.0 or 1.0, exactly one of the two additive terms is nonzero per row — the
"if/else" behavior described in Concept.

**`accuracy`: thresholding probabilities**

```python
def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Fraction correct after thresholding the probability at 0.5."""
    return float(np.mean((p >= 0.5) == y))
```

`p >= 0.5` produces a boolean array — the decision threshold from Concept. Comparing it against `y`
(NumPy treats the 0.0/1.0 floats as booleans for the comparison) gives another boolean array answering
"was this row correct?"; `np.mean` over booleans counts `True` as 1 and `False` as 0, so the mean is
exactly the fraction correct.

**`main`: the driver, piece by piece**

```python
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
```

Same split as lesson 04: a seeded random number generator produces one fixed shuffle (`perm`) of row
indices, and the first 80% become the train set, the rest the test set. Fixing `SEED` is what lets
`with_library.py` reproduce the exact same rows later, so the two scripts are comparable.

```python
    age_median = float(np.nanmedian(X_train[:, 2]))
    print(f"train median Age used for imputation: {age_median:.4f}")
    for split in (X_train, X_test):
        split[np.isnan(split[:, 2]), 2] = age_median
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
```

`X_train[:, 2]` selects the Age column (index 2 in `FEATURES`); `np.nanmedian` computes the median while
ignoring `NaN`s — the train-only statistic used to fill missing ages (the no-leakage rule from lesson 04:
this number comes only from training rows, then gets reused to fill test rows too). `np.isnan(split[:,
2])` is a boolean mask marking which rows still have a missing age; boolean-array indexing
(`split[mask, 2] = age_median`) overwrites only those cells. `mu` and `sigma` (mean and standard
deviation, again computed on train only) then standardize every feature to roughly mean 0, unit
variance — which is why the coefficients in the output are "per standard deviation," as Concept
mentions.

```python
    A_train = np.column_stack([np.ones(len(X_train)), X_train])
    A_test = np.column_stack([np.ones(len(X_test)), X_test])
```

`np.column_stack` glues a column of all-ones onto the left of the feature matrix. This is the standard
trick for folding the intercept into a single matrix multiply: instead of computing $z = \theta_0 +
x^T\theta_{1:}$ as two separate pieces, the extra constant-1 column lets one `A @ theta` compute both at
once, with `theta[0]` playing the role of the intercept.

```python
    theta = np.zeros(A_train.shape[1])
    print(f"iter {0:>5}  train log-loss: {log_loss(y_train, sigmoid(A_train @ theta)):.4f}")
    for it in range(1, ITERS + 1):
        p = sigmoid(A_train @ theta)
        grad = A_train.T @ (p - y_train) / len(y_train)
        theta -= LR * grad
        if it in CHECKPOINTS:
            print(f"iter {it:>5}  train log-loss: {log_loss(y_train, sigmoid(A_train @ theta)):.4f}")
```

This is gradient descent on the cross-entropy loss, matching the Concept formulas line for line: `theta`
starts at all zeros (the exact $\theta = 0$ case worked out by hand above); `A_train @ theta` computes
$z$ for every row in one matrix-vector product; `sigmoid(...)` turns those scores into probabilities
$p$; and `A_train.T @ (p - y_train) / len(y_train)` is exactly $\nabla J = \frac{1}{n}X^T(p-y)$.
`theta -= LR * grad` performs the update $\theta \leftarrow \theta - \text{lr} \cdot \nabla J$ in place.
Printing happens only when `it` is one of the `CHECKPOINTS`, which is why the expected output jumps
straight from iteration 0 to 10, 100, 1000, and 20000 instead of printing all 20000 lines.

```python
    print(f"{'term':<11} {'coef':>9}")
    for name, coef in zip(["intercept"] + FEATURES, theta):
        print(f"{name:<11} {coef:>9.4f}")
```

`["intercept"] + FEATURES` (plain list concatenation) lines up names with the fitted `theta` vector,
since `theta[0]` is the intercept and `theta[1:]` matches `FEATURES` in order; `zip` pairs the two lists
up for printing. `{name:<11}` and `{coef:>9.4f}` are Python's format-spec mini-language: left-align the
name in an 11-character field, right-align the coefficient in a 9-character field with 4 decimal
places — purely cosmetic column alignment, chosen so the two scripts' outputs line up character for
character.

```python
    p_train, p_test = sigmoid(A_train @ theta), sigmoid(A_test @ theta)
    print()
    print(f"train accuracy: {accuracy(y_train, p_train):.4f}  train log-loss: {log_loss(y_train, p_train):.4f}")
    print(f"test  accuracy: {accuracy(y_test, p_test):.4f}  test  log-loss: {log_loss(y_test, p_test):.4f}")
```

The final predictions reuse the same `sigmoid` and design-matrix machinery on both splits; then
`accuracy` and `log_loss` — the same two functions defined earlier and used to print every checkpoint —
score them one last time to report the final train/test numbers.

```python
if __name__ == "__main__":
    main()
```

The standard Python idiom: `main()` runs only when the file is executed directly
(`python from_scratch.py`), not when it is imported — useful if a later lesson wants to reuse `sigmoid`
or `log_loss` without re-running the whole script.

```bash
python src/lesson05_logistic_regression/from_scratch.py
```

Expected output:

```text
== Logistic regression: from scratch ==
rows: 891  train: 712  test: 179
train median Age used for imputation: 28.0000

-- Batch gradient descent (lr=0.5, 20000 iterations) --
iter     0  train log-loss: 0.6931
iter    10  train log-loss: 0.4753
iter   100  train log-loss: 0.4312
iter  1000  train log-loss: 0.4311
iter 20000  train log-loss: 0.4311

term             coef
intercept     -0.6613
Pclass        -0.9829
Sex           -1.3635
Age           -0.6395
SibSp         -0.4882
Parch         -0.0713
Fare           0.1161

train accuracy: 0.8076  train log-loss: 0.4311
test  accuracy: 0.7430  test  log-loss: 0.4950
```

Things to notice:

- The starting loss is $0.6931 = \ln 2$: with $\theta = 0$ the model says $p = 0.5$ for
  everyone — maximum ignorance.
- `Sex` has the largest coefficient (−1.36): being male (Sex=1) sharply lowers the
  survival log-odds. `Pclass` is next (−0.98): higher class *number* (3rd class) means
  lower survival. The manifest agrees with the history books.
- We run 20000 iterations to nail the optimum to many decimals (so the library
  comparison is exact), but the loss is already converged to 4 decimals by iteration
  ~1000.
- Test accuracy (0.7430) is well below train (0.8076) — the honest number is the one on
  passengers the model never saw.
- The `CHECKPOINTS` (0, 10, 100, 1000, 20000) are the only iterations printed — the loop still runs all
  20000 times, but printing on every iteration would drown the useful signal (how fast the loss flattens
  out) in noise.
- The column of ones prepended to `A_train`/`A_test` is what lets one matrix multiply, `A @ theta`,
  produce the intercept plus the weighted feature sum in a single step — see the design-matrix note in
  Step-by-step above.

## With a library

Read [`with_library.py`](../src/lesson05_logistic_regression/with_library.py). It builds
the identical matrix (pandas for loading, the same NumPy shuffle for the split) and fits
`LogisticRegression(C=1e9)`. In scikit-learn, `C` is the *inverse* regularization
strength — the default `C=1.0` adds an L2 penalty that from-scratch doesn't have, so we
crank `C` up to make the penalty negligible and the two objectives identical.

### Step-by-step: reading the code

`with_library.py` follows the identical shape as `from_scratch.py` (load → split → impute →
standardize), so only what differs is called out here.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
```

`pandas` replaces the manual `csv.DictReader` loop for loading; `LogisticRegression` is scikit-learn's
model class; `accuracy_score` and `log_loss` (from `sklearn.metrics`) replace the hand-written
`accuracy` and `log_loss` functions that `from_scratch.py` defines itself.

```python
    df = pd.read_csv(DATA)
    df["Sex"] = (df["Sex"] == "male").astype(float)  # male=1, female=0
    X = df[FEATURES].to_numpy(dtype=float)  # Age keeps NaN where missing
    y = df["Survived"].to_numpy(dtype=float)
```

`pd.read_csv` parses the same file in one call, handling quoted commas in names automatically.
`(df["Sex"] == "male")` produces a boolean column; `.astype(float)` turns it into 1.0/0.0 — the same
encoding `from_scratch.py` did with a Python conditional inside a list comprehension.
`df[FEATURES].to_numpy(dtype=float)` selects the six feature columns, in the same order as `FEATURES`,
and converts the DataFrame to a plain NumPy array — from here on the code is doing the same job as
`from_scratch.py`.

The split, Age imputation, and standardization blocks that follow are copy-identical to
`from_scratch.py` (same `SEED`, same `np.random.default_rng`, same `nanmedian`/`mean`/`std` calls) — that
identity is exactly what makes the two scripts' outputs match to 4 decimals below.

```python
    model = LogisticRegression(C=1e9, max_iter=10000, tol=1e-10)
    model.fit(X_train, y_train)
```

This is the entire training step: one object construction and one `.fit()` call replace the
from-scratch `theta = np.zeros(...)` loop. `C` is scikit-learn's *inverse* L2 regularization strength —
large `C` means almost no penalty — so `C=1e9` reproduces the unregularized objective from Concept.
`max_iter` and `tol` raise the iteration cap and tighten the convergence tolerance so the solver
(L-BFGS by default — a quasi-Newton method, not plain gradient descent) runs long enough to match the
from-scratch fit to 4 decimal places instead of stopping early.

```python
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
```

`predict_proba` returns a 2-column array — probability of class 0 and probability of class 1 for every
row; `[:, 1]` keeps only the "survived" column, giving the same $p$ vector `from_scratch.py` computes
with `sigmoid(A @ theta)`.

```python
    print(
        f"train accuracy: {accuracy_score(y_train, p_train >= 0.5):.4f}  "
        f"train log-loss: {log_loss(y_train, p_train):.4f}"
    )
```

`p_train >= 0.5` is the same threshold used in the from-scratch `accuracy` function; `accuracy_score` and
`log_loss` here are scikit-learn's metric functions computing the identical quantities from Concept.

```python
    default_model = LogisticRegression(max_iter=10000)
    default_model.fit(X_train, y_train)
    print(
        f"default LogisticRegression (C=1.0) test accuracy: "
        f"{default_model.score(X_test, y_test):.4f}"
    )
```

A second model, this time with scikit-learn's default `C=1.0` — real L2 regularization — fit on the
same data. `.score(X_test, y_test)` is a convenience method equivalent to `accuracy_score(y_test,
model.predict(X_test))`; its result is the "extra line" mentioned in the Expected output below.

```bash
python src/lesson05_logistic_regression/with_library.py
```

Expected output:

```text
== Logistic regression: with scikit-learn ==
rows: 891  train: 712  test: 179
train median Age used for imputation: 28.0000

-- LogisticRegression(C=1e9), effectively unregularized --
term             coef
intercept     -0.6613
Pclass        -0.9829
Sex           -1.3635
Age           -0.6395
SibSp         -0.4882
Parch         -0.0713
Fare           0.1161

train accuracy: 0.8076  train log-loss: 0.4311
test  accuracy: 0.7430  test  log-loss: 0.4950
```

followed by one extra line:

```text
default LogisticRegression (C=1.0) test accuracy: 0.7430
```

Every coefficient and metric matches from-scratch to all 4 printed decimals, even though
scikit-learn uses L-BFGS (a quasi-Newton optimizer) rather than plain gradient descent —
the loss is convex, so any competent optimizer run to convergence lands on the same
unique optimum. The regularized default (C=1.0) happens to give the same test *accuracy*
here — with 712 rows and only 6 features there is little overfitting for regularization
to fix; its coefficients are slightly shrunk toward zero all the same. Regularization
earns its keep when features are many and data is scarce — later lessons return to this.

## Check your understanding

1. Suppose you trained a plain linear regression on the 0/1 labels and thresholded its
   output at 0.5. Give two concrete ways this model is worse than logistic regression,
   even if its accuracy happens to be similar.
2. Why is the iteration-0 loss exactly $\ln 2 \approx 0.6931$? What would it be if you
   initialized the intercept to a large positive value instead?
3. The `Sex` coefficient is −1.3635. If we had encoded female=1 instead of male=1, what
   would the coefficient become, and what would happen to the model's predictions?
4. Derive $e^{\theta_j}$ as the odds ratio for a one-unit feature increase from the
   log-odds equation. Why does standardization change the number but not the model?
5. Imputing missing ages with the overall (train+test) median instead of the train
   median usually changes the metrics only slightly. Why do we still forbid it?
6. In `main()`, the design matrix is built as
   `A_train = np.column_stack([np.ones(len(X_train)), X_train])`. What does the column of ones
   accomplish, and what would go wrong (be specific about the shapes involved) if `theta` were instead
   fit directly against `X_train`, with no ones column?
7. `with_library.py` passes `max_iter=10000` and `tol=1e-10` to `LogisticRegression(C=1e9, ...)`, while
   `from_scratch.py` always runs a fixed `ITERS = 20000` regardless of how converged the loss already
   is. What is each script's actual stopping rule, and why does scikit-learn's solver need an explicit
   tolerance when the from-scratch loop does not?

From here, you're now ready to proceed to try and reproduce
[Lesson 06 — Model Evaluation](06-model-evaluation.md).

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
