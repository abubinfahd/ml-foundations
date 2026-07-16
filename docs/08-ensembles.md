# Lesson 08 — Ensembles

Lesson 07 capped tree depth to fight overfitting. This lesson takes the opposite route:
keep the trees deep and wrong, but grow *many* of them on different views of the data and
let them vote. You implement bagging from scratch (the base tree comes from sklearn — the
ensemble logic is the lesson here, not the tree), then compare against scikit-learn's
random forest and gradient boosting.

## Concept

**A single deep tree has high variance.** Grown to purity, a tree memorizes its training
sample; redraw the sample and you get a very different tree. Its errors are less about
systematic bias and more about instability — which is exactly the kind of error that
averaging can remove.

If "bias" and "variance" are new terms: **bias** is the error a model makes because its
assumptions are too simple to represent the true pattern (for example, a straight line
trying to fit a curve — it is wrong the same way no matter which sample you give it).
**Variance** is the error a model makes because it is too sensitive to *which* training
rows it happened to see (a deep tree draws a very different decision boundary from two
random samples drawn from the same population, even though neither sample is "wrong").
A single deep tree has low bias — it is flexible enough to represent almost any pattern —
but high variance: swap a handful of training rows and its predictions on new data shift
noticeably. Ensembling exploits a simple statistical fact: averaging many
noisy-but-unbiased estimates cancels out noise that points in random directions on each
one, while a persistent, one-directional error (bias) does not cancel because every copy
makes it the same way. That is why bagging targets variance specifically, and why it pairs
naturally with deep, low-bias, high-variance trees rather than, say, shallow ones.

**Bootstrap sampling.** A *bootstrap sample* draws $n$ rows from the training set *with
replacement* — about $1 - e^{-1} \approx 63\%$ of distinct rows appear, some multiple
times. Fitting one deep tree per bootstrap sample gives $B$ different trees that are all
roughly unbiased for the same problem but disagree in their noise. **Bagging** (bootstrap
aggregating) takes their majority vote.

Spelling out the symbols: $n$ is the number of rows in the training set, so "draw $n$ rows
with replacement" means each bootstrap sample is the same *size* as the original training
set, just resampled row by row; "with replacement" means every draw picks independently
from all $n$ rows, so the same row can be picked more than once and, on average, about a
third of the rows are left out of any given draw entirely; $e \approx 2.71828$ is Euler's
number, and $1 - e^{-1}$ is the limiting chance, as $n$ grows, that a specific row gets
picked at least once across $n$ independent draws; $B$ is the number of trees in the
ensemble (25 in this lesson's from-scratch script, later 100 in scikit-learn's random
forest).

**A small illustrative example.** Suppose you have three classifiers, each correct 70% of
the time on a given example, and suppose their mistakes are independent of one another (one
classifier being wrong tells you nothing about whether another is also wrong on that same
example). Take a majority vote of the three. The vote is wrong only when two or more of the
three are wrong simultaneously. With independent 30% error rates, the probability of two or
more simultaneous errors is $3(0.3)^2(0.7) + (0.3)^3 = 0.189 + 0.027 = 0.216$ — so the vote
is correct about 78.4% of the time, better than any single 70%-accurate member on its own.
This is the same principle Condorcet described for juries in 1785: independent,
better-than-chance voters become more reliable in aggregate, and more voters push accuracy
closer to 100%. The catch is the word "independent" — if the three classifiers tend to make
the *same* mistakes (their errors are correlated), the majority vote inherits those shared
mistakes and the gain shrinks or vanishes. That catch is exactly what the variance formula
below quantifies.

**Why averaging works.** For $B$ models with individual variance $\sigma^2$ and average
pairwise correlation $\rho$, the variance of their average is

$$\mathrm{Var}(\bar{X}) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

where $\bar{X}$ is the ensemble's averaged output (the fraction of trees voting for a
class, in the majority-vote case above), $\sigma^2$ is the variance any *single* one of the
$B$ models would have by itself, and $\rho$ (the Greek letter rho) measures how correlated
two models' errors tend to be: $\rho = 1$ means the models always err together and
averaging buys nothing, while $\rho = 0$ means their errors are statistically independent
and averaging buys the most.

The second term vanishes as $B$ grows, but the first does not: correlated models share
mistakes that voting cannot fix. So the game is to *decorrelate* the trees (lower $\rho$)
without making each one much worse (higher bias).

**Feature subsampling** is the classic decorrelation trick: show each tree only a random
subset of $\sqrt{d}$ features, and trees can no longer all lean on the same dominant
feature. Here $d$ is the total number of features available (35 in this lesson's dataset)
and $\sqrt{d}$ (about 5 or 6) is the smaller number of features randomly offered to any one
tree, forcing different trees to build their splits around different, individually weaker
signals so that their errors decorrelate. This upgrades bagging toward a *random forest* —
with one important caveat: a real random forest redraws the $\sqrt{d}$ candidate features
**at every split**, while our from-scratch version draws them **once per tree**. The output
below shows why that difference matters.

**Boosting** is the other big ensemble idea, and it is *not* averaging: trees are built
*sequentially*, each one trained to fix the errors the current ensemble still makes, and
predictions are a weighted sum of many shallow trees. Bagging reduces variance of deep
trees; boosting reduces bias of shallow ones.

**Feature importance caveats.** A forest's impurity-based importances are free but biased:
they inflate features with many possible split points (continuous ones like `age`) relative
to binary one-hot columns, and they spread credit across correlated features. Read them as
hints, not truth; permutation importance on held-out data is the more honest measurement.

## Dataset

[UCI Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult) — 32,561 rows from
the 1994 US census; the task is predicting whether income exceeds $50K/yr. License: CC BY 4.0.

```powershell
.\scripts\download_data.ps1 adult
```

Expected output:

```text
[get ] adult.data
[ ok ] adult.data (checksum verified)
Done.
```

Take a look at the data before modeling (always do this):

```bash
head -3 data/adult.data
```

```text
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
```

No header row; fields are separated by comma+space; missing values are `?`. The columns
are: age, workclass, fnlwgt, education, education-num, marital-status, occupation,
relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country,
income. Both scripts drop every row containing a `?` (32,561 → 30,162 rows), keep the five
numeric features (age, education-num, capital-gain, capital-loss, hours-per-week), and
one-hot encode workclass, marital-status, occupation and sex in sorted category order —
35 features total.

## From scratch

Read [`from_scratch.py`](../src/lesson08_ensembles/from_scratch.py). `fit_ensemble` is the
whole idea in ~15 lines: for tree $i$, draw a bootstrap sample with `default_rng(42 + i)`,
optionally draw $\sqrt{d} = 5$ of the 35 features, fit a *deep* sklearn tree
(`max_depth=None, random_state=i`), and `vote` by majority (an exact tie predicts `<=50K`,
the first label in sorted order). The 80/20 split shuffles with `default_rng(42)`.

### Step-by-step: reading the code

Read the file top to bottom; this walks the same path in the same order.

**Imports and constants.** The file borrows only the base learner from scikit-learn:

```python
import csv
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
```

`csv` reads the comma-separated data file, `pathlib.Path` locates it relative to the repo
root, `numpy` supplies the random-number generators used for bootstrap and feature
sampling, and `DecisionTreeClassifier` is the single-tree building block — everything that
turns many such trees into an ensemble is code you are about to read, not library code.
Further down, `SEED = 42` and `N_TREES = 25` fix the randomness and the ensemble size $B$
from the Concept section so a re-run reproduces the exact numbers below.

**`load_data` — reading and one-hot encoding.** This function is identical in spirit to
earlier lessons: drop rows with missing values, then turn each categorical column into a
block of 0/1 columns, one per category:

```python
rows = [dict(zip(COLUMNS, r)) for r in raw if "?" not in r]
categories = {c: sorted({r[c] for r in rows}) for c in CATEGORICAL}
names = list(NUMERIC) + [f"{c}={v}" for c in CATEGORICAL for v in categories[c]]
```

The first line keeps only rows with no `"?"` anywhere. The second collects, for each
categorical column, the sorted set of distinct values it takes (sorting makes the encoding
deterministic — the same category always lands in the same column, run after run). The
third builds the final feature-name list: the five numeric names first, then one
`column=value` name per category, in that sorted order. Later, the actual 0/1 values are
written in:

```python
for j, c in enumerate(NUMERIC):
    X[i, j] = float(r[c])
col = len(NUMERIC)
for c in CATEGORICAL:
    X[i, col + categories[c].index(r[c])] = 1.0
    col += len(categories[c])
```

For row `i`, the numeric columns are copied straight across as floats; then, for each
categorical column, `categories[c].index(r[c])` finds *where in that column's sorted
category list* this row's value sits, and a single `1.0` is placed at that offset within the
block reserved for that column (`col` tracks where each column's block starts). Every other
entry in `X` stays `0.0` from the initial `np.zeros`. This is one-hot encoding done by hand:
the same operation `OneHotEncoder` performs in `with_library.py`.

**`fit_ensemble` — bagging, the core of the lesson.** This is where the Concept section's
formulas become code:

```python
for i in range(n_trees):
    rng = np.random.default_rng(SEED + i)
    boot = rng.integers(0, n, size=n)  # sample n rows *with replacement*
    if max_features is None:
        feats = np.arange(d)
    else:
        feats = np.sort(rng.choice(d, size=max_features, replace=False))
    tree = DecisionTreeClassifier(max_depth=None, random_state=i)
    tree.fit(X[boot][:, feats], y[boot])
    trees.append((tree, feats))
```

Each loop iteration builds one tree of the ensemble. `rng.integers(0, n, size=n)` draws $n$
random integers from $0$ up to (but not including) $n$, and — because nothing stops the
same integer from being drawn twice — this *is* the bootstrap sample from the Concept
section: $n$ row indices, with replacement, so some rows appear multiple times in `boot` and
others (about 37%, on average) do not appear at all. Indexing `X[boot]` and `y[boot]` then
materializes that resampled training set. `rng.choice(d, size=max_features, replace=False)`
is a different sampling idiom: it draws `max_features` *distinct* column indices out of the
`d` available (`replace=False` forbids repeats), which is exactly the $\sqrt{d}$ feature
subsampling from the Concept section — you want several different columns, so drawing the
same column twice would waste a slot. Rows are sampled *with* replacement because that is
the definition of a bootstrap sample; features are sampled *without* replacement because you
are choosing a subset of distinct columns, not resampling a population. Every tree gets its
own `rng` seeded from `SEED + i`, and its own `random_state=i` for the tree's internal
tie-breaking — together these make the entire 25-tree ensemble exactly reproducible from a
single `SEED`. The function returns a list of `(tree, feats)` pairs, because at prediction
time each tree must be fed only the same subset of columns it was trained on.

**`vote` — turning many predictions into one.** This is the "aggregating" half of bagging:

```python
def vote(trees: list, X: np.ndarray) -> np.ndarray:
    """Majority vote over the trees. An exact tie (possible when the number
    of trees is even) goes to class 0 ('<=50K', first in sorted label order)."""
    votes = np.zeros(len(X))
    for tree, feats in trees:
        votes += tree.predict(X[:, feats])
    return (2 * votes > len(trees)).astype(int)
```

Because the two class labels are encoded as `0` and `1`, summing every tree's prediction
into `votes` literally counts how many trees voted for class `1` for each test row — no
separate vote-counting logic is needed. `2 * votes > len(trees)` is the same test as
`votes / len(trees) > 0.5` (more than half the trees said `1`), just rearranged to avoid a
division; on an exact tie, `2 * votes` equals `len(trees)`, the comparison is `False`, and
the row is predicted as class `0`, matching the docstring and the Concept section's mention
of tie-breaking.

**The driver block (`main`).** First, the same deterministic 80/20 split used throughout
this course:

```python
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(y))
n_train = int(0.8 * len(y))
train, test = perm[:n_train], perm[n_train:]
```

`rng.permutation(len(y))` shuffles every row index once; slicing the first 80% into `train`
and the rest into `test` then gives a reproducible split with no overlap. Next, the overfit
baseline:

```python
single = DecisionTreeClassifier(max_depth=None, random_state=0)
single.fit(X_train, y_train)
```

one tree, unlimited depth, trained on the *unresampled* training set — this is the
high-variance model the rest of the script tries to tame by averaging. Then the two
ensembles from the Concept section are built side by side:

```python
k = int(np.sqrt(X.shape[1]))  # sqrt(d) features per tree
bagged = fit_ensemble(X_train, y_train, N_TREES, max_features=None)
forest = fit_ensemble(X_train, y_train, N_TREES, max_features=k)
```

`bagged` passes `max_features=None`, so every tree sees all 35 columns and only the
row-level bootstrap resampling decorrelates the trees. `forest` passes `max_features=k`
(`k = 5`, i.e. $\sqrt{d}$), adding the per-tree feature subsampling on top of the same
row-level resampling — this is the "random-forest-style" variant referred to in the
Concept section and the Expected output below. Finally, the growth-curve loop reuses the
same 25 already-fitted trees and simply varies how many of them get a vote:

```python
for m in [1, 5, 10, 25]:
    a_bag = float((vote(bagged[:m], X_test) == y_test).mean())
    a_rf = float((vote(forest[:m], X_test) == y_test).mean())
```

Slicing `bagged[:m]` and `forest[:m]` hands `vote` only the first `m` trees, so this loop
directly visualizes the $\frac{1-\rho}{B}\sigma^2$ term from the Concept section shrinking
as $B$ (here, `m`) grows from 1 to 25 — no new fitting happens, only re-voting.

```bash
python src/lesson08_ensembles/from_scratch.py
```

Expected output:

```text
== Ensembles (bagging / random forest): from scratch ==
rows in file: 32561
rows after dropping '?': 30162
features: 5 numeric + 30 one-hot = 35
train/test split: 24129/6033 (seed 42)

single deep tree (max_depth=None), the overfit baseline:
  train accuracy: 0.9721
  test accuracy:  0.8149

bagging, 25 deep trees (all features):
  test accuracy:  0.8409
random-forest-style, 25 deep trees (sqrt(d)=5 features per tree):
  test accuracy:  0.7471

test accuracy as the ensemble grows (same fitted trees):
  trees   bagging  rf-style
      1    0.8107    0.7539
      5    0.8326    0.7509
     10    0.8386    0.7484
     25    0.8409    0.7471
```

Things to notice:

- The single deep tree scores 0.9721 on data it has seen and 0.8149 on data it hasn't —
  the textbook overfitting gap. (Train accuracy isn't 1.0 only because some rows have
  identical features but different labels, so no tree can memorize them all.)
- Bagging the *same kind* of overfit tree 25 times lifts test accuracy to 0.8409 (+2.6
  points), and the growth curve shows most of the gain arriving by ~10 trees — the
  $\frac{1-\rho}{B}\sigma^2$ term shrinking.
- The per-tree feature subsampling **hurts** here (0.7471): with 5 of 35 features, each
  tree has only a $5/35 = 1/7$ chance of seeing any given feature, so most trees never see
  strong predictors like `capital-gain` or `marital-status=Married-civ-spouse` at all.
  Decorrelation was bought with a large bias increase, and more trees can't fix bias —
  the rf-style curve actually drifts *down* as voting converges to the biased average.
  This is exactly why real random forests resample the feature subset **per split**, not
  per tree: every tree still gets to use every important feature somewhere.
- `bagged` and `forest` are trained on exactly the same 25 bootstrap samples (same `SEED +
  i` per tree) — the only thing that differs between the two accuracy numbers is whether
  `max_features` restricted the columns, isolating that one variable so the comparison is
  fair.

## With a library

Read [`with_library.py`](../src/lesson08_ensembles/with_library.py) — identical
preprocessing and split, then sklearn's two workhorse ensembles.

### Step-by-step: reading the code

`load_data` here is the same one-hot logic explained above, copied rather than imported so
this script has no hidden dependency on `from_scratch.py` — both scripts build the identical
`X`. The interesting change is the import line and everything after the split:

```python
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
```

Both classes replace hand-written ensemble code: `fit_ensemble` and `vote` from
`from_scratch.py` collapse into a single `.fit()` call each.

```python
forest = RandomForestClassifier(n_estimators=100, random_state=SEED)
forest.fit(X[train], y[train])
```

`n_estimators=100` is $B$ from the Concept section — four times as many trees as the
from-scratch ensemble. `RandomForestClassifier` performs the same two decorrelation moves as
`fit_ensemble` (bootstrap row sampling, then feature subsampling) automatically, but it draws
the candidate feature subset **at every split**, not once per tree — the fix for the bias
problem the from-scratch rf-style ensemble ran into. `random_state=SEED` seeds sklearn's
internal bootstrap and split randomness, playing the same reproducibility role `SEED + i`
and `random_state=i` played in `fit_ensemble`.

```python
boost = HistGradientBoostingClassifier(random_state=SEED)
boost.fit(X[train], y[train])
```

This is the Concept section's **boosting**, not bagging: no bootstrap sampling and no
voting happen here at all. Internally, `HistGradientBoostingClassifier` fits shallow trees
one after another, each targeting the errors the ensemble so far still makes, and sums their
weighted outputs — a sequential, bias-reducing mechanism rather than the parallel,
variance-reducing one built by hand above.

```python
order = sorted(range(len(names)), key=lambda j: (-forest.feature_importances_[j], j))
for j in order[:5]:
    print(f"  {names[j]:<34} {forest.feature_importances_[j]:.4f}")
```

`forest.feature_importances_` is an array with one impurity-based importance per feature
column, summing to 1 across all 35. The `sorted(..., key=lambda j: (-importance, j))` call
ranks column indices by importance descending (the leading `-`), breaking ties by column
index; printing the first five gives the "top-5" table in the Expected output below. These
are the same impurity importances flagged in the Concept section's caveats paragraph — read
the ranking as a hint, not a definitive answer.

```bash
python src/lesson08_ensembles/with_library.py
```

Expected output:

```text
== Ensembles: with scikit-learn ==
rows in file: 32561
rows after dropping '?': 30162
features: 5 numeric + 30 one-hot = 35
train/test split: 24129/6033 (seed 42)

random forest (100 trees):
  test accuracy:  0.8381
hist gradient boosting:
  test accuracy:  0.8705

top-5 feature importances (random forest):
  age                                0.2512
  education-num                      0.1450
  hours-per-week                     0.1221
  capital-gain                       0.1175
  marital-status=Married-civ-spouse  0.1082
```

Comparing all the ensembles: our 25-tree scratch **bagging** (0.8409) and sklearn's
100-tree **random forest** (0.8381) land within 0.3 points of each other — both are
variance-reduction machines built on independent bootstrapped trees, and per-split feature
subsampling lets sklearn's forest decorrelate without the bias catastrophe our per-tree
version suffered. **Boosting** (0.8705) beats both by ~3 points, and the reason is the
mechanism: bagging trains every tree independently on the same task and averages away
their noise, which does nothing about what all the trees systematically get wrong;
boosting trains each new (shallow) tree *on the residual errors of the ensemble so far*,
so it attacks bias sequentially, one correction at a time. On tabular data like this,
gradient boosting is usually the accuracy king, with random forests close behind at a
fraction of the tuning effort.

On the importances: `age` outranking `capital-gain` should raise an eyebrow — impurity
importance favors `age` partly because a continuous feature offers vastly more split
points than a one-hot column, one of the caveats from the Concept section.

## Check your understanding

1. In a bootstrap sample of size $n$, why does a given row appear with probability
   $1 - (1 - 1/n)^n \approx 63\%$? What could you do with the ~37% of rows a tree never saw?
2. Using $\mathrm{Var}(\bar{X}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$: what is the
   variance of a 25-tree ensemble when $\rho = 1$? When $\rho = 0$? Which knob does feature
   subsampling turn?
3. Our rf-style ensemble got *worse* than a single bagged tree ensemble. Explain the
   bias/variance trade it made, and why sklearn's per-split subsampling avoids it.
4. Why does bagging use deep (unpruned) trees while boosting uses shallow ones?
5. Would bagging help a linear regression model as much as it helps a deep tree? Why not?
   (Hint: what kind of error does averaging remove?)
6. In `fit_ensemble`, row indices are drawn with `rng.integers(0, n, size=n)` while feature
   indices are drawn with `rng.choice(d, size=max_features, replace=False)`. Why does one of
   these sampling steps allow repeats and the other forbid them?
7. The growth-curve loop reports ensembles of size 1, 5, 10, and 25. For which of these
   sizes could `vote()` actually produce an exact tie, and what would it predict if it did?

From here, you're now ready to proceed to try and reproduce
[Lesson 09 — Clustering](09-clustering.md).

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
