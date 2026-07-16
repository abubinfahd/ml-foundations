# Lesson 06 — Model Evaluation

You have trained models in lessons 04 and 05; now you learn to judge them properly. One
accuracy number from one split is a noisy, sometimes misleading verdict. This lesson
implements the standard evaluation toolkit from scratch — k-fold cross-validation, the
confusion matrix, precision/recall/F1, and ROC-AUC — around a model you already know.
To be clear: **the model here is scikit-learn's `LogisticRegression`** (on lesson 05's
Titanic features) in *both* scripts. What you implement from scratch this time is the
*evaluation*, not the model.

## Concept

**Why one split isn't enough.** A single 80/20 split evaluates on 179 passengers. Change
which rows land in the test set and accuracy easily moves by several points — same model,
different verdict. The split is a random variable; one draw of it is a noisy estimate.

Concretely: an "80/20 split" means 80% of the rows are randomly chosen to train the
model, and the remaining 20% are held out to test it. Swap which rows fall into that
20% and the reported accuracy shifts — not because the model changed, but because the
test set did. k-fold cross-validation, next, is the standard fix for that noise.

**k-fold cross-validation.** Shuffle once, cut the data into $k$ equal folds, and take
turns: train on $k-1$ folds, evaluate on the held-out one. Every row is tested exactly
once, and you get $k$ scores instead of one — report their mean and spread. The spread
is the point: it tells you how much any *single* split's number is to be trusted. One
rule carries over from lessons 04–05: imputation and standardization must be *refit
inside each fold* on its training part, or fold statistics leak into evaluation.

In plain terms: $k$ is just the number of pieces the shuffled data is cut into — this
lesson uses $k = 5$, so the 891 rows become 5 roughly equal "folds." Each fold takes one
turn as the test set while the other four are concatenated into that round's training
set; after 5 rounds, every row has served once as test data and four times as training
data. "Fold statistics leak into evaluation" describes a specific mistake: if you
computed, say, the median passenger age from all 891 rows and used that one number to
fill missing ages *before* splitting into folds, every fold's held-out rows would have
quietly influenced the number used to patch their own missing values. That is a small
but real leak of test information into training, and it makes the reported score too
optimistic. The fix — recompute the median (and the mean/std used for standardization)
from *only* that fold's training rows, every fold — is exactly what the `preprocess`
function in `from_scratch.py` does on every call; see the step-by-step below.

**Confusion matrix.** For binary predictions, four counts say everything:

| | predicted 0 | predicted 1 |
|---|---|---|
| **actual 0** | TN (true negative) | FP (false positive) |
| **actual 1** | FN (false negative) | TP (true positive) |

Every metric below is arithmetic on these four cells.

In words: TN and TP are the two ways a prediction can be *right* — predicting "died"
for someone who died, or "survived" for someone who survived. FP and FN are the two
ways it can be *wrong* — FP is a false alarm (predicted "survived", actually died), FN
is a miss (predicted "died", actually survived). Accuracy itself is nothing more than
$(TP + TN) / (TP + TN + FP + FN)$: the fraction of all predictions sitting on the
table's diagonal.

### A hand-computable example

Before trusting the code, compute one of these by hand. Say a tiny model makes 10
predictions:

| actual | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| predicted | 1 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |

Walk the 10 columns one at a time and sort each into a cell: columns 1, 3, 8, 10 are
actual=1/predicted=1 (TP, 4 of them); column 2 is actual=1/predicted=0 (FN, 1); column 5
is actual=0/predicted=1 (FP, 1); columns 4, 6, 7, 9 are actual=0/predicted=0 (TN, 4).
That sorts into:

| | predicted 0 | predicted 1 |
|---|---|---|
| **actual 0** | TN = 4 | FP = 1 |
| **actual 1** | FN = 1 | TP = 4 |

From here: accuracy $= (4+4)/10 = 0.8$, precision $= 4/(4+1) = 0.8$,
recall $= 4/(4+1) = 0.8$, F1 $= 0.8$. Redo this same arithmetic later on the real
fixed-split confusion matrix in the Expected output below (84, 26, 20, 49) before
reading on — matching your own hand arithmetic to the script's printed
precision/recall/F1 is the fastest way to confirm you understand the formulas rather
than just recognize them.

**Precision vs recall.**

$$\text{precision} = \frac{TP}{TP + FP} \qquad \text{recall} = \frac{TP}{TP + FN}$$

Precision: of the passengers the model *claims* survived, how many really did? Recall:
of the passengers who *really* survived, how many did the model find? They pull against
each other. Imagine a rescue coordinator using the model to decide where to search for
survivors: missing a real survivor (FN) is far costlier than a wasted search (FP), so
they would lower the 0.5 threshold and buy recall at the price of precision. An
archivist auto-tagging biographies "survivor" would do the opposite. The threshold is a
policy decision; the metrics quantify the trade.

Notice the denominators differ: precision divides by $TP + FP$, which is *everyone the
model predicted positive* (right and wrong positive predictions); recall divides by
$TP + FN$, which is *everyone who is actually positive* (found and missed positives).
Same numerator, two different universes to compare it against — that difference in
denominator is the entire reason the two numbers can disagree.

**F1.** The harmonic mean $F_1 = \frac{2 \cdot P \cdot R}{P + R}$ — unlike the
arithmetic mean, it is dragged toward the *worse* of the two, so you cannot score well
by maxing one and ignoring the other.

For intuition, take an extreme case: precision $=1.0$, recall $=0.02$ (the model is
right every time it dares predict positive, but it almost never dares). The arithmetic
mean $(1.0+0.02)/2 = 0.51$ looks respectable; the harmonic mean
$F_1 = 2(1.0)(0.02)/(1.0+0.02) \approx 0.039$ correctly reports that this model is
nearly useless.

**Why not just accuracy?** Accuracy rewards betting on the majority class. In our test
split, 110 of 179 passengers died: the constant model "everybody dies" scores 0.6145
with zero insight. On a 99%-negative dataset (fraud, rare disease) a useless model
scores 99%. Always ask what the majority-class baseline gets.

The general habit worth building: before trusting any accuracy number, compute the
accuracy of the simplest possible baseline — always predict the majority class — and
treat your model's score as meaningless unless it clears that bar by a comfortable
margin. Precision, recall, and F1 are the tools that keep working when that bar is 99%
and accuracy alone would hide a model that never finds the minority class at all.

**ROC and AUC.** Instead of fixing the threshold at 0.5, sweep it from high to low. At
each threshold, plot the *true positive rate* (recall) against the *false positive rate*
$FPR = FP/(FP+TN)$. That trace is the ROC curve; the area under it (by trapezoidal
integration) is the AUC. AUC has a beautiful probabilistic meaning:

$$\text{AUC} = P\big(\text{score(random positive)} > \text{score(random negative)}\big)$$

(ties counted half) — which is why it can also be computed with no curve at all, from
the ranks of the scores (the Mann–Whitney U statistic). The script computes it both
ways and they must agree. AUC = 0.5 is coin-flipping, 1.0 is perfect ranking, and it is
unchanged by any monotonic rescaling of the scores — it evaluates the *ranking*, not
the calibration.

To unpack that: a "threshold" is the cutoff probability at which a prediction flips
from "died" to "survived" — predict survived when the model's output probability
$p \geq \text{threshold}$. At threshold $\approx 1$, almost nobody clears the bar, so
both TPR and FPR start near 0 (the bottom-left corner of the curve). As the threshold
sweeps down toward 0, more and more passengers get predicted "survived": TPR and FPR
both climb toward 1 (the top-right corner). A perfect model reaches TPR = 1 while FPR
is still 0 — it hugs the top-left corner before rising; a coin-flip model walks the
diagonal, contributing an AUC of 0.5. The rank-based version restates the same area as
a counting problem: line up every test-set score, and ask, of all (positive, negative)
pairs you could draw, in what fraction does the positive actually get the higher
score? That fraction *is* the AUC, no curve-drawing required — which is exactly why
`from_scratch.py` can compute it two structurally different ways and check that they
agree to four decimal places.

## Dataset

Titanic again — you downloaded it in lesson 05, so the script now verifies the checksum
and skips the download:

```powershell
.\scripts\download_data.ps1 titanic
```

Expected output:

```text
[skip] titanic.csv (already downloaded, checksum OK)
Done.
```

Features and preprocessing are exactly lesson 05's: `Pclass`, `Sex` (male=1), `Age`
(train-median imputed), `SibSp`, `Parch`, `Fare`; target `Survived`.

## From scratch

Read [`from_scratch.py`](../src/lesson06_model_evaluation/from_scratch.py) first. The
only scikit-learn import is the model itself; the folds, the confusion matrix, and both
AUC computations are plain NumPy.

```bash
python src/lesson06_model_evaluation/from_scratch.py
```

Expected output:

```text
== Model evaluation: from scratch ==
rows: 891

-- 5-fold cross-validation (manual folds) --
fold 1  n=179  accuracy: 0.8045
fold 2  n=178  accuracy: 0.8090
fold 3  n=178  accuracy: 0.8034
fold 4  n=178  accuracy: 0.8090
fold 5  n=178  accuracy: 0.7472
mean accuracy: 0.7946 +/- 0.0238

-- Fixed 80/20 split (lesson 05's): train 712, test 179 --

confusion matrix (rows = actual, columns = predicted):
                    pred died  pred survived
actual died                84             26
actual survived            20             49

accuracy : 0.7430
precision: 0.6533
recall   : 0.7101
F1       : 0.6806

ROC curve points (unique thresholds + origin): 169
AUC via threshold sweep + trapezoid: 0.8234
AUC via Mann-Whitney rank formula  : 0.8234
```

Things to notice:

- Fold accuracies range from 0.7472 to 0.8090 — a 6-point spread from nothing but *which
  rows landed where*. This is exactly why a single-split number deserves suspicion, and
  our fixed split's 0.7430 sits at the unlucky end of that range; reading that one
  number alone, without the fold spread around it, would have been misleading.
- Precision 0.6533 = 49/(49+26) and recall 0.7101 = 49/(49+20) — read them straight off
  the matrix, and F1 = 2(0.6533)(0.7101)/(0.6533+0.7101) = 0.6806 follows the same way,
  the harmonic mean pulling toward whichever of the two is smaller.
- The two AUC computations agree to 4 decimals despite sharing no code: one integrates a
  curve, the other counts rank inversions. Same quantity, two derivations — a strong
  argument for always keeping a cross-check like this when one is cheap to compute.
- 169 curve points for 179 test rows: some passengers share identical features, hence
  identical predicted probabilities — tied scores must enter the curve together (and
  count half in the rank formula), which is what the tie-handling code is for.

### Step-by-step: reading the code

Read the functions in the order they appear in the file — that is also the order in
which `main()` calls them.

**`load_data`** turns the CSV into the two NumPy arrays every other function expects:

```python
def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Same features as lesson 05: Pclass, Sex (male=1), Age (NaN if
    missing), SibSp, Parch, Fare. Target: Survived."""
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

`csv.DictReader` reads each row as a dict keyed by column name, so `r["Pclass"]` reads
the passenger class as a string; wrapping it in `float(...)` converts it to a number
NumPy can work with. `Sex` is text ("male"/"female"), so it is recoded to 1.0/0.0 right
here — logistic regression needs numbers, not strings. `Age` is the one column with
missing values in this dataset: an empty string becomes `np.nan` (NumPy's "not a
number" placeholder) instead of crashing on `float("")`. The whole list of per-row
lists is handed to `np.array(...)` once, producing a single `(891, 6)` matrix `X` — one
row per passenger, one column per feature, in the order given by `FEATURES`. `y` is a
separate `(891,)` array of 0.0/1.0 labels. Nothing here is *fit* to the data yet — this
function only loads; the next one is where fold-by-fold fitting starts.

**`preprocess`** is where the no-leakage rule from the Concept section is enforced in
code:

```python
def preprocess(X_train: np.ndarray, X_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lesson 05's preprocessing: impute Age with the TRAIN median, then
    standardize with TRAIN mean/std. Done inside every fold, so no fold ever
    leaks statistics into the model evaluated on it."""
    X_train, X_eval = X_train.copy(), X_eval.copy()
    age_median = float(np.nanmedian(X_train[:, 2]))
    for split in (X_train, X_eval):
        split[np.isnan(split[:, 2]), 2] = age_median
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_eval - mu) / sigma
```

`.copy()` matters first: without it, the in-place assignment two lines down would
silently overwrite the caller's original array. `np.nanmedian` computes the median
while ignoring NaNs — the median age of the *training* rows only, never touching
`X_eval`. `np.isnan(split[:, 2])` is a boolean mask over column 2 (Age); indexing an
array with a boolean mask of the same length selects only the `True` positions, so
`split[mask, 2] = age_median` fills in exactly the missing ages and nothing else. `mu`
and `sigma` (mean and standard deviation, one value per column, via `axis=0`) are
likewise computed from `X_train` alone, then applied to *both* splits — standardizing
`X_eval` with statistics that never saw `X_eval`. This function is called once per fold
inside the cross-validation loop, and once more for the fixed 80/20 split, so the rule
is applied uniformly everywhere a train/test boundary exists.

**`fit_model`** is intentionally the shortest function in the file:

```python
def fit_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    return LogisticRegression(max_iter=1000).fit(X_train, y_train)
```

This is the one piece of the script that is *not* from scratch, by design — the lesson
reuses lesson 05's model so everything else in the file can be judged against a model
you already understand, isolating what is new here to the evaluation code.

**`roc_auc_trapezoid`** builds the ROC curve described in the Concept section and
integrates under it:

```python
def roc_auc_trapezoid(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    order = np.argsort(-p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    tp = np.cumsum(y_sorted)  # positives captured at each cut
    fp = np.cumsum(1 - y_sorted)  # negatives wrongly captured
    last_of_tie = np.r_[np.nonzero(np.diff(p_sorted))[0], len(p_sorted) - 1]
    tpr = np.r_[0.0, tp[last_of_tie] / n_pos]
    fpr = np.r_[0.0, fp[last_of_tie] / n_neg]
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return auc, len(tpr)
```

`np.argsort(-p, kind="stable")` sorts by *descending* probability (negating first,
since `argsort` only sorts ascending) — this is "sweeping the threshold from high to
low" made concrete: row 0 after sorting is the passenger the model is most confident
survived. `kind="stable"` keeps tied rows in their original relative order, which
matters for reproducibility but not for the math. `np.cumsum(y_sorted)` is a running
total: its $i$-th entry is how many true positives you would have if you drew the
threshold just below the $i$-th sorted score — i.e., predicted "survived" for exactly
the top $i$ scores. `np.cumsum(1 - y_sorted)` does the same for false positives. The
`last_of_tie` line finds, for every group of equal probabilities, the index of the
*last* row in that group (`np.diff(p_sorted)` is zero within a tied block and nonzero
where the value changes, so `np.nonzero(np.diff(...))` finds the boundaries; `np.r_`
appends the final index, since the last tie group has no following boundary). Cutting
the curve only at those indices — rather than after every single row — is what makes
tied scores enter the curve together, as the Concept section's tie-handling promised.
`tpr` and `fpr` prepend a `0.0` (the curve always starts at the origin), then divide
the cumulative counts by the total positives/negatives to turn raw counts into rates.
The final loop is the trapezoidal rule itself: for each consecutive pair of curve
points, the area of the trapezoid between them is the horizontal step
`fpr[i] - fpr[i-1]` times the average height `(tpr[i] + tpr[i-1]) / 2`, and summing
those areas across the whole curve gives the AUC.

**`roc_auc_rank`** computes the same AUC a completely different way, as a cross-check:

```python
def roc_auc_rank(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.empty(len(p))
    i = 0
    while i < len(p):  # assign midranks to ties
        j = i
        while j + 1 < len(p) and p_sorted[j + 1] == p_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average of ranks i+1..j+1
        i = j + 1
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

Here the sort is *ascending* (lowest probability first), because what's needed now is
each row's rank (1 = lowest score) rather than a threshold sweep. The `while` loop
walks the sorted array and, whenever it finds a run of equal probabilities from index
`i` to `j`, assigns every row in that run the *average* of the ranks it would have
occupied (`(i + j) / 2 + 1`, using 1-based ranks) — this is the "ties count half" rule
from the Concept section made concrete: two tied rows straddling a rank boundary each
get the mean rank rather than an arbitrary tiebreak. The final line is the
Mann–Whitney U formula: `ranks[y == 1].sum()` adds up the ranks of all the actual
positives; subtracting $n_{pos}(n_{pos}+1)/2$ (the minimum possible sum of $n_{pos}$
ranks) leaves the number of positive-negative pairs where the positive outranked the
negative; dividing by $n_{pos} \cdot n_{neg}$ (the count of all such pairs) turns that
count into the probability from the Concept section's AUC formula. That it lands on
the same 4-decimal answer as `roc_auc_trapezoid` — despite sharing no code — is the
point of computing it twice.

**The driver, `main()`,** ties the functions together in three stages.

Stage 1 runs manual k-fold cross-validation:

```python
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    folds = np.array_split(perm, K)
    ...
    for k, fold in enumerate(folds):
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])
        X_tr, X_te = preprocess(X[train_idx], X[fold])
        model = fit_model(X_tr, y[train_idx])
        acc = float(np.mean(model.predict(X_te) == y[fold]))
```

`np.random.default_rng(SEED)` creates a seeded random generator — same seed, same
shuffle, every run, which is why the Expected output below is reproducible.
`rng.permutation(n)` shuffles the row indices 0..890 once; `np.array_split(perm, K)`
cuts that shuffled list into `K` (5) pieces — the "cut into $k$ equal folds" step from
the Concept section (`array_split` handles 891 not dividing evenly by 5 by making some
folds one row larger). The loop then takes each fold in turn as the test set (`fold`),
concatenates *all other* folds into that round's training set (`train_idx`), and calls
`preprocess` and `fit_model` on that pairing — exactly the "train on $k-1$ folds,
evaluate on the held-out one" procedure. `model.predict(X_te) == y[fold]` produces an
array of `True`/`False`; NumPy treats those as 1/0 under `np.mean`, so the mean is
simply the fraction of correct predictions — accuracy.

Stage 2 fixes one 80/20 split (matching lesson 05's) and computes the confusion matrix
cells directly from boolean masks:

```python
    tp = int(np.sum((pred == 1) & (y_test == 1)))
    fp = int(np.sum((pred == 1) & (y_test == 0)))
    fn = int(np.sum((pred == 0) & (y_test == 1)))
    tn = int(np.sum((pred == 0) & (y_test == 0)))
```

Each line is a direct translation of one cell of the Concept section's table into
code: `&` combines two boolean arrays element-wise (not Python's `and`, which does not
work on arrays), so `(pred == 1) & (y_test == 1)` is `True` exactly at the rows that
are true positives; `np.sum` over a boolean array counts the `True`s. Precision,
recall, and F1 follow as the exact formulas from the Concept section, computed
directly from `tp`, `fp`, `fn`:

```python
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
```

Stage 3 hands the fixed split's predicted *probabilities* (not just the 0/1
predictions) to both AUC functions:

```python
    auc_trap, n_points = roc_auc_trapezoid(y_test, prob)
    auc_rank = roc_auc_rank(y_test, prob)
```

`prob` (set earlier in `main` from `model.predict_proba(X_te)[:, 1]`) is the model's
predicted probability of survival for each test passenger — the continuous score that
ROC/AUC sweeps a threshold over, as opposed to `pred`, which is already the
thresholded 0/1 decision at the default 0.5 cutoff. Printing `auc_trap` and `auc_rank`
side by side is what lets the Expected output demonstrate that two independently
derived computations agree to four decimal places.

## With a library

Read [`with_library.py`](../src/lesson06_model_evaluation/with_library.py). Cross-
validation uses `KFold(n_splits=5, shuffle=True, random_state=42)` with
`cross_val_score`, and a `Pipeline` so the imputer and scaler are refit per fold —
the same no-leakage discipline as the manual loop, automated. Note: `KFold`'s internal
shuffle assigns rows to folds differently than our manual permutation, so the *per-fold*
numbers legitimately differ from the from-scratch run; the means should be (and are)
close. The fixed-split metrics, by contrast, are computed from identical predictions and
identical definitions, so they must match from-scratch *exactly*.

```bash
python src/lesson06_model_evaluation/with_library.py
```

Expected output:

```text
== Model evaluation: with scikit-learn ==
rows: 891

-- 5-fold cross-validation (KFold, shuffle=True, random_state=42) --
fold 1  accuracy: 0.7989
fold 2  accuracy: 0.7809
fold 3  accuracy: 0.8258
fold 4  accuracy: 0.7697
fold 5  accuracy: 0.7753
mean accuracy: 0.7901 +/- 0.0204

-- Fixed 80/20 split (lesson 05's): train 712, test 179 --

confusion matrix (rows = actual, columns = predicted):
                    pred died  pred survived
actual died                84             26
actual survived            20             49

accuracy : 0.7430
precision: 0.6533
recall   : 0.7101
F1       : 0.6806

ROC-AUC  : 0.8234
```

The CV means (0.7946 manual vs 0.7901 KFold) differ by half a point — two equally valid
5-fold estimates of the same quantity, which is itself a lesson in how much resolution
these numbers really have. Everything below the fold section matches from-scratch
character for character.

### Step-by-step: reading the code

`with_library.py` follows the same three stages as `from_scratch.py`, but delegates
each computation to scikit-learn (and pandas for loading):

```python
    df = pd.read_csv(DATA)
    df["Sex"] = (df["Sex"] == "male").astype(float)  # male=1, female=0
    X = df[FEATURES].to_numpy(dtype=float)  # Age keeps NaN where missing
    y = df["Survived"].to_numpy(dtype=float)
```

`pd.read_csv` replaces the manual `csv.DictReader` loop; pandas already reads missing
`Age` cells as `NaN`, so there is no explicit `if r["Age"] != ""` check here.
`(df["Sex"] == "male").astype(float)` is pandas' vectorized equivalent of the
from-scratch `1.0 if r["Sex"] == "male" else 0.0` list comprehension — same recoding,
applied to the whole column at once.

Cross-validation replaces the manual permutation-and-loop with two library calls:

```python
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    cv = KFold(n_splits=K, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=cv)
```

`SimpleImputer(strategy="median")` is scikit-learn's version of the from-scratch
`age_median` fill; `StandardScaler` is the mean/std standardization from
`preprocess`. Chaining them with `LogisticRegression` inside `make_pipeline`
guarantees the same no-leakage rule from the Concept section: `cross_val_score`
refits the *entire* pipeline — imputer, scaler, and model — from scratch on each
fold's training data alone, exactly as `preprocess` and `fit_model` are called fresh
inside the manual loop. `KFold(n_splits=5, shuffle=True, random_state=42)` plays the
same role as `rng.permutation(n)` plus `np.array_split(perm, K)` — it just uses its
own internal shuffle, which is why the *per-fold* numbers differ from the manual
version even though both are valid 5-fold splits of the same data.

The fixed 80/20 split's confusion matrix and metrics come from `sklearn.metrics`:

```python
    (tn, fp), (fn, tp) = confusion_matrix(y_test, pred)
    ...
    print(f"accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"precision: {precision_score(y_test, pred):.4f}")
    print(f"recall   : {recall_score(y_test, pred):.4f}")
    print(f"F1       : {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, prob):.4f}")
```

`confusion_matrix` returns the same four counts `from_scratch.py` computes by hand
with boolean masks, already arranged as a 2x2 array — unpacking it as
`(tn, fp), (fn, tp)` relies on scikit-learn's fixed row/column order (actual rows,
predicted columns, class 0 before class 1), matching the Concept section's table.
`accuracy_score`, `precision_score`, `recall_score`, and `f1_score` are direct
implementations of this lesson's formulas; `roc_auc_score` replaces *both*
`roc_auc_trapezoid` and `roc_auc_rank` with one call — internally it implements the
same rank-based computation, which is why its output matches the from-scratch value
exactly rather than merely approximately.

## Check your understanding

1. The test split has 110 non-survivors out of 179. Verify the 0.6145 accuracy of the
   "everybody dies" model. Our model scores 0.7430 — how impressed should you be, and
   which other printed metric best captures what the constant model lacks?
2. Recompute precision and recall from the confusion matrix by hand. If you raised the
   decision threshold from 0.5 to 0.8, which of the two would rise and which would fall?
   Trace the answer through the FP and FN cells.
3. The five fold accuracies have std 0.0238. Given that, was our fixed split's 0.7430 a
   surprising result? What does this say about comparing two models that differ by one
   accuracy point on a single split?
4. State AUC's probabilistic interpretation in one sentence about two randomly chosen
   Titanic passengers. Why does replacing every probability $p$ with $p^3$ leave AUC
   unchanged but potentially change accuracy at threshold 0.5?
5. Suppose you standardized all 891 rows *once* before cross-validation. Nothing crashes
   and the numbers barely move. What, precisely, is still wrong?
6. `preprocess` in `from_scratch.py` is called separately inside every fold and again
   for the fixed split, rather than once on the whole dataset. Tie this back to the
   "fold statistics leak into evaluation" warning in the Concept section: what number,
   specifically, would be computed differently — and from which rows — if `preprocess`
   were instead called once before any splitting happened?
7. In `roc_auc_trapezoid`, the code cuts the ROC curve only at `last_of_tie` indices
   instead of after every single sorted row. Using the 169-vs-179 fact from "Things to
   notice," explain in your own words what would go wrong with the curve — and with the
   rank formula's "ties counted half" rule — if tied probabilities were instead treated
   as if they were in some arbitrary order.

From here, you're now ready to proceed to try and reproduce
[Lesson 07 — Decision Trees](07-decision-trees.md).

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
