# Lesson 07 — Decision Trees

Linear models draw one global boundary; decision trees instead ask a sequence of simple
yes/no questions ("is petal length ≤ 2.45 cm?") and partition the feature space into
boxes. In this lesson you implement CART for classification — Gini impurity, exhaustive
split search, greedy recursion — and then check that scikit-learn grows the very same tree.

## Concept

**What a decision tree is.** Before the formulas: a decision tree is a tree-shaped stack of
yes/no questions. You start at the root, answer the question stored there, follow the
branch that matches your answer down to a child node, answer *its* question, and so on
until you land on a **leaf** — a node with no further question, only a stored prediction.
Building the tree means choosing, at each node, the question that best separates the
classes; predicting for a new sample means walking the tree from the root, following
whichever branch each answer points to, until a leaf is reached.

**Recursive partitioning.** A decision tree splits the data on one feature and one
threshold, then splits each half again, recursively. Every internal node is a question
$x_j \le t$, where $x_j$ is the value of feature $j$ (a column of the data — e.g. feature 2
is petal length) for one sample, and $t$ is a threshold: a cutoff value chosen from the
training data. Every leaf predicts the majority class of the training samples that reached
it — whichever class has the most training samples that would have answered every question
on the path from the root to that leaf the same way this new sample does.

**Gini impurity.** To pick a good question we need to score how "mixed" a node is, i.e. how
thoroughly it separates the classes among the samples that reach it. Say a node's samples
belong to $K$ possible classes; $p_k$ is the *proportion* of the node's samples belonging to
class $k$ (so $\sum_k p_k = 1$ — the proportions across all classes add up to the whole
node). The Gini impurity is

$$G = 1 - \sum_{k} p_k^2$$

A pure node ($p_k = 1$ for one class, 0 for the rest) has $G = 0$: no mixing, no
uncertainty about which class a sample belongs to. For three perfectly mixed classes
$G = 1 - 3\cdot(1/3)^2 = 2/3$, the largest $G$ can get with three classes. Intuition: $G$ is
the probability that two samples drawn at random from the node (independently, with
replacement) belong to different classes — the more mixed the node, the more likely two
random draws disagree, the higher $G$. Minimizing $G$ in the children is exactly what a good
split should do.

**Worked example by hand.** Suppose a node holds 8 samples: 6 of class A and 2 of class B,
so $p_A = 6/8 = 0.75$ and $p_B = 2/8 = 0.25$, giving

$$G_{\text{parent}} = 1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 0.375$$

Now consider a candidate split that sends 5 samples left (all class A, so
$G_L = 1 - 1^2 = 0$, a pure child) and 3 samples right (1 A and 2 B, so $p_A = 1/3$,
$p_B = 2/3$, and $G_R = 1 - (1/9 + 4/9) = 4/9 \approx 0.4444$). Using the impurity-decrease
formula below with $n_L = 5$, $n_R = 3$, $n = 8$, the weighted child impurity is
$(5 \cdot 0 + 3 \cdot 0.4444) / 8 \approx 0.1667$, so this split's
$\Delta \approx 0.375 - 0.1667 = 0.2083$. Work through the arithmetic yourself with a
calculator — `gini` and `best_split` in `from_scratch.py` do exactly this, just repeated
over every feature and every candidate threshold instead of one hand-picked split.

**Impurity decrease of a split.** A split sends $n_L$ samples to the left child and $n_R$
samples to the right child, with $n_L + n_R$ equal to the number of samples at the parent
node. Its quality is how much impurity it removes, weighted by how many samples land in
each child (purifying a large child matters more than purifying a tiny one):

$$\Delta = G_{\text{parent}} - \frac{n_L\,G_L + n_R\,G_R}{n_L + n_R}$$

Here $G_{\text{parent}}$ is the impurity before splitting, and $G_L$, $G_R$ are the
impurities of the left and right children after the split. CART tries every feature and
every threshold — it is enough to test the midpoints between consecutive sorted unique
values, since any threshold between the same two values produces the same partition — and
keeps the split with the largest $\Delta$, i.e. the split that reduces mixing the most.

**Greedy top-down induction.** The best split is chosen locally at each node, with no
lookahead: the algorithm never asks "would a worse split now enable a better split later?"
A globally optimal tree is NP-hard to find, so nobody builds one — greedy induction is the
practical compromise, and it works well in practice because every accepted split is still
guaranteed to improve (never worsen) the impurity of the data it was given. Recursion stops
when a node is pure, when the depth cap is hit, or when no split improves impurity.

**Overfitting and depth.** An unlimited tree can carve one leaf per training sample and
memorize the data — a high-variance, low-bias model that fits noise as if it were signal
and predicts poorly on new data. Capping `max_depth` (here: 3) is the simplest regularizer:
fewer, larger leaves generalize better, because each leaf's majority-vote prediction rests
on more samples. Lesson 08 shows the other cure: keep the trees deep but average many of
them.

**Interpretability.** A depth-3 tree is one of the few models you can read out loud. The
printed tree below *is* the model — every prediction can be traced through at most three
questions. This transparency is why trees survive in domains where "the model said so" is
not an acceptable answer.

## Dataset

[UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) — 150 iris flowers, 4 measurements
each, 3 species (50 per species). License: CC BY 4.0.

```powershell
.\scripts\download_data.ps1 iris
```

Expected output:

```text
[get ] iris.data
[ ok ] iris.data (checksum verified)
Done.
```

Take a look at the data before modeling (always do this):

```bash
head -3 data/iris.data
```

```text
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
```

Note: the file has **no header row** (the columns are sepal_length, sepal_width,
petal_length, petal_width, class) and ends with an empty line, which both scripts skip.

## From scratch

Read [`from_scratch.py`](../src/lesson07_decision_trees/from_scratch.py) first — NumPy plus
the standard library, ~150 lines. `gini` and `best_split` are the two formulas above;
`build` is the greedy recursion (`max_depth=3`, `min_samples_leaf=1`). Leaf ties are broken
by class-name order, and split ties by the first (lowest feature index, then lowest
threshold) candidate. The 80/20 split shuffles with `default_rng(42)`.

### Step-by-step: reading the code

The walkthrough below follows the file top to bottom, in the order the code actually runs
— read it side by side with the source file.

**Imports and constants.**

```python
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "iris.data"

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SEED = 42
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 1
```

Only `csv`, `pathlib`, and `numpy` are needed — no ML library at all. `DATA` climbs from
this file (`parents[2]`: file → `lesson07_decision_trees` → `src` → repo root) into
`data/iris.data`, so the script works regardless of your current directory. `FEATURES`
gives human-readable names for printing. `SEED` fixes the train/test shuffle so the run is
reproducible. `MAX_DEPTH` and `MIN_SAMPLES_LEAF` are the two stopping-rule knobs from the
"Overfitting and depth" discussion above.

**Loading the data.**

```python
def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, class_names). iris.data has no header and ends with an
    empty line, which we skip. y holds indices into the sorted class names."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.reader(f) if r]  # skip the trailing empty line
    classes = sorted({r[4] for r in rows})
    X = np.array([[float(v) for v in r[:4]] for r in rows])
    y = np.array([classes.index(r[4]) for r in rows])
    return X, y, classes
```

`csv.reader` parses each line into a list of strings; `if r` filters out the trailing empty
line (an empty list is falsy in Python). `classes` is the sorted, de-duplicated set of
species names — sorting makes the encoding deterministic: `Iris-setosa` always becomes
class 0, `Iris-versicolor` class 1, `Iris-virginica` class 2, matching the "classes (count
order)" line in Expected output. `X` is a 150×4 float matrix of the first four columns; `y`
replaces each label string with its index into `classes` — the same label-encoding
convention used elsewhere in this course.

**Gini impurity.**

```python
def gini(counts: np.ndarray) -> float:
    """Gini impurity G = 1 - sum_k p_k^2 from class counts."""
    p = counts / counts.sum()
    return 1.0 - float((p**2).sum())
```

This is the formula from Concept, applied directly. `counts` is a per-class tally, e.g.
`[37, 0, 0]`; dividing by `counts.sum()` turns counts into proportions — NumPy divides
every entry by the same scalar, computing all the $p_k$ at once. `(p**2).sum()` computes
$\sum_k p_k^2$, and `1.0 - ...` gives $G$. Note that `gini` takes *counts*, not raw labels
— this matters in `best_split` below, where child counts are derived by subtraction instead
of recounting.

**Finding the best split — setup.**

```python
def best_split(X: np.ndarray, y: np.ndarray, n_classes: int):
    """Return (decrease, feature, threshold) of the best split, or None.

    Candidate thresholds are the midpoints between consecutive sorted unique
    values of each feature. Ties in impurity decrease are broken by the first
    (lowest feature index, then lowest threshold) candidate examined.
    """
    n = len(y)
    parent_counts = np.bincount(y, minlength=n_classes)
    parent_gini = gini(parent_counts)
    best = None  # (decrease, feature, threshold)
```

`np.bincount(y, minlength=n_classes)` counts how many samples at this node belong to each
class — exactly the `counts` vector `gini` expects. `parent_gini` is $G_{\text{parent}}$
from the impurity-decrease formula. `best` starts as `None`, meaning "no split examined
yet"; it will hold the running-best $(\Delta, j, t)$ triple as candidates are tried.

**Finding the best split — the search.**

```python
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])  # sorted unique values
        for lo, hi in zip(values[:-1], values[1:]):
            thr = (lo + hi) / 2
            mask = X[:, f] <= thr
            n_left = int(mask.sum())
            if n_left < MIN_SAMPLES_LEAF or n - n_left < MIN_SAMPLES_LEAF:
                continue
            left = np.bincount(y[mask], minlength=n_classes)
            right = parent_counts - left
            child = (n_left * gini(left) + (n - n_left) * gini(right)) / n
            decrease = parent_gini - child
            if best is None or decrease > best[0]:  # strict > keeps the first tie
                best = (decrease, f, thr)
    return best
```

The outer loop walks every feature `f` (column index — this is $j$ in $x_j \le t$). For
that feature, `np.unique` gives its sorted distinct values, and pairing consecutive values
(`values[:-1]` with `values[1:]`) and averaging each pair (`thr = (lo + hi) / 2`) generates
exactly the midpoint thresholds described in Concept — a complete set of candidates, since
any threshold strictly between the same two values produces an identical partition.
`mask = X[:, f] <= thr` evaluates the question $x_j \le t$ for every training sample at
once (a vectorized boolean array); `n_left` and `n - n_left` are $n_L$ and $n_R$. The
`MIN_SAMPLES_LEAF` check skips candidates that would leave a child empty or under the
minimum size. `left = np.bincount(y[mask], ...)` counts classes among the samples that went
left; `right = parent_counts - left` gets the right child's counts by subtraction instead
of recounting, since left and right must add up to the parent. `child` is the weighted
child impurity — the fraction in $\Delta$'s formula — and `decrease` is $\Delta$ itself.
The comparison `decrease > best[0]` uses strict greater-than, not `>=`: because the loops
visit features from index 0 upward and, within a feature, thresholds from low to high, the
*first* candidate to reach the best decrease seen so far is kept and never displaced by a
later, merely-equal one. That is the "ties broken by lowest feature index, then lowest
threshold" rule named in the docstring, and it is the reason the "With a library" section
below can predict exactly which side of a tie scikit-learn or this code will land on.

**Building the tree — leaves and stopping.**

```python
def build(X: np.ndarray, y: np.ndarray, n_classes: int, depth: int) -> dict:
    """Recursively grow the tree (greedy, top-down)."""
    counts = np.bincount(y, minlength=n_classes)
    # np.argmax returns the first maximum, i.e. ties go to the class earliest
    # in sorted class-name order.
    leaf = {"leaf": True, "counts": counts, "pred": int(np.argmax(counts))}
    if depth == MAX_DEPTH or gini(counts) == 0.0 or len(y) < 2 * MIN_SAMPLES_LEAF:
        return leaf
    split = best_split(X, y, n_classes)
    if split is None or split[0] <= 0.0:
        return leaf
```

This is the first place the code represents a tree node explicitly — and it is just a
plain Python `dict`, not a special class. A leaf is a dict tagged `"leaf": True`, holding
the class counts and `pred`, the predicted class index: whichever class has the most
counts, chosen by `np.argmax` (which, on ties, returns the first maximum — the earliest
class in sorted order, consistent with `load_data`'s encoding). Before searching for a
split, `build` always prepares a `leaf` dict as a fallback, then checks three stopping
conditions: the depth cap has been reached, the node is already pure (`gini(counts) ==
0.0`, nothing left to separate), or there are too few samples to make two children of at
least `MIN_SAMPLES_LEAF` each. If none of those stop it, it calls `best_split`; if no split
was found, or the best one found has `decrease <= 0` (no split actually helps), it also
falls back to the leaf.

**Building the tree — the recursive step.**

```python
    _, f, thr = split
    mask = X[:, f] <= thr
    return {
        "leaf": False,
        "feature": f,
        "threshold": thr,
        "left": build(X[mask], y[mask], n_classes, depth + 1),
        "right": build(X[~mask], y[~mask], n_classes, depth + 1),
    }
```

This is recursion: `build` calls itself to construct the left and right subtrees, each on
the subset of rows that answered the question that way (`X[mask]`/`X[~mask]`, `y[mask]`/
`y[~mask]`), one level deeper (`depth + 1`). In plain terms: "the left child's whole subtree
is: run this exact same tree-building procedure again, but only on the rows that satisfied
`x_f <= thr`, one level deeper." Because `build` always either returns a leaf immediately or
recurses on a strictly smaller subset of the data, and because `MAX_DEPTH` bounds how many
times it can call itself along any path, the recursion is guaranteed to terminate. The
returned dict for an internal (non-leaf) node stores which feature and threshold to
question (`feature`, `threshold`) plus its two children (`left`, `right`) — each itself
either a leaf dict or another internal-node dict, built the same way. This nested structure
of dicts *is* the tree; reading it back out requires traversing it, which is what
`print_tree` and `predict_one` do next.

**Printing the tree.**

```python
def print_tree(node: dict, classes: list[str], indent: str = "") -> None:
    if node["leaf"]:
        counts = ", ".join(str(c) for c in node["counts"])
        print(f"{indent}|--- leaf: {classes[node['pred']]}  counts=[{counts}]")
        return
    name, thr = FEATURES[node["feature"]], node["threshold"]
    print(f"{indent}|--- {name} <= {thr:.4f}")
    print_tree(node["left"], classes, indent + "|   ")
    print(f"{indent}|--- {name} >  {thr:.4f}")
    print_tree(node["right"], classes, indent + "|   ")
```

Another recursive traversal, this time to print instead of build. The base case is a leaf:
print its class and counts, then `return` — stop recursing down this branch. Otherwise,
look up the feature's readable name, print the question with 4 decimal places, recurse into
the left subtree with one extra level of `"|   "` indent, print the mirror-image "greater
than" branch, and recurse into the right subtree the same way. The indentation deepens by
one `"|   "` per recursive call, which is exactly why the nesting in Expected output's
printed tree lines up with tree depth.

**Predicting one sample.**

```python
def predict_one(node: dict, x: np.ndarray) -> int:
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["pred"]
```

This walks a single root-to-leaf path for one sample `x`. Unlike `build` and `print_tree`,
it is written iteratively — a `while` loop that keeps reassigning `node` — rather than
recursively. Both styles are valid traversals, but `predict_one` only ever needs to follow
*one* child at each step, so a loop is enough; `build` and `print_tree` need to do work on
*both* children (build both subtrees, print both branches), which is naturally expressed as
recursion instead. At each iteration, `x[node["feature"]] <= node["threshold"]` evaluates
the same question as `best_split`'s `mask`, but for a single row instead of the whole
training set; `node` becomes whichever child matches. The loop stops as soon as
`node["leaf"]` is true, and the stored prediction is returned.

**Scoring accuracy.**

```python
def accuracy(node: dict, X: np.ndarray, y: np.ndarray) -> float:
    pred = np.array([predict_one(node, x) for x in X])
    return float((pred == y).mean())
```

Calls `predict_one` once per row to build an array of predicted class indices, compares it
elementwise against the true labels `y`, and takes the mean of the resulting boolean array
— the mean of `True`/`False` values is the *fraction* that are `True`, i.e. the accuracy.
This one function produces both the "train accuracy" and "test accuracy" lines in Expected
output, just called on different `(X, y)` pairs.

**The driver block.**

```python
def main() -> None:
    X, y, classes = load_data()

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    n_train = int(0.8 * len(y))
    train, test = perm[:n_train], perm[n_train:]

    print("== Decision tree (CART): from scratch ==")
    print(f"rows in file: {len(y)}")
    print(f"train/test split: {len(train)}/{len(test)} (seed {SEED})")
    print(f"classes (count order): {', '.join(classes)}")
    print()
```

`main` is what actually runs when you execute the script. It loads the data, then creates a
seeded random generator (`default_rng(SEED)`) and uses it to produce `perm`, a random
permutation of the row indices 0 to 149 — reproducible because the seed is fixed. The first
80% of `perm` (`n_train = 120`) becomes the training indices, the rest (30) the test
indices; that is why Expected output reports "train/test split: 120/30 (seed 42)", and why
running the script again with the same seed always reproduces the same split, tree, and
accuracies. The rest of this excerpt just prints the header lines that appear verbatim at
the top of Expected output.

```python
    tree = build(X[train], y[train], len(classes), depth=0)
    print_tree(tree, classes)
    print()
    print(f"train accuracy: {accuracy(tree, X[train], y[train]):.4f}")
    print(f"test accuracy:  {accuracy(tree, X[test], y[test]):.4f}")
```

`build` is called only on the training rows, starting at `depth=0` — this single call
triggers the whole recursion described above and returns the finished tree. `print_tree`
renders it. Then `accuracy` is called twice: once against the very data the tree was built
from (train accuracy) and once against rows the tree never saw while building (test
accuracy). The gap between the two numbers in Expected output (0.9833 vs 0.9667) is the
generalization gap from the "Overfitting and depth" discussion in Concept. Finally, the
`if __name__ == "__main__": main()` guard at the bottom of the file (not shown above) means
`main()` runs when the file is executed directly, as the command below does, but not if the
file is imported as a module.

```bash
python src/lesson07_decision_trees/from_scratch.py
```

Expected output:

```text
== Decision tree (CART): from scratch ==
rows in file: 150
train/test split: 120/30 (seed 42)
classes (count order): Iris-setosa, Iris-versicolor, Iris-virginica

|--- petal_length <= 2.4500
|   |--- leaf: Iris-setosa  counts=[37, 0, 0]
|--- petal_length >  2.4500
|   |--- petal_width <= 1.6500
|   |   |--- petal_length <= 4.9500
|   |   |   |--- leaf: Iris-versicolor  counts=[0, 42, 0]
|   |   |--- petal_length >  4.9500
|   |   |   |--- leaf: Iris-virginica  counts=[0, 1, 4]
|   |--- petal_width >  1.6500
|   |   |--- petal_length <= 4.8500
|   |   |   |--- leaf: Iris-virginica  counts=[0, 1, 2]
|   |   |--- petal_length >  4.8500
|   |   |   |--- leaf: Iris-virginica  counts=[0, 0, 33]

train accuracy: 0.9833
test accuracy:  0.9667
```

Things to notice:

- The root question alone (`petal_length <= 2.45`) isolates *all 37* setosas in the
  training set — one feature separates one species perfectly. In Gini terms, that single
  split drives the left child's impurity straight to $G = 0$, a pure leaf that needs no
  further splitting.
- The remaining two levels of depth are spent untangling versicolor from virginica, and the
  tree never touches the sepal features at all: petal measurements carry all the signal for
  this dataset.
- Train accuracy is 0.9833, not 1.0 — two training samples sit in leaves dominated by the
  other class (see the `counts=[0, 1, 4]` and `counts=[0, 1, 2]` leaves above: each has one
  minority-class sample outvoted by the majority). A deeper tree would chase those two
  samples down to their own leaves and overfit — exactly the "Overfitting and depth"
  trade-off from Concept.

## With a library

The same model, fit by scikit-learn. Read
[`with_library.py`](../src/lesson07_decision_trees/with_library.py) — it reuses the exact
same loader and split, then calls `DecisionTreeClassifier(max_depth=3, random_state=42)`
and prints the tree with `sklearn.tree.export_text`.

### Step-by-step: reading the code

`load_data` and the seeded shuffle in `main` are byte-for-byte identical to
`from_scratch.py` — same CSV parsing, same sorted class list, same label encoding, same
`default_rng(SEED)` permutation — which is what makes the two scripts' outputs directly
comparable. The only new code is the fit/print/score sequence.

```python
    clf = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    clf.fit(X[train], y[train])
```

`DecisionTreeClassifier` defaults to `criterion="gini"`, so internally it scores splits with
the same $G = 1 - \sum_k p_k^2$ formula as `gini` in `from_scratch.py`, searching the same
kind of threshold candidates over every feature — the library's compiled equivalent of
`best_split`. `max_depth=3` is the same recursion cap as `MAX_DEPTH`, and `random_state=SEED`
fixes sklearn's internal tie-breaking when candidate splits are equally good (see "The two
trees match node for node" below). `.fit` runs sklearn's own greedy top-down induction — the
library equivalent of calling `build(X[train], y[train], len(classes), depth=0)`.

```python
    print(export_text(clf, feature_names=FEATURES, class_names=classes, decimals=4), end="")
```

`export_text` walks the fitted tree and renders it with the same `|---` indentation style as
`print_tree`, rounded to 4 decimal places to match. It plays the same role as calling
`print_tree(tree, classes)` by hand.

```python
    print(f"train accuracy: {clf.score(X[train], y[train]):.4f}")
    print(f"test accuracy:  {clf.score(X[test], y[test]):.4f}")
```

`clf.score(X, y)` predicts every row (walking the fitted tree the same way `predict_one`
does) and returns the fraction that match `y` — the library equivalent of
`accuracy(tree, X, y)`.

```bash
python src/lesson07_decision_trees/with_library.py
```

Expected output:

```text
== Decision tree (CART): with scikit-learn ==
rows in file: 150
train/test split: 120/30 (seed 42)
classes (count order): Iris-setosa, Iris-versicolor, Iris-virginica

|--- petal_length <= 2.4500
|   |--- class: Iris-setosa
|--- petal_length >  2.4500
|   |--- petal_width <= 1.6500
|   |   |--- petal_length <= 4.9500
|   |   |   |--- class: Iris-versicolor
|   |   |--- petal_length >  4.9500
|   |   |   |--- class: Iris-virginica
|   |--- petal_width >  1.6500
|   |   |--- petal_length <= 4.8500
|   |   |   |--- class: Iris-virginica
|   |   |--- petal_length >  4.8500
|   |   |   |--- class: Iris-virginica

train accuracy: 0.9833
test accuracy:  0.9667
```

**The two trees match node for node**: same features, same thresholds (to 4 decimals), same
leaf classes, same train and test accuracy. Both implementations use the same criterion
(Gini) and the same candidate thresholds (midpoints between consecutive sorted unique
values) — but the exact match is partly luck. On this training set the root is actually
*tied*: `petal_width <= 0.80` isolates the same 37 setosas as `petal_length <= 2.45`, with
exactly the same impurity decrease (0.3203). Our rule breaks ties toward the lowest feature
index, so it picks petal_length; sklearn breaks them by the order in which its
`random_state`-shuffled search visits the features, and with `random_state=42` it happens
to land on petal_length too. Re-fit with `random_state=0` and sklearn roots the tree at
`petal_width <= 0.80` instead — a different but *equally good* tree (same impurity
decrease, same accuracy). Tie-breaking is part of the algorithm's specification, not a
detail. With the seeds fixed as given, though, your output must match the blocks above
exactly — if it doesn't, something is wrong; open an issue.

## Check your understanding

1. A node holds 10 samples of class A and 10 of class B. Compute its Gini impurity, and the
   impurity decrease of a split that produces children (10 A, 2 B) and (0 A, 8 B).
2. Why is it sufficient to test only midpoints between consecutive sorted unique feature
   values, rather than every real number, as thresholds?
3. With `max_depth=None` and `min_samples_leaf=1`, what is the training accuracy of a CART
   tree on any dataset without duplicate-feature/conflicting-label rows? Why is that a
   problem?
4. Decision trees don't care whether you measure petal length in cm or log-cm, but linear
   and logistic regression do. Explain.
5. Greedy induction picks the locally best split. Sketch (or describe) a dataset where the
   best first split does *not* lead to the best overall tree — hint: think XOR.
6. In `best_split`, the comparison is `decrease > best[0]` (strict greater-than), not
   `>=`. What tie-breaking rule does that enforce, and what would change about the chosen
   split if the code used `>=` instead?
7. `predict_one` walks the tree with a `while` loop, but `build` and `print_tree` call
   themselves recursively. Why doesn't prediction need recursion the way tree-building and
   printing do?

From here, you're now ready to proceed to try and reproduce
[Lesson 08 — Ensembles](08-ensembles.md).

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
