# Lesson 09 — Clustering

Every lesson so far had labels to learn from. This one doesn't: you get 333 penguins'
measurements with the species column *hidden*, and the algorithm must discover the group
structure on its own. You implement k-means (Lloyd's algorithm) from scratch, pick $k$
with the elbow method, and then check that scikit-learn finds exactly the same optimum —
and that the discovered clusters largely recover the actual species.

## Concept

**Unsupervised learning.** With no target to predict, the goal changes: find structure —
groups of similar points. There is no accuracy to compute during training, only an
internal notion of "tight groups".

**The k-means objective.** Choose $k$ centroids $\mu_1, \dots, \mu_k$ and an assignment
$c(i)$ of each point to a centroid, minimizing the *within-cluster sum of squares*
(a.k.a. **inertia**):

$$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

In plain English, term by term:

- $n$ is the number of data points (333 penguins here).
- $x_i$ is the $i$-th data point itself, a vector of feature values — in this lesson, the
  four standardized measurements for one penguin.
- $k$ is the number of clusters you decide to look for *before* running the algorithm
  (k-means never learns $k$ on its own — that is a separate problem, the elbow method
  below).
- $\mu_1, \dots, \mu_k$ are the **centroids**: one vector per cluster, playing the role of
  that cluster's "center of mass". A centroid is not necessarily one of the actual data
  points — it is the mean of whichever points currently belong to that cluster.
- $c(i)$ is the index (between $1$ and $k$) of the centroid that point $i$ is currently
  assigned to — "which cluster does point $i$ belong to right now".
- $\lVert x_i - \mu_{c(i)} \rVert^2$ is the squared Euclidean distance between point $i$
  and its own assigned centroid: subtract the two vectors coordinate-by-coordinate, square
  each difference, add them up. This is the same squared-difference idea behind variance
  in Lesson 01 — here applied between a point and its cluster's center instead of between
  a point and the overall mean.
- $J$, the sum of all of those squared distances over every point, is a single number that
  measures how tightly packed the clusters are overall. Small $J$ means every point sits
  close to its own centroid; large $J$ means points are scattered far from whatever center
  they were assigned to.

Minimizing $J$ is a reasonable stand-in for "find compact groups" because a centroid that
sits at the mean of its members is, by a basic least-squares fact, the single point that
minimizes the summed squared distance to those members (the same reason the mean minimizes
squared deviations when you compute variance). So low $J$ really does mean each group has
collapsed around a sensible center.

**Lloyd's algorithm** is alternating minimization on $J$: (1) *assign* each point to its
nearest centroid ($J$ can only drop, centroids fixed); (2) *update* each centroid to the
mean of its points (the mean is the point minimizing summed squared distance, assignments
fixed). Each pass lowers $J$, and there are finitely many assignments, so it converges — in
a handful of iterations here.

Why neither step can make $J$ worse: in the assignment step, every point independently
picks whichever centroid is currently nearest to it — nothing else about $J$ depends on
that one point's choice, so switching to a strictly closer centroid can only shrink (or
leave unchanged) that point's contribution to the total. In the update step, the
assignments are frozen and each cluster's centroid is recomputed as the mean of its
current members; since the mean is provably the minimizer of summed squared distance to a
fixed set of points, moving the centroid anywhere else could only raise that cluster's
contribution to $J$. "Alternating minimization" is exactly this: fix one thing, optimize
the other, swap, repeat — each half-step can only help, so $J$ decreases (or stays flat)
every single pass until nothing changes anymore.

**Sensitivity to initialization → restarts.** Lloyd's converges to a *local* minimum of
$J$; a bad random start can leave two centroids splitting one true group. The standard fix
is brute force: run many times from different random starts and keep the lowest inertia
(we use 10 restarts; sklearn's `n_init=10` is the same idea, plus a smarter "k-means++"
seeding).

Why local minima happen at all: the number of ways to partition $n$ points into $k$
non-empty groups is astronomically large, and Lloyd's algorithm only ever moves between
these partitions one reassignment at a time, always downhill in $J$. That means it can
walk into a partition where no single point's reassignment would help — a *local* optimum
— while a completely different partition, reachable only by reassigning several points at
once, has strictly lower $J$. Different starting centroids send the search down different
paths through this landscape, which is exactly why restarts from varied starting points
raise the odds of landing in (or near) the true global optimum, or at least the best one
found so far.

**Choosing $k$ — the elbow.** Inertia always decreases as $k$ grows ($k = n$ gives
$J = 0$), so you can't just minimize it. Instead you look for the "elbow": the $k$ after
which extra clusters stop buying much. It is a judgment call, not a statistic — reasonable
people can read the same curve differently. That subjectivity is a real limitation.

**Standardize first.** k-means lives entirely on Euclidean distance. Unstandardized, body
mass (std ≈ 800 g) would dwarf bill depth (std ≈ 2 mm) and the clustering would be
body-mass-only. z-scoring every feature gives each an equal vote. (A pleasant side effect:
with standardized features, the $k=1$ inertia is exactly $n \times d = 333 \times 4 = 1332$.)

As a reminder from Lesson 01, z-scoring a feature means replacing every value $x$ with
$z = (x - \text{mean}) / \text{std}$: subtract the column's mean, divide by the column's
standard deviation. After this transform every feature has mean 0 and standard deviation
1, so a "difference of 1" means the same thing (one standard deviation) no matter which
feature you're looking at — which is what makes summing squared differences *across
different features* in the distance formula fair.

**Assumptions.** Minimizing squared Euclidean distance means k-means looks for compact,
roughly *spherical*, similar-sized blobs. Elongated, nested, or crescent-shaped clusters
break it — no number of restarts fixes an objective that is wrong for the shape of your data.

### A tiny worked example, by hand

Five toy points in 2 dimensions, $k=2$, one full iteration of Lloyd's algorithm, done by
hand (no standardization needed here — the numbers are already on comparable scales):

| point | coordinates |
|---|---|
| A | (1, 1) |
| B | (1, 2) |
| C | (4, 4) |
| D | (5, 4) |
| E | (5, 5) |

**Initialize** by picking two of the points themselves as starting centroids (the same
strategy `from_scratch.py` uses): $\mu_1 = A = (1,1)$, $\mu_2 = C = (4,4)$.

**Assignment step.** Compute the squared distance from every point to both centroids and
keep the smaller one:

| point | dist² to μ₁=(1,1) | dist² to μ₂=(4,4) | assigned cluster |
|---|---|---|---|
| A | 0 | 18 | 1 |
| B | 1 | 13 | 1 |
| C | 18 | 0 | 2 |
| D | 25 | 1 | 2 |
| E | 32 | 2 | 2 |

Cluster 1 = {A, B}, cluster 2 = {C, D, E}. Summing the "assigned" column gives the current
inertia: $J = 0 + 1 + 0 + 1 + 2 = 4$.

**Update step.** Move each centroid to the mean of its current members:

$$\mu_1 = \text{mean}(A, B) = \left(\tfrac{1+1}{2}, \tfrac{1+2}{2}\right) = (1, 1.5)$$

$$\mu_2 = \text{mean}(C, D, E) = \left(\tfrac{4+5+5}{3}, \tfrac{4+4+5}{3}\right) = \left(\tfrac{14}{3}, \tfrac{13}{3}\right) \approx (4.667, 4.333)$$

Recomputing distances to these new centroids gives a smaller total inertia
($0.5$ for cluster 1, about $1.33$ for cluster 2, total $\approx 1.83$, down from $4$) —
exactly the guarantee above: the update step can only lower $J$.

**Convergence check.** Run the assignment step again with the new centroids: every point
is still closest to the same centroid as before (A and B still nearest to $\mu_1$; C, D,
E still nearest to $\mu_2$). Nothing changed, so Lloyd's algorithm has converged after a
single iteration for this toy set. This is precisely what the convergence check in
`from_scratch.py` tests for — "did the assignment vector stay identical to last time?" —
and it is covered step by step below.

## Dataset

[Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) again (License: CC0) —
you already downloaded it in Lesson 01, so the script just verifies the checksum:

```powershell
.\scripts\download_data.ps1 penguins
```

Expected output:

```text
[skip] penguins.csv (already downloaded, checksum OK)
Done.
```

We use the four numeric measurement columns (bill length/depth, flipper length, body
mass), drop the rows with missing values (344 → 333), and standardize all four. The
species column is held aside — not as a training signal, but as an answer key to inspect
afterwards.

## From scratch

Read [`from_scratch.py`](../src/lesson09_clustering/from_scratch.py). The whole of Lloyd's
algorithm is the ~20-line `kmeans` function: init with 3 distinct random data points,
assign (distance ties go to the lowest cluster index), update (an empty cluster keeps its
old centroid), stop when assignments stop changing. Restart seeds are derived from
`default_rng(42)`; the best run is the lowest inertia (first wins ties). Since cluster
numbering is arbitrary, output clusters are relabeled largest-first for stability.

```bash
python src/lesson09_clustering/from_scratch.py
```

Expected output:

```text
== k-means clustering (Lloyd's algorithm): from scratch ==
rows used: 333 (rows with missing values dropped)
features (standardized): bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g

k=3, 10 restarts (seed 42):
best inertia: 370.7661
iterations of best run: 10
cluster sizes (largest first): 129, 119, 85

cluster x species contingency (clusters sorted by size):
                 Adelie  Chinstrap     Gentoo
  cluster 1         124          5          0
  cluster 2           0          0        119
  cluster 3          22         63          0

elbow: best inertia by k (10 restarts each):
  k=1  inertia=1332.0000
  k=2  inertia= 552.6710
  k=3  inertia= 370.7661
  k=4  inertia= 293.9048
  k=5  inertia= 228.5065
  k=6  inertia= 200.6772
  k=7  inertia= 182.9965
  k=8  inertia= 172.7676
```

Things to notice:

- Without ever seeing a species label, k-means put all 119 Gentoos alone in one cluster,
  and largely separated Adelie from Chinstrap (27 of 333 penguins sit in the "wrong"
  cluster). The structure was in the measurements all along — the four numbers per
  penguin (bill length/depth, flipper length, body mass) already encode enough of the
  species differences that distance alone recovers most of the grouping.
- The elbow: the drop $1332 \to 553 \to 371$ is steep, then flattens ($371 \to 294 \to
  229 \to \dots$). Most people would read the bend at $k = 2$ or $3$ — the true species
  count, 3, is a *defensible* reading, not an obvious one. That ambiguity is the method:
  nothing in the numbers *forces* $k=3$, it is simply where the returns start
  diminishing.
- $k=1$ inertia is exactly 1332 = $333 \times 4$, the total variance of standardized data
  (with ddof=0) — a good sanity check for your implementation: with one cluster the
  centroid is just the overall mean, so inertia collapses to the sum of squared z-scores,
  which is $n \times d$ by construction.
- The best run of $k=3$ converged in only 10 assign/update passes (`iterations of best
  run: 10`), far short of the `max_iter=300` safety cap in the code — a sign the clusters
  are well separated enough that assignments settle quickly, much like the 1-iteration
  convergence in the tiny worked example above.

### Step-by-step: reading the code

Walk through `from_scratch.py` in the order it actually runs: imports and constants,
`load_data`, `kmeans`, `best_of`, then the `main` driver.

**Imports and constants.**

```python
import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data" / "penguins.csv"

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
SEED = 42
K = 3
N_RESTARTS = 10
```

No scikit-learn anywhere — only NumPy plus the standard-library `csv` and `pathlib`
modules. `FEATURES` names the four columns used for distance (matching the Concept
section's four-dimensional $x_i$); `SEED` is the master seed that makes every run of this
script produce identical numbers; `K = 3` and `N_RESTARTS = 10` are the $k$ and the
restart count discussed above.

**`load_data` — reading and standardizing.**

```python
def load_data() -> tuple[np.ndarray, list[str]]:
    """Return (X standardized, species). Rows with any missing value are
    dropped (344 -> 333); features are z-scored (mean 0, std 1, ddof=0 —
    the same convention as sklearn's StandardScaler)."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.DictReader(f) if all(v not in ("", "NA") for v in r.values())]
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, [r["species"] for r in rows]
```

`csv.DictReader` turns each CSV row into a dictionary keyed by column name; the list
comprehension keeps only rows where every value is neither empty string nor `"NA"` — this
is the 344 → 333 drop mentioned throughout the lesson. The next line builds a 2-D NumPy
array of shape `(333, 4)` by pulling the four `FEATURES` out of every row and converting
to `float`. The line after that is the z-score formula from the Concept section, applied
to the whole array at once:

`X.mean(axis=0)` computes one mean *per column* (`axis=0` reduces down the rows), giving a
length-4 array; `X.std(axis=0)` does the same for standard deviation. Subtracting and
dividing a `(333, 4)` array by two length-4 arrays works because of NumPy **broadcasting**:
NumPy lines up the length-4 array against the last axis of `X` and repeats it for every
one of the 333 rows, so effectively every row gets its own 4 means and 4 stds subtracted
and divided out, without writing an explicit loop. The function returns the standardized
matrix plus the species column, kept as a plain Python list and never touched again until
the diagnostic contingency table at the end.

**`kmeans` — initialization.**

```python
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.full(len(X), -1)
    n_iter = 0
```

`rng.choice(len(X), size=k, replace=False)` draws $k$ distinct row indices uniformly at
random, without repeats, from `0 .. len(X)-1`. Indexing `X` with that array of indices
picks out $k$ actual data points to serve as the starting centroids $\mu_1, \dots, \mu_k$
— using real data points as the initial guess rather than arbitrary coordinates.
`.copy()` matters because `centroids` gets overwritten in place later; without it, the
slice could alias back into `X`. `labels` starts as an array of `-1` sentinels — no point
belongs to any real cluster yet — which guarantees the very first comparison in the loop
below counts as "assignments changed".

**`kmeans` — the assignment step (distance via broadcasting, then `argmin`).**

```python
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)  # ties go to the lowest cluster index
        if np.array_equal(new_labels, labels):
            break
```

This is the heart of the algorithm, and it is worth reading slowly. `X[:, None, :]`
inserts a new axis in the middle, reshaping `X` from `(333, 4)` to `(333, 1, 4)`.
`centroids[None, :, :]` inserts a new axis at the front, reshaping `centroids` from
`(k, 4)` to `(1, k, 4)`. Subtracting a `(333, 1, 4)` array from a `(1, k, 4)` array
broadcasts both up to shape `(333, k, 4)` — NumPy effectively computes *every point minus
every centroid*, for all `333 * k` pairs at once, with no explicit Python loop over points
or clusters. Squaring and then `.sum(axis=2)` collapses the trailing feature axis, leaving
a `(333, k)` matrix `d2` where `d2[i, j]` is exactly $\lVert x_i - \mu_j \rVert^2$ from the
Concept formula — the squared distance from point $i$ to centroid $j$, for every $(i, j)$
pair simultaneously. `d2.argmin(axis=1)` then scans each row (each point) and returns the
column index of the smallest entry — the closest centroid — which is precisely $c(i)$,
the assignment function from the Concept section. NumPy's `argmin` returns the *first*
occurrence of the minimum, which is why the comment notes that distance ties go to the
lowest cluster index. Finally, `np.array_equal(new_labels, labels)` compares the freshly
computed assignment against the previous one; if nothing moved, Lloyd's algorithm has
converged and the loop `break`s — this is the formal version of the "run the assignment
step again, nothing changed" check performed by hand in the worked example above.

**`kmeans` — the update step and inertia.**

```python
        labels = new_labels
        n_iter += 1
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:  # an empty cluster keeps its old centroid
                centroids[j] = members.mean(axis=0)
    inertia = float(((X - centroids[labels]) ** 2).sum())
    return labels, inertia, n_iter
```

Once assignments have actually changed, `n_iter` counts the pass and the loop recomputes
each centroid. `labels == j` is a boolean mask (`True` where a point belongs to cluster
`j`); `X[labels == j]` uses that mask to select just that cluster's rows — NumPy's
boolean/fancy indexing. `members.mean(axis=0)` averages those rows column-by-column,
producing the new centroid — exactly "move the centroid to the mean of its assigned
points" from Lloyd's algorithm in the Concept section. The `if len(members) > 0` guard
protects against a cluster that lost every member during the assignment step (which can
happen, especially early on or with larger $k$): rather than computing the mean of zero
rows (which would produce `NaN`), the code simply leaves that centroid where it was.
After the loop exits — either by `break`ing on convergence or by exhausting `max_iter` —
`centroids[labels]` uses fancy indexing to build a `(333, 4)` array whose $i$-th row is the
centroid currently assigned to point $i$; subtracting from `X`, squaring, and summing
every remaining element collapses everything down to one scalar — exactly $J$ from the
Concept formula. The function returns the final labels, that inertia, and how many
iterations it took.

**`best_of` — restarts.**

```python
def best_of(X: np.ndarray, k: int, seeds: np.ndarray):
    """Run kmeans once per seed, keep the lowest inertia (first wins ties)."""
    best = None
    for s in seeds:
        result = kmeans(X, k, np.random.default_rng(int(s)))
        if best is None or result[1] < best[1]:
            best = result
    return best
```

This is the "restarts" idea from the Concept section made concrete: `kmeans` is called
once per seed, and each call gets its own independent `np.random.default_rng(int(s))`, so
each restart starts from a different random set of centroids. `result[1]` pulls out the
inertia element of the `(labels, inertia, n_iter)` tuple `kmeans` returns; the strict `<`
comparison means only a strictly lower inertia replaces `best`, so the first run to reach
the lowest inertia found so far wins any tie.

**`main` — setup.**

```python
def main() -> None:
    X, species = load_data()
    species_names = sorted(set(species))
    # One deterministic seed per restart, derived from the master seed.
    seeds = np.random.default_rng(SEED).integers(0, 2**31 - 1, size=N_RESTARTS)
```

`species_names` gives a stable, alphabetically sorted list of species for the contingency
table's column headers. The `seeds` line seeds one master RNG from `SEED = 42`, then draws
10 large random integers from it — each of those 10 integers becomes the seed for one
restart's own independent RNG inside `best_of`. This is what makes the whole script fully
reproducible (same `SEED` always produces the same 10 restart seeds, hence the same
output) while still giving each of the 10 restarts a genuinely different starting point.

**`main` — running k=3 and relabeling by size.**

```python
    labels, inertia, n_iter = best_of(X, K, seeds)
    print(f"k={K}, {N_RESTARTS} restarts (seed {SEED}):")
    print(f"best inertia: {inertia:.4f}")
    print(f"iterations of best run: {n_iter}")

    # Cluster indices are arbitrary; relabel by size (largest first, ties by
    # original index) so the output is stable and comparable across runs.
    sizes = np.bincount(labels, minlength=K)
    order = sorted(range(K), key=lambda j: (-sizes[j], j))
    print(f"cluster sizes (largest first): {', '.join(str(sizes[j]) for j in order)}")
```

`np.bincount(labels, minlength=K)` counts how many points carry each label value (0, 1,
2, …), giving the size of every cluster. `sorted(range(K), key=lambda j: (-sizes[j], j))`
sorts the cluster indices by descending size (the `-sizes[j]`), breaking any tie by the
original index `j` — producing a presentation order only, since (as the Concept section
notes) the numeric cluster labels k-means assigns are arbitrary and carry no meaning of
their own.

**`main` — the contingency table.**

```python
    print("cluster x species contingency (clusters sorted by size):")
    print(f"  {'':<10}" + "".join(f"{s:>11}" for s in species_names))
    for rank, j in enumerate(order, start=1):
        counts = [
            sum(1 for lab, sp in zip(labels, species) if lab == j and sp == name)
            for name in species_names
        ]
        print(f"  {f'cluster {rank}':<10}" + "".join(f"{c:>11}" for c in counts))
```

The header line right-aligns each species name in an 11-character column. For every
cluster, in size order, `counts` zips the predicted `labels` together with the true
`species` list and counts, for each species name, how many points landed in this cluster
*and* belong to that species. This is purely a diagnostic: it compares what k-means found
against the species column that was held aside, but k-means itself never saw that column
during clustering.

**`main` — the elbow loop.**

```python
    print(f"elbow: best inertia by k ({N_RESTARTS} restarts each):")
    for k in range(1, 9):
        _, k_inertia, _ = best_of(X, k, seeds)
        print(f"  k={k}  inertia={k_inertia:9.4f}")
```

The same `best_of` function (and the same 10 seeds, reused for every value of $k$ for a
fair comparison) is called for every $k$ from 1 to 8, printing the best inertia found at
each. This loop produces exactly the elbow-curve numbers used in the Concept section to
discuss choosing $k$.

## With a library

Read [`with_library.py`](../src/lesson09_clustering/with_library.py) — same preprocessing,
then `KMeans(n_clusters=3, n_init=10, random_state=42)` plus an *adjusted Rand index*
against the species answer key.

```bash
python src/lesson09_clustering/with_library.py
```

Expected output:

```text
== k-means clustering: with scikit-learn ==
rows used: 333 (rows with missing values dropped)
features (standardized): bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g

KMeans(n_clusters=3, n_init=10, random_state=42):
inertia: 370.7661
cluster sizes (largest first): 129, 119, 85

cluster x species contingency (clusters sorted by size):
                 Adelie  Chinstrap     Gentoo
  cluster 1         124          5          0
  cluster 2           0          0        119
  cluster 3          22         63          0

adjusted Rand index (clusters vs species): 0.7994
```

The inertia **matches the from-scratch run to all 4 decimals (370.7661)**, and the cluster
sizes and contingency table are identical — different code, different initialization
(k-means++ vs. random points), same global optimum, because with enough restarts on
well-separated data both searches find the same best partition. One subtlety makes the
comparison possible at all: **cluster labels are arbitrary**. Run k-means twice and "cluster
0" may become "cluster 2" — the partition is the same, the names are not. That's why both
scripts sort clusters by size before printing, and why the adjusted Rand index (0.7994,
where 1.0 = identical partitions, ≈0 = random) is computed from pair co-memberships and is
invariant to any relabeling. Never compare clusterings by matching label integers directly.

### Step-by-step: reading the code

`with_library.py` follows the same shape as `from_scratch.py`, but replaces the
hand-written math with three scikit-learn pieces.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
```

`StandardScaler` replaces the hand-written z-score line; `KMeans` replaces the whole
`kmeans` / `best_of` pair — init, assign, update, convergence check, and restarts, all in
one class; `adjusted_rand_score` replaces a metric you did not have to derive yourself.

```python
    X = StandardScaler().fit_transform(X)
```

`fit_transform` computes the same per-column mean and standard deviation as the
from-scratch line `(X - X.mean(axis=0)) / X.std(axis=0)` and applies them in a single
call, using the same `ddof=0` convention — which is exactly why the $k=1$ inertia sanity
check ($n \times d = 1332$) holds for both scripts.

```python
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED)
    labels = km.fit_predict(X)
```

`n_clusters=K` is the $k=3$ from the Concept section. `n_init=10` is the restart count —
the same idea as the `seeds` loop in `from_scratch.py` — except scikit-learn's default
initialization is "k-means++", a smarter, distance-weighted seeding scheme that tends to
spread the initial centroids out rather than dropping them at $k$ uniformly random data
points; both scripts use 10 restarts here purely to make the comparison fair.
`random_state=SEED` plays the same reproducibility role as the `SEED` constant in
`from_scratch.py`. `fit_predict` runs the entire assign/update loop to convergence,
across all `n_init` restarts, and returns the labels of the best run — everything that
`kmeans` and `best_of` do together, in one call.

```python
    print(f"inertia: {km.inertia_:.4f}")
```

`km.inertia_` is the fitted estimator's stored value of $J$ — the identical formula the
from-scratch script computes by hand at the end of `kmeans`: the sum of squared distances
from every point to its assigned centroid.

The cluster-size and contingency-table printing block that follows is copied verbatim
from `from_scratch.py` (same `np.bincount`, same sorted `order`, same zip-and-count logic)
— no new concepts there, just the same presentation code applied to scikit-learn's labels
instead of the hand-rolled ones.

```python
    ari = adjusted_rand_score(species, labels)
    print(f"adjusted Rand index (clusters vs species): {ari:.4f}")
```

`adjusted_rand_score` compares two labelings of the same points — here, the true species
and the discovered clusters — by looking at every pair of points and checking whether both
labelings agree on "same group" or "different group" for that pair, then correcting for
the level of agreement expected by chance. Because it only ever asks "are these two points
in the same group under labeling A, and under labeling B", it does not care what the
integer label values themselves are — which is exactly the relabeling-invariance the
paragraph above relies on.

## Check your understanding

1. Prove (or argue) that each of Lloyd's two steps — reassignment and centroid update —
   can never increase the inertia $J$. Why does that guarantee convergence?
2. Why can't you choose $k$ by simply minimizing inertia over $k$? What does the elbow
   method do instead, and what makes it subjective?
3. What would the contingency table look like if we had skipped standardization? Which
   single feature would dominate the distance, and why?
4. The adjusted Rand index is invariant to permuting cluster labels. Why would plain
   "accuracy between cluster labels and species labels" be a broken metric here?
5. Sketch a 2-D dataset with two clearly visible clusters where k-means with $k=2$ must
   fail no matter how many restarts you give it.
6. In the tiny worked example, redo the assignment step with both initial centroids drawn
   from the same tight group (say $\mu_1 = A$ and $\mu_2 = B$) instead of one from each
   group. Does a single restart still find the {A,B} / {C,D,E} partition? What does this
   tell you about why `N_RESTARTS = 10` matters more on some datasets than others?
7. `kmeans` in `from_scratch.py` keeps a cluster's old centroid when that cluster ends up
   with zero members after an assignment step. Under what circumstances can a cluster
   become empty mid-run, and why is "keep the old centroid" a safer fallback than, say,
   resetting it to the origin or to a new random point?

From here, you're now ready to proceed to try and reproduce [Lesson 10 — PCA](10-pca.md).

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
