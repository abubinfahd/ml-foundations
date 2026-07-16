# Lesson 02 — Probability & Distributions

Statistics summarizes the data you have; probability describes the process that generated
it. In this lesson you simulate coin flips to watch the law of large numbers at work,
compute the binomial pmf and normal pdf from their formulas, and see the central limit
theorem turn skewed exponential draws into bell-shaped sample means — then check every
closed-form number against `scipy.stats`.

## Concept

**Probability as long-run frequency.** Saying a fair coin has $P(\text{heads}) = 0.5$ is a
claim about the *long run*: as the number of flips $n$ grows, the proportion of heads
converges to 0.5. Any short run can wander — 10 flips can easily give 0.4 or 0.7.

Here $P(\cdot)$ just means "the probability of the thing inside the parentheses," so
$P(\text{heads})$ reads as "the probability that a single flip lands heads." Try a short
run by hand: flip a coin four times (or imagine it) and get, say, H, T, H, H. The running
proportion of heads after each flip is $1/1=1.000$, $1/2=0.500$, $2/3\approx0.667$,
$3/4=0.750$ — nowhere near settling down. That is expected; four flips is a "short run."
The claim above is only about what happens if you keep flipping for a *long* time, which is
exactly what the simulation later in this lesson does at a scale (100,000 flips) nobody
would attempt by hand.

**Law of large numbers.** For i.i.d. draws $X_1, \dots, X_n$ with mean $\mu$, the sample
mean $\bar{X}_n = \frac{1}{n}\sum_i X_i$ converges to $\mu$ as $n \to \infty$. This is the
license to *estimate probabilities by simulation*: simulate many draws, count the fraction
of the time an event happens, and you have an estimate that improves with $n$.

Unpacking the notation: "i.i.d." means *independent and identically distributed* — every
draw $X_i$ comes from the same underlying random process (identical), and no draw
influences any other (independent); repeated coin flips are the textbook example. The list
$X_1, \dots, X_n$ is just the first $n$ draws, $\mu$ ("mu") is the true, usually unknown,
long-run average of the process, $\sum_i$ means "add up over all $i$ from 1 to $n$," and
$\bar{X}_n$ (read "X-bar sub n") is the ordinary arithmetic average of those $n$ draws. Why
this matters: it is the reason you are allowed to *compute* a probability by simulating it
instead of deriving it. Roll an ordinary six-sided die five times and write down what you
get — say 3, 6, 1, 4, 2. The sample mean is $(3+6+1+4+2)/5 = 3.2$. The true mean of a fair
die is $\mu = 3.5$. One more roll could move the sample mean noticeably; a million rolls
could barely move it at all — that shrinking wobble as $n$ grows is the law of large
numbers, and it is what lets `law_of_large_numbers()` below stand in for a formal proof.

**Distributions, pmf, pdf.** A distribution assigns probability to every possible outcome
of a random variable. For a *discrete* variable the *probability mass function* (pmf)
gives $P(X = k)$ directly, and the masses sum to 1. For a *continuous* variable,
$P(X = x)$ is zero for any single $x$; instead the *probability density function* (pdf)
$f(x)$ gives probability per unit of $x$, and probabilities are areas:
$P(a \le X \le b) = \int_a^b f(x)\,dx$. A density can exceed 1 — only the total area must
equal 1.

A *random variable* $X$ is just a name for "the numeric outcome of a random process": $X$
could be "number of heads in 10 flips" or "height of a randomly chosen adult." It is
*discrete* if its outcomes are countable (0, 1, 2, ... heads) and *continuous* if its
outcomes fill a range of real numbers (any height in centimeters). For a discrete variable
the pmf is easy to picture: a fair six-sided die has $P(X{=}k) = 1/6$ for each
$k \in \{1,2,3,4,5,6\}$, and those six numbers sum to exactly 1 — every possible outcome is
accounted for. A continuous variable has infinitely many possible values, so asking "what is
the probability height is *exactly* 170.0000... cm" is asking about a single point on a
continuum, and the honest answer is 0; what you can ask instead is "what is the probability
height falls *between* 169 and 171 cm," which is the area under the density curve $f(x)$
between those bounds — the integral $\int_a^b f(x)\,dx$ is calculus notation for "the area
under the curve from $a$ to $b$." That is why a density value like 0.3989 (below) is not
itself a probability and is allowed to exceed 1 for narrow enough distributions.

**Binomial distribution.** The number of successes in $n$ independent trials, each with
success probability $p$, is $X \sim \text{Binomial}(n, p)$ with pmf

$$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$$

$\binom{n}{k}$ counts the arrangements of $k$ successes among $n$ trials (`math.comb` in
Python); $p^k(1-p)^{n-k}$ is the probability of any one such arrangement.

Read $X \sim \text{Binomial}(n, p)$ as "$X$ is distributed as a Binomial with parameters
$n$ and $p$." $n$ is the number of trials (10 coin flips), $p$ is the probability that any
*one* trial is a "success" (0.3, for a coin biased toward tails), and $k$ is the particular
success count whose probability you want. $\binom{n}{k}$ ("n choose k") counts the distinct
orders in which $k$ successes can appear among $n$ trials — it accounts for the fact that
"success, success, failure" and "failure, success, success" are different sequences that
both count as "2 successes in 3 trials." Work a tiny case by hand: $n=3$, $p=0.5$ (three
fair coin flips), $k=2$. $\binom{3}{2}=3$ (the arrangements HHT, HTH, THH), $p^k=0.5^2=0.25$,
$(1-p)^{n-k}=0.5^1=0.5$, so $P(X{=}2)=3 \times 0.25 \times 0.5 = 0.375$. Doing this for
$k=0,1,2,3$ gives $0.125$, $0.375$, $0.375$, $0.125$, which sums to exactly 1, as any pmf
must. This distribution matters whenever you count successes out of a fixed number of
independent yes/no trials — clicks out of impressions, defective parts out of a batch,
correct answers out of a quiz.

**Normal distribution.** The pdf of $\mathcal{N}(\mu, \sigma^2)$ is

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

The *standard* normal has $\mu = 0$, $\sigma = 1$; its density peaks at
$1/\sqrt{2\pi} \approx 0.3989$ and is symmetric, so $f(-x) = f(x)$.

$\mathcal{N}(\mu, \sigma^2)$ reads "Normal distribution with mean $\mu$ and variance
$\sigma^2$": $\mu$ sets *where* the bell curve is centered, and $\sigma$ (the standard
deviation, $\sigma^2$ the variance) sets *how wide* it is spread. $e$ is Euler's number
($\approx 2.71828$) and $\pi$ is the usual circle constant; both are fixed constants that
make the total area under the curve integrate to exactly 1. The exponent is easiest to read
through the *z-score* $z = (x-\mu)/\sigma$ — "how many standard deviations $x$ is from the
mean" — which turns the formula into the simpler $\frac{1}{\sigma\sqrt{2\pi}} e^{-z^2/2}$:
the further $x$ is from $\mu$ in standard-deviation units, the smaller the density,
symmetrically in both directions. This distribution matters because it is the limiting
shape of sums and averages of almost anything (see the CLT next), which is why measurement
errors, test scores, and — as you will use in Lesson 03 — sampling distributions are so
often modeled as normal.

**Central limit theorem.** Take sample means of size $n$ from *any* distribution with
mean $\mu$ and finite variance $\sigma^2$ — even a heavily skewed one. As $n$ grows, the
distribution of $\bar{X}_n$ tends to $\mathcal{N}(\mu, \sigma^2/n)$: normal, centered at
$\mu$, with standard deviation $\sigma/\sqrt{n}$. This is why the normal distribution
appears everywhere: averages are normal-ish even when the raw data is not. We test it on
Exponential(rate=1), which has $\mu = 1$, $\sigma = 1$, so the prediction for $n = 30$ is
mean 1 and standard deviation $1/\sqrt{30} \approx 0.1826$.

This is arguably the single most important fact in the lesson, so it is worth restating
plainly: no matter how strange the shape of your original data is — skewed, lopsided,
multi-humped — if you repeatedly take a sample of $n$ values and average them, the
distribution of *those averages* looks more and more like a normal curve as $n$ grows,
centered on the same mean $\mu$ as the original data, but with a shrinking spread of
$\sigma/\sqrt{n}$ (a quantity often called the *standard error of the mean*). The only
requirements are that the draws are independent and that $\mu$ and $\sigma^2$ are both
finite — a distribution with infinite variance is one of the few things that can break this.
The arithmetic for the exponential example: $\sigma=1$ and $n=30$, so
$\sigma/\sqrt{n} = 1/\sqrt{30}$; $\sqrt{30} \approx 5.477$, and $1/5.477 \approx 0.1826$,
exactly the "CLT prediction" row printed below. This same result is what lets Lesson 03's
hypothesis tests treat sample means as approximately normal even when the underlying data is
not.

## Dataset

No download this time — every number in this lesson is **simulated**, and the simulation
itself is the tool being taught. `numpy.random.default_rng(42)` is a deterministic
pseudo-random generator: same seed, same "random" numbers, on every machine. That is what
makes a lesson about randomness reproducible character for character.

## From scratch

Read [`src/lesson02_probability/from_scratch.py`](../src/lesson02_probability/from_scratch.py)
first — it uses only `math` from the standard library plus NumPy's generator, no scipy.
`binomial_pmf` and `normal_pdf` are the two formulas from the Concept section, verbatim.

### Step-by-step: reading the code

Walk through the file in the order Python actually executes it — imports and constants
first, then each function definition (which does nothing until called), then the `main()`
driver at the bottom that calls them in sequence.

```python
import math

import numpy as np

SEED = 42
```

`math` is the standard-library module that supplies `comb` (binomial coefficients) and
`exp`/`sqrt` for the pdf formula; `numpy` supplies the random number generator used for
every simulated draw in this file. `SEED = 42` is an arbitrary fixed integer. *Seeding* a
pseudo-random generator means telling it exactly where to start in its (deterministic, but
random-looking) sequence of numbers: the same seed always produces the same sequence, on
any machine. That is the whole reason the "Expected output" block further down can list
exact numbers instead of "approximately."

```python
def binomial_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p): C(n, k) * p^k * (1-p)^(n-k)."""
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)
```

This is the Binomial pmf formula from the Concept section, transcribed line for line.
`math.comb(n, k)` computes $\binom{n}{k}$ using exact integer arithmetic (no rounding), and
`p**k * (1 - p) ** (n - k)` computes $p^k(1-p)^{n-k}$. Multiplying the two gives
$P(X=k)$. The type hints (`k: int, n: int, p: float`) and `-> float` are optional
documentation for humans and tools — Python does not enforce them at runtime, but they make
the formula's inputs and output unambiguous at a glance.

```python
def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Density of Normal(mu, sigma^2) at x."""
    z = (x - mu) / sigma
    return math.exp(-(z**2) / 2) / (sigma * math.sqrt(2 * math.pi))
```

Same idea for the normal pdf. `z = (x - mu) / sigma` computes the z-score described in the
Concept section, and the return line is exactly
$\frac{1}{\sigma\sqrt{2\pi}} e^{-z^2/2}$, `math.exp` being $e^{(\cdot)}$. The default
arguments `mu: float = 0.0, sigma: float = 1.0` mean "if the caller does not specify a mean
or standard deviation, assume the standard normal" — which is exactly how the `normal()`
function below calls it: `normal_pdf(x)` with no second or third argument.

```python
def law_of_large_numbers() -> None:
    """Running proportion of heads in fair coin flips converges to 0.5."""
    rng = np.random.default_rng(SEED)
    flips = rng.integers(0, 2, size=100_000)  # 1 = heads, 0 = tails
    heads = flips.cumsum()

    print("-- Law of large numbers: fair coin, true P(heads) = 0.5 --")
    print(f"{'n':>8} {'prop(heads)':>12} {'|prop - 0.5|':>13}")
    for n in [10, 100, 1_000, 10_000, 100_000]:
        prop = heads[n - 1] / n
        print(f"{n:>8} {prop:>12.4f} {abs(prop - 0.5):>13.4f}")
```

`np.random.default_rng(SEED)` creates a `Generator` object — NumPy's modern, recommended
way to get reproducible random numbers (the older `np.random.seed(...)` plus bare
`np.random.rand(...)` style is considered legacy). `rng.integers(0, 2, size=100_000)` draws
100,000 independent random integers from `{0, 1}` in a single call and returns them as one
NumPy array of shape `(100000,)`. This is a *vectorized* operation: instead of writing a
Python `for` loop that appends one random number at a time (slow, because each iteration
pays Python's interpreter overhead), NumPy generates the whole array in compiled C code and
hands it back as one object. `flips.cumsum()` is another vectorized call — "cumulative
sum" — that returns an array of the same length where each position holds the running total
of everything before and including it; so `heads[i]` is the number of heads among the first
`i + 1` flips. The loop then just looks up `heads[n - 1]` (arrays are 0-indexed, so the
$n$-th flip is at position `n - 1`) and divides by `n` to get the running proportion at each
checkpoint — a direct, literal implementation of "sample mean as $n$ grows" from the Concept
section, where "heads" is coded as 1 and "tails" as 0, so the mean of the flips *is* the
proportion of heads.

```python
def binomial() -> None:
    """Binomial pmf from the formula vs estimated from simulated draws."""
    rng = np.random.default_rng(SEED)
    n, p, n_draws = 10, 0.3, 100_000
    draws = rng.binomial(n, p, size=n_draws)
    counts = np.bincount(draws, minlength=n + 1)

    print(f"-- Binomial(n={n}, p={p}) pmf: formula vs {n_draws} simulated draws --")
    print(f"{'k':>3} {'formula':>10} {'simulated':>10}")
    for k in range(n + 1):
        print(f"{k:>3} {binomial_pmf(k, n, p):>10.4f} {counts[k] / n_draws:>10.4f}")
```

Notice a fresh `rng = np.random.default_rng(SEED)` is created again here, rather than
reusing the one from `law_of_large_numbers()`. Re-seeding independently in each function
means every section's numbers depend only on that section's own draws, not on how many
random numbers a previous section happened to consume — so the output stays exactly
reproducible even if you reorder or comment out other functions. `rng.binomial(n, p,
size=n_draws)` draws 100,000 samples directly from a Binomial(10, 0.3) distribution — NumPy
has this distribution built in, so each of the 100,000 numbers is itself "how many successes
out of 10 trials," simulating flipping a $p=0.3$ coin 10 times, 100,000 times over.
`np.bincount(draws, minlength=n + 1)` counts how many times each integer appeared in
`draws` and returns those counts as an array; `minlength=n + 1` guarantees the array has (at
least) positions `0` through `n`, even if some large value like `10` never happened to occur
in the simulation. Dividing `counts[k] / n_draws` turns a raw count into an estimated
probability — the same law-of-large-numbers logic as before, now applied separately to each
possible outcome $k$ instead of only to the overall mean, letting the loop print the formula
and the simulated estimate for every $k$ side by side.

```python
def normal() -> None:
    """Standard normal density at a few points."""
    print("-- Standard normal pdf --")
    print(f"{'x':>3} {'pdf(x)':>10}")
    for x in [-2, -1, 0, 1, 2]:
        print(f"{x:>3} {normal_pdf(x):>10.4f}")
```

No randomness at all here — this is pure formula evaluation at five fixed points, using
`normal_pdf`'s defaults (`mu=0.0, sigma=1.0`), i.e., the standard normal from the Concept
section. It exists mainly so the next script, `with_library.py`, has exact numbers to check
itself against.

```python
def central_limit_theorem() -> None:
    """Means of Exponential(rate=1) samples look normal around 1 +/- 1/sqrt(n)."""
    rng = np.random.default_rng(SEED)
    n, n_means = 30, 10_000
    sample_means = rng.exponential(scale=1.0, size=(n_means, n)).mean(axis=1)

    print(f"-- Central limit theorem: {n_means} means of n={n} draws, Exponential(rate=1) --")
    print(f"{'':<22} {'observed':>9} {'CLT prediction':>15}")
    print(f"{'mean of sample means':<22} {sample_means.mean():>9.4f} {1.0:>15.4f}")
    print(f"{'std  of sample means':<22} {sample_means.std(ddof=1):>9.4f} {1 / math.sqrt(n):>15.4f}")
```

`rng.exponential(scale=1.0, size=(n_means, n))` introduces a *2-D* array shape:
`size=(10_000, 30)` asks for a table with 10,000 rows and 30 columns, i.e., 10,000
independent "experiments," each consisting of 30 draws from Exponential(rate=1). Calling
`.mean(axis=1)` on that table computes a mean *per row*: `axis=1` means "collapse across
columns, separately for each row," turning each row's 30 raw draws into that row's single
sample mean, and leaving a 1-D array of 10,000 sample means — this array *is* the collection
of $\bar{X}_n$ values ($n=30$ here) that the central limit theorem makes a claim about.
Calling `.mean()` again with no `axis` argument then averages that entire 1-D array down to
one number: the observed "mean of sample means." `.std(ddof=1)` computes the sample standard
deviation of that same array; `ddof=1` ("delta degrees of freedom") tells NumPy to divide by
$n-1$ rather than $n$ — Bessel's correction, which gives a less biased estimate of the
population's standard deviation from a sample. With 10,000 sample means the difference
between dividing by 10,000 and 9,999 is tiny, but it is good practice to default to `ddof=1`
whenever you are estimating a standard deviation from data rather than computing it from a
known, fixed population. `1 / math.sqrt(n)` is the closed-form CLT prediction
$\sigma/\sqrt{n}$ with $\sigma=1$, printed alongside the observed value for direct
comparison.

```python
def main() -> None:
    print("== Probability & distributions: from scratch ==")
    print()
    law_of_large_numbers()
    print()
    binomial()
    print()
    normal()
    print()
    central_limit_theorem()


if __name__ == "__main__":
    main()
```

`main()` calls the four demonstration functions in the same order they appear in the
"Expected output" block below (law of large numbers, binomial, normal, CLT), with a bare
`print()` between each call to produce the blank separator lines you will see in the output.
`if __name__ == "__main__":` is a standard Python idiom meaning "only run `main()` when this
file is executed directly as a script" (`python from_scratch.py`) — it does *not* run when
another file does `from from_scratch import binomial_pmf`, which is what lets other code
reuse the formula functions without also triggering the entire printout.

```bash
python src/lesson02_probability/from_scratch.py
```

Expected output:

```text
== Probability & distributions: from scratch ==

-- Law of large numbers: fair coin, true P(heads) = 0.5 --
       n  prop(heads)  |prop - 0.5|
      10       0.4000        0.1000
     100       0.5200        0.0200
    1000       0.5090        0.0090
   10000       0.4939        0.0061
  100000       0.5009        0.0009

-- Binomial(n=10, p=0.3) pmf: formula vs 100000 simulated draws --
  k    formula  simulated
  0     0.0282     0.0285
  1     0.1211     0.1199
  2     0.2335     0.2329
  3     0.2668     0.2679
  4     0.2001     0.2006
  5     0.1029     0.1021
  6     0.0368     0.0372
  7     0.0090     0.0091
  8     0.0014     0.0014
  9     0.0001     0.0001
 10     0.0000     0.0000

-- Standard normal pdf --
  x     pdf(x)
 -2     0.0540
 -1     0.2420
  0     0.3989
  1     0.2420
  2     0.0540

-- Central limit theorem: 10000 means of n=30 draws, Exponential(rate=1) --
                        observed  CLT prediction
mean of sample means      0.9984          1.0000
std  of sample means      0.1825          0.1826
```

Things to notice:

- The coin's error shrinks from 0.1 at $n=10$ to 0.0009 at $n=100{,}000$ — the law of
  large numbers in action. It is not monotone in general; convergence is a trend, not a
  promise for each step. Look at the table: the error actually goes 0.1000, 0.0200,
  0.0090, 0.0061, 0.0009 — mostly shrinking, but you cannot promise the very next row will
  always beat the previous one.
- The simulated binomial pmf agrees with the formula to about $\pm 0.001$ — that is the
  simulation noise you get with 100,000 draws (it shrinks like $1/\sqrt{n}$, so quadrupling
  the draws roughly halves it).
- $k = 10$ prints as 0.0000, but $P(X{=}10) = 0.3^{10} \approx 5.9 \times 10^{-6}$ is not
  zero — it is just below our 4-decimal rounding. The same is true of the `simulated`
  column: with only 100,000 draws and a true probability under one in a hundred thousand,
  seeing zero exact hits is unsurprising, not a bug.
- The exponential distribution is strongly right-skewed, yet its sample means land almost
  exactly on the CLT prediction (0.9984 vs 1, 0.1825 vs 0.1826). That is the central limit
  theorem doing its job: the raw exponential draws are nothing like a bell curve, but their
  averages already are, at a sample size of only $n=30$.

## With a library

The same closed-form numbers, computed by `scipy.stats`. Read
[`with_library.py`](../src/lesson02_probability/with_library.py) — `binom.pmf`,
`norm.pdf`, and `expon.mean()/std()` replace our hand-written formulas. The print formats
match, so the columns line up with the from-scratch run.

### Step-by-step: reading the code

```python
import math

from scipy import stats
```

`scipy.stats` is a module full of ready-made distribution objects. `stats.binom`,
`stats.norm`, and `stats.expon` (used below) each expose methods like `.pmf()`, `.pdf()`,
`.mean()`, and `.std()` that implement exactly the formulas from the Concept section, tested
and optimized by the scipy project instead of hand-written by us. `math` is still imported
for `math.sqrt` in the CLT calculation, the one piece of arithmetic scipy is not asked to do
here.

```python
def binomial() -> None:
    n, p = 10, 0.3
    print(f"-- Binomial(n={n}, p={p}) pmf: scipy.stats.binom --")
    print(f"{'k':>3} {'pmf':>10}")
    for k in range(n + 1):
        print(f"{k:>3} {stats.binom.pmf(k, n, p):>10.4f}")
```

`stats.binom.pmf(k, n, p)` is a drop-in replacement for our hand-written
`binomial_pmf(k, n, p)` — same $\binom{n}{k}p^k(1-p)^{n-k}$ formula from the Concept
section, computed by a battle-tested library instead of `math.comb` and `**`. There is no
`simulated` column here: this script only checks the closed-form numbers against
scipy, it does not draw any random samples of its own.

```python
def normal() -> None:
    print("-- Standard normal pdf: scipy.stats.norm --")
    print(f"{'x':>3} {'pdf(x)':>10}")
    for x in [-2, -1, 0, 1, 2]:
        print(f"{x:>3} {stats.norm.pdf(x):>10.4f}")
```

`stats.norm.pdf(x)` defaults to $\mu=0$, $\sigma=1$ — the standard normal — exactly like
our `normal_pdf(x)` defaults, replacing the z-score formula with a library call.

```python
def central_limit_theorem() -> None:
    n = 30
    mu = stats.expon.mean()  # Exponential(rate=1): mean = 1
    sigma = stats.expon.std()  # Exponential(rate=1): std = 1
    print(f"-- CLT prediction for Exponential(rate=1), n={n}: scipy.stats.expon --")
    print(f"population mean: {mu:.4f}")
    print(f"population std : {sigma:.4f}")
    print(f"{'':<22} {'':>9} {'CLT prediction':>15}")
    print(f"{'mean of sample means':<22} {'':>9} {mu:>15.4f}")
    print(f"{'std  of sample means':<22} {'':>9} {sigma / math.sqrt(n):>15.4f}")
```

Instead of simulating 10,000 sample means, this function reads the exponential
distribution's *true* mean and standard deviation straight from scipy —
`stats.expon.mean()` and `stats.expon.std()` — then plugs them into the same
$\sigma/\sqrt{n}$ formula used in `from_scratch.py`. `stats.expon` defaults to
`scale=1.0`, and scipy parameterizes the exponential distribution by `scale` (which equals
the mean, $1/\text{rate}$) rather than by `rate` directly — with the default `scale=1.0`
this happens to equal `rate=1`, but it is a common gotcha if you later want a different
rate. Because there is no simulation, the printed table has only a `CLT prediction`
column and no `observed` column — there is nothing observed, only computed.

```python
def main() -> None:
    print("== Probability & distributions: with scipy ==")
    print()
    binomial()
    print()
    normal()
    print()
    central_limit_theorem()


if __name__ == "__main__":
    main()
```

Same driver pattern as `from_scratch.py`: call each demonstration function in the order it
appears in the "Expected output" block below, with blank `print()` calls between them, gated
behind `if __name__ == "__main__":` so importing this file elsewhere would not automatically
print anything.

```bash
python src/lesson02_probability/with_library.py
```

Expected output:

```text
== Probability & distributions: with scipy ==

-- Binomial(n=10, p=0.3) pmf: scipy.stats.binom --
  k        pmf
  0     0.0282
  1     0.1211
  2     0.2335
  3     0.2668
  4     0.2001
  5     0.1029
  6     0.0368
  7     0.0090
  8     0.0014
  9     0.0001
 10     0.0000

-- Standard normal pdf: scipy.stats.norm --
  x     pdf(x)
 -2     0.0540
 -1     0.2420
  0     0.3989
  1     0.2420
  2     0.0540

-- CLT prediction for Exponential(rate=1), n=30: scipy.stats.expon --
population mean: 1.0000
population std : 1.0000
                                  CLT prediction
mean of sample means                      1.0000
std  of sample means                      0.1826
```

The `pmf` and `pdf(x)` columns must match the from-scratch `formula` and `pdf(x)` columns
exactly — those are deterministic formulas, so any difference means a bug. The *simulated*
column from the from-scratch run is the only one scipy does not reproduce: scipy computes
the formula, our simulation estimates it.

## Check your understanding

1. The pdf of the standard normal at $x=0$ is 0.3989. Why is it wrong to say "the
   probability that $X = 0$ is 0.3989"? What is $P(X = 0)$ exactly?
2. The simulated binomial pmf differs from the formula in the third decimal. If you used
   1,000,000 draws instead of 100,000, roughly how much smaller would the noise get?
3. The CLT predicts the *standard deviation* of sample means is $\sigma/\sqrt{n}$. If you
   increased the sample size from $n=30$ to $n=120$, what would the predicted sd become?
4. The law of large numbers says the *proportion* of heads converges to 0.5. Does the
   *count* of heads minus $n/2$ also converge to 0? (Hint: at $n=100{,}000$ the proportion
   error is 0.0009 — how many flips is that?)
5. The exponential distribution is skewed, yet its sample means are nearly normal at
   $n=30$. Name a condition in the CLT statement that a distribution could violate, making
   this fail.
6. `central_limit_theorem()` in `from_scratch.py` uses `sample_means.std(ddof=1)` (Bessel's
   correction) rather than `ddof=0`. With `n_means = 10,000` sample means, would switching
   to `ddof=0` change the printed value at 4 decimal places? Why or why not?
7. `stats.expon` in `with_library.py` is parameterized by `scale` (the mean), not `rate`.
   If you wanted the CLT prediction for an Exponential distribution with rate 2 (mean 0.5),
   what argument would you pass to `stats.expon.mean(...)` and `stats.expon.std(...)`?

From here, you're now ready to proceed to try and reproduce
[Lesson 03 — Hypothesis Testing](03-hypothesis-testing.md).

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
