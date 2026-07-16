# ML Journey 🧭

A hands-on, reproducible onboarding path for **Statistics and Machine Learning**,
inspired by the [castorini onboarding](https://github.com/castorini/onboarding) model.

Every lesson is a markdown guide with **copy-paste PowerShell commands** for Windows. You
download a small open dataset, implement the concept **from scratch in NumPy**, then verify
your results against a standard library (scikit-learn, statsmodels, SciPy, or PyTorch).
Every script is seeded — your output should match the expected output in the guide (metrics
may differ in the last decimal or two from one machine to another; see
[Reproducibility rules](#reproducibility-rules)).

When you successfully reproduce a lesson, you add one line to its **Reproduction Log** and
send a pull request. That's the whole idea: *if you can reproduce it, you understood it.*

## Prerequisites

- **OS:** Windows 10 or 11.
- **RAM:** 8 GB minimum
- **GPU:** not needed — everything runs on CPU in minutes
- **Software:** [Python 3.10+](https://www.python.org/downloads/windows/) (tick *"Add
  python.exe to PATH"* during install) and [Git for Windows](https://git-scm.com/download/win).
  Commands are run in **PowerShell**.
- **Disk:** ~2 GB free (Python packages + datasets)

No prior ML experience is required. Comfort with Python and a terminal is assumed.

## The Path

Work through the lessons **in order** — each builds on the previous ones.

| # | Lesson | You implement | You verify with | Dataset |
|---|--------|---------------|-----------------|---------|
| 00 | [Setup](docs/00-setup.md) | your environment | — | — |
| 01 | [Descriptive Statistics](docs/01-descriptive-stats.md) | mean, median, variance, quantiles, correlation | pandas | Palmer Penguins |
| 02 | [Probability & Distributions](docs/02-probability-distributions.md) | simulation, law of large numbers, CLT | SciPy | simulated |
| 03 | [Hypothesis Testing](docs/03-hypothesis-testing.md) | t-test, permutation test, chi-square | SciPy | Palmer Penguins |
| 04 | [Linear Regression](docs/04-linear-regression.md) | normal equation, gradient descent | scikit-learn, statsmodels | Wine Quality |
| 05 | [Logistic Regression](docs/05-logistic-regression.md) | sigmoid, log-loss, gradient descent | scikit-learn | Titanic |
| 06 | [Model Evaluation](docs/06-model-evaluation.md) | k-fold CV, precision/recall/F1, ROC-AUC | scikit-learn | Titanic |
| 07 | [Decision Trees](docs/07-decision-trees.md) | CART with Gini impurity | scikit-learn | Iris |
| 08 | [Ensembles](docs/08-ensembles.md) | bagging / simplified random forest | scikit-learn | Adult Income |
| 09 | [Clustering](docs/09-clustering.md) | k-means, elbow method | scikit-learn | Palmer Penguins |
| 10 | [PCA](docs/10-pca.md) | covariance eigendecomposition | scikit-learn | Wine Quality |
| 11 | [Neural Networks from Scratch](docs/11-neural-networks.md) | MLP with backpropagation in NumPy | — | MNIST (subset) |
| 12 | [Intro to Deep Learning](docs/12-deep-learning-intro.md) | — | PyTorch (CPU): MLP & small CNN | Fashion-MNIST |

## How to participate

New to Git, GitHub, or pull requests? Don't worry — follow these steps exactly and you'll be
fine. Each step tells you *what* to do and *why*.

### Step 1 — Fork the repository (make your own copy)

A **fork** is your personal copy of this repo on GitHub. You do all your work in the fork, then
propose your changes back to the original.

1. Make sure you're signed in to [GitHub](https://github.com/) (create a free account if you
   don't have one).
2. Open the project page: `https://github.com/abubinfahd/ml-foundations`.
3. Click the **Fork** button in the top-right corner, then **Create fork**.
4. You now have your own copy at `https://github.com/<your-username>/ml-foundations`.

### Step 2 — Get the code onto your computer

Open **PowerShell** (search "PowerShell" in the Start menu) and run these, replacing
`<your-username>` with your GitHub username:

```powershell
git clone https://github.com/<your-username>/ml-foundations.git
cd ml-foundations
git remote add upstream https://github.com/abubinfahd/ml-foundations.git
```

- `git clone` downloads your fork to a new `ml-foundations` folder.
- `cd` moves into that folder — **run every later command from here** (the "repo root").
- `git remote add upstream …` remembers the original repo so you can pull in updates later.

### Step 3 — Set up your environment

Complete [Lesson 00 — Setup](docs/00-setup.md). It walks you through installing Python, creating
a virtual environment, installing the packages, and verifying everything works. **Do this once
before any other lesson.**

### Step 4 — Work through a lesson

Open the lessons **in order** (start with [Lesson 01](docs/01-descriptive-stats.md)). For each one:

1. Read the **Concept** section.
2. Run the download command to get the dataset.
3. Run the **From scratch** script and read the code.
4. Run the **With a library** script and confirm the two outputs agree.
5. Compare your terminal output to the **Expected output** block in the doc. They should match
   (tiny last-decimal differences are normal — see [Reproducibility rules](#reproducibility-rules)).

### Step 5 — Record your success (the Reproduction Log)

When your output matches, you've earned the right to sign the lesson's log. Append **one line** to
the *Reproduction Log* section at the bottom of that lesson's doc:

```
+ Results reproduced by [@yourusername](https://github.com/yourusername) on 2026-07-14 (commit [abc1234](https://github.com/abubinfahd/ml-foundations/commit/abc1234567890abcdef1234567890abcdef1234))
```

To fill in the commit part, run these in PowerShell and copy the values:

```powershell
git rev-parse --short HEAD   # the short hash → use as the [abc1234] anchor text
git rev-parse HEAD           # the full hash  → use at the end of the commit URL
```

The commit must be one that exists on the `master` branch. Replace the date with the day you
reproduced it (`YYYY-MM-DD`).

### Step 6 — Send your work back (open a pull request)

Once you've finished a few lessons, submit them **all together in one pull request** — see the
step-by-step Git commands in [CONTRIBUTING.md](CONTRIBUTING.md).

> **Don't open a separate pull request for each lesson.** Batch your log lines into a single PR.

### Something wrong?

Found a bug, a broken link, or an output that doesn't match no matter what you try?
[Open an issue](../../issues/new/choose) — questions and bug reports are welcome and help improve
the course for the next person.

## Repository layout

```
docs/       one markdown guide per lesson
src/        Python code, one folder per lesson (from_scratch.py + with_library.py)
scripts/    download_data.ps1 (datasets + checksums), verify_setup.py
data/       datasets land here (gitignored — never committed)
outputs/    plots land here (gitignored)
tests/      runs every lesson script and diffs its output against the doc
```

## Reproducibility rules

- Every script sets `SEED = 42` for `random`, NumPy (and PyTorch in lesson 12), and pins
  BLAS/PyTorch to a single thread so the maths is as deterministic as possible.
- Printed metrics are rounded to 4 decimal places.
- `scripts/download_data.ps1` verifies the SHA-256 checksum of every file it downloads.
- **Last-decimal differences are normal.** The neural-network and deep-learning lessons
  (11–12) train with mini-batch SGD, and the last decimals of a loss or accuracy depend on
  your CPU. The test suite (`tests/`) checks output within a small tolerance rather than
  character-for-character, so these lessons still "pass" on any machine. If a metric is off
  by more than ~0.05, or a *from-scratch vs. library* comparison disagrees noticeably,
  something is wrong — open an issue rather than logging a reproduction.

## License

Code and documentation are [MIT licensed](LICENSE). Datasets belong to their original
authors; each lesson notes the source and license of the dataset it uses.
