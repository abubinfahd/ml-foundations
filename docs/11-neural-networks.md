# Lesson 11 — Neural Networks from Scratch

This is the capstone of the from-scratch track. You already have every ingredient: linear
models (lesson 04), the softmax idea (lesson 05), and gradient descent. Here you stack
them into a two-layer neural network, derive its gradients with the chain rule, and train
it in pure NumPy to read handwritten digits — no framework, every formula visible as code.
Lesson 12 rebuilds the same network in PyTorch.

## Concept

**Plain-English vocabulary first.** The rest of this section is precise math; here is what
each term means before you see the symbols, for a reader who has done lessons 01–10 but
never touched a neural network:

- **Layer** — one computational step that turns one vector of numbers into another vector
  of numbers: multiply by a matrix, add a bias vector, then (usually) apply a
  nonlinearity. Our network has exactly two layers: a hidden layer and an output layer.
- **Weight matrix** — a table of numbers, one entry per (input feature, output unit) pair,
  recording how strongly that input should push that output up or down. Multiplying a row
  of inputs by this matrix computes every output's weighted sum in one step; that matrix
  product *is* the layer.
- **Activation** — the numbers a layer produces after its nonlinearity: the signal handed
  to the next layer. `a1` in the code is the hidden layer's activation, computed from `z1`
  (the hidden layer's raw, pre-nonlinearity output).
- **Gradient** — for a single parameter (one entry of a weight matrix or bias), the
  gradient is one number: how much the loss would change if that parameter were nudged up
  by a tiny amount, and in which direction. Stack the gradients of every entry of a
  parameter into an array the same shape as that parameter, and "subtract a small step in
  the gradient's direction" becomes a single array subtraction — that is gradient descent.
- **Epoch** — one full pass through the entire training set. With 10,000 training images
  and mini-batches of 64, one epoch is $\lceil 10000/64 \rceil = 157$ mini-batch updates;
  the script runs 8 epochs (`EPOCHS = 8`), so 1,256 parameter updates in total.

**A neuron is a linear model plus a nonlinearity.** One neuron computes
$a = \varphi(\mathbf{w}^\top \mathbf{x} + b)$ — with $\varphi$ the sigmoid, that is exactly
the logistic regression of lesson 05. A *layer* is many neurons in parallel, which is one
matrix product: $\mathbf{z} = \mathbf{x} W + \mathbf{b}$ (we use row vectors, matching the
code, where a batch of $m$ inputs is a matrix $X$ with $m$ rows). A *network* is layers
composed: our model is

$$Z_1 = X W_1 + \mathbf{b}_1 \qquad A_1 = \mathrm{ReLU}(Z_1) \qquad Z_2 = A_1 W_2 + \mathbf{b}_2 \qquad \hat{Y} = \mathrm{softmax}(Z_2)$$

with $W_1 \in \mathbb{R}^{784 \times 128}$ and $W_2 \in \mathbb{R}^{128 \times 10}$ —
101,770 trainable parameters in total.

**A tiny worked example.** Before scaling up to 784 inputs and 128 hidden units, here is
the exact same arithmetic done by hand for a toy network with 2 inputs and a single hidden
unit. Say the input is $\mathbf{x} = [2,\ 1]$, the hidden weight vector (a $2\times1$
weight matrix) is $\mathbf{w}_1 = [0.5,\ -0.3]$ with bias $b_1 = 0.1$, and the output
weight is $w_2 = 1.5$ with bias $b_2 = -1.0$:

$$z_1 = \mathbf{x} \cdot \mathbf{w}_1 + b_1 = (2)(0.5) + (1)(-0.3) + 0.1 = 0.8$$

$$a_1 = \mathrm{ReLU}(z_1) = \max(0,\ 0.8) = 0.8 \quad\text{(positive, so it passes through unchanged)}$$

$$z_2 = a_1 w_2 + b_2 = (0.8)(1.5) + (-1.0) = 0.2$$

Now the same weights with a different input, $\mathbf{x} = [1,\ 3]$:

$$z_1 = (1)(0.5) + (3)(-0.3) + 0.1 = -0.3 \qquad a_1 = \mathrm{ReLU}(-0.3) = 0 \qquad z_2 = (0)(1.5) - 1.0 = -1.0$$

This second row shows the hidden unit "switched off": once $z_1$ drops below zero, ReLU
clips it to exactly 0, the output collapses to just the bias $b_2$, and — as the backward
pass below will show — no gradient flows back through this unit for this particular
example. The real network runs this same arithmetic 128 times in parallel for the hidden
layer (one column of $W_1$ per hidden unit) and 10 times for the output layer, for every
row of a batch of up to 64 images simultaneously — which is exactly why the code is
written as matrix multiplication rather than a Python loop over neurons.

**Why the nonlinearity is not optional.** A composition of linear maps is linear:

$$(X W_1) W_2 = X (W_1 W_2)$$

so any stack of purely linear layers, however deep, collapses to a single linear layer —
it could never do better than lesson 05's linear decision boundaries. The nonlinearity
between layers is what makes depth buy anything; with enough hidden units, one hidden
layer can approximate any continuous function (the universal approximation theorem). A
nonlinearity used this way — applied elementwise between two linear layers — is called an
*activation function*; ReLU (below) is this network's choice.

**ReLU.** The *rectified linear unit* is $\mathrm{ReLU}(z) = \max(0, z)$. Its derivative
is 1 where $z > 0$ and 0 where $z < 0$ — in backprop it acts as a mask that passes or
blocks the gradient. It is cheap and, unlike the sigmoid, does not saturate for positive
inputs (a saturated sigmoid has a near-zero gradient, so learning stalls).

**Softmax + cross-entropy.** The hidden layer's job is to build useful internal features;
the output layer's job is different — it must produce something interpretable as "how
likely is each of the 10 digits". The output layer produces 10 raw scores ("logits").
Softmax turns them into probabilities:

$$\hat{y}_j = \frac{e^{z_j}}{\sum_{c=1}^{10} e^{z_c}}$$

and the *cross-entropy* loss penalizes the probability assigned to the true class:
$L = -\log \hat{y}_{\text{true}}$. The pair is standard because the gradient of the loss
with respect to the logits collapses to something beautiful — for a one-hot label
$\mathbf{y}$:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$$

All the exponentials and logarithms cancel; the error signal is literally "predicted
probabilities minus truth". (Numerical note: softmax is shift-invariant,
$\mathrm{softmax}(\mathbf{z}) = \mathrm{softmax}(\mathbf{z} - c)$, so the code subtracts
the row maximum before exponentiating to avoid overflow.)

**Backpropagation is the chain rule, organized layer by layer.** Training means adjusting
every weight and bias so the loss goes down, and the chain rule tells us exactly how much
to adjust each one: start from how the loss changes with respect to the *output* of the
last layer, then walk backward one layer at a time, converting "how the loss changes with
respect to this layer's output" into "how the loss changes with respect to this layer's
inputs and parameters". Every gradient below is an array the same shape as the quantity it
is a gradient of ($\partial L/\partial W_2$ has the same shape as $W_2$, and so on), so
each one plugs directly into a plain array subtraction during the update step. For a
mini-batch of $m$ examples with mean loss $L = -\frac{1}{m}\sum_{i=1}^{m} \log \hat{Y}_{i, y_i}$,
walk the forward pass backwards. At the output (the softmax + cross-entropy shortcut,
averaged over the batch):

$$\frac{\partial L}{\partial Z_2} = \frac{1}{m}\left(\hat{Y} - Y\right)$$

Through the second linear layer $Z_2 = A_1 W_2 + \mathbf{b}_2$:

$$\frac{\partial L}{\partial W_2} = A_1^\top \frac{\partial L}{\partial Z_2} \qquad\qquad \frac{\partial L}{\partial \mathbf{b}_2} = \sum_{i=1}^{m} \left(\frac{\partial L}{\partial Z_2}\right)_{i}$$

Back through $W_2$ into the hidden activations, then through the ReLU (elementwise mask):

$$\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial Z_2} W_2^\top \qquad\qquad \frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbf{1}[Z_1 > 0]$$

And through the first linear layer:

$$\frac{\partial L}{\partial W_1} = X^\top \frac{\partial L}{\partial Z_1} \qquad\qquad \frac{\partial L}{\partial \mathbf{b}_1} = \sum_{i=1}^{m} \left(\frac{\partial L}{\partial Z_1}\right)_{i}$$

Each equation is exactly one line in the script. Notice the pattern that generalizes to
any depth: the gradient for a weight matrix is (its layer's input)$^\top$ times (the
gradient at its output), and the gradient flows backward through a linear layer by
multiplying by $W^\top$. The "Step-by-step: reading the code" walkthrough below points at
the exact line of `from_scratch.py` that implements each of the five equations above.

**Mini-batch SGD.** The full-batch gradient is exact but gives one update per pass over
the data (one update per epoch, as defined above); a single example gives noisy, cheap
updates. Mini-batches (here 64) are the compromise: 157 updates per epoch, each averaged
over enough examples to point roughly downhill. Every parameter moves by
$\theta \leftarrow \theta - \eta\, \partial L / \partial \theta$ with a fixed learning rate
$\eta$. The training set is reshuffled every epoch — with a seeded generator, so every run
shuffles identically.

**He initialization.** Starting from all zeros fails completely: every hidden unit would
compute the same output, receive the same gradient, and stay identical forever (the
symmetry is never broken). Random weights break symmetry, but their scale matters — too
large and activations explode, too small and gradients vanish. He initialization draws
$W \sim \mathcal{N}(0,\ 2/n_{\text{in}})$: the $1/n_{\text{in}}$ keeps the variance of
$\mathbf{x} W$ independent of the layer width, and the factor 2 compensates for the ReLU
zeroing out half of its inputs.

## Dataset

[MNIST](https://yann.lecun.com/exdb/mnist/) — 70,000 grayscale 28×28 images of handwritten
digits (60,000 train, 10,000 test), collected by NIST and assembled by LeCun, Cortes and
Burges; the download script fetches the CVDF mirror. License: CC BY-SA 3.0.

```powershell
.\scripts\download_data.ps1 mnist
```

Expected output:

```text
[get ] mnist-train-images-idx3-ubyte.gz
[ ok ] mnist-train-images-idx3-ubyte.gz (checksum verified)
[get ] mnist-train-labels-idx1-ubyte.gz
[ ok ] mnist-train-labels-idx1-ubyte.gz (checksum verified)
[get ] mnist-t10k-images-idx3-ubyte.gz
[ ok ] mnist-t10k-images-idx3-ubyte.gz (checksum verified)
[get ] mnist-t10k-labels-idx1-ubyte.gz
[ ok ] mnist-t10k-labels-idx1-ubyte.gz (checksum verified)
Done.
```

(If the files were downloaded earlier, you will see `[skip]` lines instead.)

These are not CSVs — you can't `head` them. They are gzipped files in the IDX binary
format, which takes about ten lines of NumPy to parse (see `load_idx_images` /
`load_idx_labels` in the script):

| file kind | header (big-endian int32) | then |
|-----------|---------------------------|------|
| images    | magic `0x00000803`, count, rows, cols (16 bytes) | one `uint8` per pixel, 0–255 |
| labels    | magic `0x00000801`, count (8 bytes)              | one `uint8` per label, 0–9  |

To keep the run fast on a single CPU thread, the script trains on the **first 10,000**
training images and evaluates on the **first 2,000** test images — a deterministic subset,
so no sampling seed is even needed. Pixels are scaled from 0–255 to $[0, 1]$.

## From scratch

Read [`src/lesson11_neural_networks/from_scratch.py`](../src/lesson11_neural_networks/from_scratch.py)
first. It is pure NumPy. The heart of it is the training loop: a forward pass of six
lines, and a backward pass of seven gradient lines — each one commented with the equation
it implements from the Concept section.

### Step-by-step: reading the code

This subsection walks through `from_scratch.py` in the order it actually executes,
quoting the real lines from the file. Have the file open alongside this walkthrough; it
is the most code-dense lesson in the course, so take it slowly.

**Imports and determinism setup.** Before NumPy is even imported, the script pins every
BLAS thread-count environment variable to `1`:

```python
import gzip
import math
import os
from pathlib import Path

# Pin BLAS to one thread *before* importing NumPy: multi-threaded matrix
# products can change the order of floating-point additions, which changes
# the last bits of the result from machine to machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
```

NumPy's matrix multiply is normally handed off to a multi-threaded BLAS library, and
floating-point addition is not associative — summing the same numbers in a different order
can change the last few bits of a result. Pinning to one thread is what makes the
"Expected output" numbers below reproducible bit-for-bit across machines.

**Hyperparameters.** All the numbers that shape training live in one place, as module
constants:

```python
SEED = 42
N_TRAIN = 10_000  # first 10k of the 60k training images (deterministic subset)
N_TEST = 2_000    # first 2k of the 10k test images
HIDDEN = 128
BATCH = 64
LR = 0.3
EPOCHS = 8
```

These are exactly the numbers quoted throughout the Concept section: 128 hidden units,
mini-batches of 64, learning rate 0.3, 8 epochs.

**Parsing MNIST's IDX format.** `load_idx_images` and `load_idx_labels` turn the gzipped
binary files described in the Dataset section into NumPy arrays:

```python
def load_idx_images(path: Path) -> np.ndarray:
    """Parse an IDX image file: 16-byte header (magic, count, rows, cols as
    big-endian int32), then one uint8 per pixel. Returns shape (n, rows*cols)."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n, rows, cols = np.frombuffer(raw, dtype=">i4", count=4)
    assert magic == 0x00000803, f"bad magic in {path.name}: {magic:#x}"
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows * cols)
```

`np.frombuffer` reinterprets a block of raw bytes as an array without copying it, given a
dtype. `">i4"` means "big-endian ($\texttt{>}$), 4-byte signed integer" — that is how the
IDX header packs `magic`, `n`, `rows`, and `cols`, four of them back to back, hence
`count=4`. The pixel data is read the same way with `offset=16` (skip the 16-byte header)
and `dtype=np.uint8` (one byte per pixel), then `.reshape(n, rows * cols)` folds the flat
byte stream into one row of 784 pixels per image — flattening the 28×28 grid into a single
784-length vector is exactly what feeds the $X W_1$ product in the Concept section.
`load_idx_labels` is the same idea with an 8-byte header and no reshape, since labels are
already one number per image:

```python
def load_idx_labels(path: Path) -> np.ndarray:
    """Parse an IDX label file: 8-byte header (magic, count), then uint8 labels."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n = np.frombuffer(raw, dtype=">i4", count=2)
    assert magic == 0x00000801, f"bad magic in {path.name}: {magic:#x}"
    return np.frombuffer(raw, dtype=np.uint8, offset=8)[:n]
```

**The forward pass, as a reusable function.** `predict` is the same forward pass used both
for the final accuracy number and inside the training loop's per-epoch evaluation:

```python
def predict(x, w1, b1, w2, b2) -> np.ndarray:
    """Forward pass, returning the predicted class per row. Softmax is
    monotonic, so argmax over the logits equals argmax over the probabilities."""
    a1 = np.maximum(x @ w1 + b1, 0.0)
    return (a1 @ w2 + b2).argmax(axis=1)
```

Three NumPy idioms appear here for the first time in the script:

- **`@` is matrix multiplication.** `x @ w1` multiplies an $(m, 784)$ batch matrix by the
  $(784, 128)$ weight matrix $W_1$, producing an $(m, 128)$ result — one dot product per
  (example, hidden unit) pair, computed in a single vectorized call instead of a Python
  loop over 128 neurons times $m$ examples.
- **Broadcasting.** `+ b1` adds a length-128 vector to every row of that $(m, 128)$ matrix.
  NumPy "broadcasts" the shorter shape `(128,)` across the batch dimension automatically,
  so every row gets the same bias added without an explicit loop.
- **`argmax`.** `.argmax(axis=1)` returns, for each row, the column index of the largest
  value — the predicted digit class. The docstring's comment about monotonicity is why
  `predict` never bothers computing softmax probabilities: softmax preserves order (the
  largest logit is always the largest probability), so comparing raw logits gives the same
  predicted class as comparing probabilities, for less arithmetic.

`np.maximum(z, 0.0)` is ReLU applied elementwise: for every entry it keeps the value if
positive, otherwise substitutes 0 — precisely $\max(0, z)$ from the Concept section.

**Loading data and scaling pixels.** Inside `main`, the four arrays are loaded and sliced
down to the deterministic subsets described in the Dataset section, then rescaled:

```python
x_train = load_idx_images(DATA / "mnist-train-images-idx3-ubyte.gz")[:N_TRAIN]
y_train = load_idx_labels(DATA / "mnist-train-labels-idx1-ubyte.gz")[:N_TRAIN]
x_test = load_idx_images(DATA / "mnist-t10k-images-idx3-ubyte.gz")[:N_TEST]
y_test = load_idx_labels(DATA / "mnist-t10k-labels-idx1-ubyte.gz")[:N_TEST]

# Scale pixels from [0, 255] to [0, 1]. float32 keeps the arrays small.
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
```

`[:N_TRAIN]` and `[:N_TEST]` are plain Python slicing, but note what they buy: only a
sixth of the 60,000 training images are used, which is why the run finishes in about a
second. Dividing by `255.0` puts every pixel on a $[0, 1]$ scale, which keeps `w1 @ x`,
`z1`, and friends in a well-behaved numeric range (the same feature-scaling reasoning as
lesson 04).

**Parameter initialization (He init).** This is the numeric form of the He initialization
described in the Concept section:

```python
w1 = rng.standard_normal((d, HIDDEN), dtype=np.float32) * math.sqrt(2.0 / d)
b1 = np.zeros(HIDDEN, dtype=np.float32)
w2 = rng.standard_normal((HIDDEN, k), dtype=np.float32) * math.sqrt(2.0 / HIDDEN)
b2 = np.zeros(k, dtype=np.float32)
```

`rng.standard_normal((d, HIDDEN))` draws a $(784, 128)$ array of independent samples from
a standard normal distribution ($\mu=0,\ \sigma=1$); multiplying by
`math.sqrt(2.0 / d)` rescales the standard deviation to $\sqrt{2/784}$, matching
$W \sim \mathcal{N}(0,\ 2/n_{\text{in}})$ with $n_{\text{in}} = d = 784$ for the first
layer, and $n_{\text{in}} = \texttt{HIDDEN} = 128$ for the second. Both bias vectors start
at all zeros, which the Concept section explains is safe (only the weights need
randomness to break symmetry).

**The training loop: epochs, shuffling, and mini-batches.** The outer loop runs once per
epoch; the inner loop slices one mini-batch at a time:

```python
for epoch in range(1, EPOCHS + 1):
    # Reuse the same rng: the shuffle sequence is deterministic across runs.
    order = rng.permutation(n)
    loss_sum = 0.0
    for start in range(0, n, BATCH):
        idx = order[start : start + BATCH]
        x, y = x_train[idx], y_train[idx]
        m = len(idx)
```

`rng.permutation(n)` returns a random reordering of the integers `0` to `n - 1` — a
shuffled index array (this is the shuffle mentioned in the Concept section's "Mini-batch
SGD" paragraph). Because `rng` was created once from a fixed `SEED` and is never
recreated, calling `.permutation(n)` again next epoch continues drawing from the same
pseudo-random sequence rather than repeating it, so every epoch sees a *different* shuffle,
yet the whole run is still reproducible from `SEED` alone. `order[start : start + BATCH]`
slices out 64 shuffled indices at a time, and `x_train[idx]` / `y_train[idx]` use that
index array to pull out those rows — "fancy indexing", selecting arbitrary rows in one
call rather than looping. The final mini-batch of an epoch may hold fewer than 64 examples
(10,000 is not a multiple of 64), which is why the code computes `m = len(idx)` instead of
assuming 64 everywhere.

**Forward pass.** Six lines, each a direct translation of an equation from the Concept
section:

```python
z1 = x @ w1 + b1                     # Z1 = X W1 + b1
a1 = np.maximum(z1, 0.0)             # A1 = ReLU(Z1)
z2 = a1 @ w2 + b2                    # Z2 = A1 W2 + b2 (logits)
z2 -= z2.max(axis=1, keepdims=True)  # softmax is shift-invariant; avoids overflow
e = np.exp(z2)
p = e / e.sum(axis=1, keepdims=True)  # Yhat = softmax(Z2)
```

`z2.max(axis=1, keepdims=True)` finds each row's largest logit; `keepdims=True` keeps the
result as an $(m, 1)$ column instead of collapsing it to a flat length-$m$ array, so the
subtraction on the next line broadcasts correctly across all 10 columns of `z2`. The last
two lines are the softmax formula $\hat{y}_j = e^{z_j} / \sum_c e^{z_c}$ applied a row at a
time: `e.sum(axis=1, keepdims=True)` sums each row's 10 exponentials, and dividing gives
`p`, the $(m, 10)$ matrix of predicted probabilities ($\hat{Y}$ in the Concept section).

**Loss.** One line, the cross-entropy formula summed over the batch:

```python
# Cross-entropy: L = -(1/m) sum_i log Yhat[i, y_i]
loss_sum += -np.log(p[np.arange(m), y]).sum()
```

`p[np.arange(m), y]` pairs up row indices `0, 1, ..., m-1` with the true label in `y` for
that row, pulling out exactly one probability per example — the probability the model
assigned to the *correct* digit. `-np.log(...)` is the per-example cross-entropy loss, and
`.sum()` adds them up (dividing by `n` to get the mean happens later, when the epoch's
average is printed).

**Backward pass.** This is the longest block in the script, and it is the code-form of
every equation derived in the Concept section's "Backpropagation" paragraph, in the same
order:

```python
y_onehot = np.zeros((m, k), dtype=np.float32)
y_onehot[np.arange(m), y] = 1.0
dz2 = (p - y_onehot) / m   # dL/dZ2 = (Yhat - Y)/m  (softmax+CE shortcut)
dw2 = a1.T @ dz2           # dL/dW2 = A1^T dL/dZ2
db2 = dz2.sum(axis=0)      # dL/db2 = column sums of dL/dZ2
da1 = dz2 @ w2.T           # dL/dA1 = dL/dZ2 W2^T
dz1 = da1 * (z1 > 0)       # dL/dZ1 = dL/dA1 * ReLU'(Z1), ReLU' = 1[Z1 > 0]
dw1 = x.T @ dz1            # dL/dW1 = X^T dL/dZ1
db1 = dz1.sum(axis=0)      # dL/db1 = column sums of dL/dZ1
```

Line by line, matched to Concept:

- `y_onehot[np.arange(m), y] = 1.0` builds the one-hot label matrix $Y$: the same
  fancy-indexing trick as the loss line, but used to *write* a 1 into column `y[i]` of row
  `i` instead of reading from it.
- `dz2 = (p - y_onehot) / m` is $\partial L/\partial Z_2 = \frac{1}{m}(\hat{Y} - Y)$,
  exactly.
- `dw2 = a1.T @ dz2` is $\partial L/\partial W_2 = A_1^\top\, \partial L/\partial Z_2$.
  `.T` transposes `a1` from $(m, 128)$ to $(128, m)$, so the matrix product with the
  $(m, 10)$ array `dz2` yields a $(128, 10)$ result — the same shape as `w2`, as every
  gradient must match the shape of the parameter it belongs to.
- `db2 = dz2.sum(axis=0)` is $\partial L/\partial \mathbf{b}_2 = \sum_i (\partial
  L/\partial Z_2)_i$: `axis=0` sums down the rows (across the batch), collapsing $(m, 10)$
  to a length-10 vector, the same shape as `b2`.
- `da1 = dz2 @ w2.T` is $\partial L/\partial A_1 = \partial L/\partial Z_2\, W_2^\top$,
  pushing the gradient back through the second linear layer.
- `dz1 = da1 * (z1 > 0)` is $\partial L/\partial Z_1 = \partial L/\partial A_1 \odot
  \mathbf{1}[Z_1 > 0]$. `(z1 > 0)` is a **boolean mask**: an array the same shape as `z1`
  holding `True`/`False`, which NumPy treats as `1.0`/`0.0` when multiplied elementwise
  (`*`) against `da1`. This zeroes the gradient wherever the hidden unit was inactive
  (recall the second row of the worked example in Concept, where `a1 = 0`) and passes it
  through unchanged wherever the unit fired — the derivative of ReLU, applied without ever
  writing an `if`.
- `dw1 = x.T @ dz1` and `db1 = dz1.sum(axis=0)` repeat the same two patterns as `dw2` and
  `db2`, one layer earlier: $\partial L/\partial W_1 = X^\top\, \partial L/\partial Z_1$
  and $\partial L/\partial \mathbf{b}_1$ as column sums.

**Parameter update.** Plain gradient descent, one line per parameter:

```python
w1 -= LR * dw1
b1 -= LR * db1
w2 -= LR * dw2
b2 -= LR * db2
```

`-=` updates each array in place: $\theta \leftarrow \theta - \eta\, \partial L/\partial
\theta$ from the Concept section, with `LR` (0.3) playing the role of $\eta$. This block
runs once per mini-batch — 157 times per epoch — not once per epoch.

**End of epoch: evaluate and print.** After all mini-batches in an epoch, the script
measures accuracy on the held-out test set and prints one summary line:

```python
acc = (predict(x_test, w1, b1, w2, b2) == y_test).mean()
print(f"epoch {epoch}  train_loss={loss_sum / n:.4f}  test_acc={acc:.4f}")
```

`predict(x_test, ...) == y_test` compares the predicted class array against the true
labels elementwise, producing a boolean array; `.mean()` on a boolean array counts the
fraction of `True` values (NumPy treats `True` as 1 and `False` as 0), which is exactly
the accuracy metric from lesson 06. `loss_sum / n` divides the summed loss by the full
10,000-example training set size to report the mean training loss for the epoch — the
`train_loss` column in the "Expected output" below.

```bash
python src/lesson11_neural_networks/from_scratch.py
```

Expected output:

```text
== Neural network: from scratch ==
architecture: 784 -> 128 (ReLU) -> 10 (softmax), 101770 parameters
train: first 10000 MNIST train images, test: first 2000 t10k images
mini-batch SGD: batch=64, lr=0.3, epochs=8

epoch 1  train_loss=0.5283  test_acc=0.8065
epoch 2  train_loss=0.2575  test_acc=0.8640
epoch 3  train_loss=0.1900  test_acc=0.9130
epoch 4  train_loss=0.1507  test_acc=0.9200
epoch 5  train_loss=0.1186  test_acc=0.9260
epoch 6  train_loss=0.0974  test_acc=0.9305
epoch 7  train_loss=0.0806  test_acc=0.9135
epoch 8  train_loss=0.0686  test_acc=0.9375

final test accuracy: 0.9375
```

Things to notice:

- **93.75% accuracy** (chance is 10%) from ~140 lines of NumPy, a sixth of the training
  data, and about a second of CPU time. The state of the art on MNIST is above 99.7%, but
  the distance from 10% to 93.75% is the part this lesson is about.
- The train loss falls **every** epoch, but test accuracy does not — it dips at epoch 7
  before recovering. SGD with a fixed learning rate bounces around the minimum, and train
  loss and test accuracy are different quantities measured on different data. This is the
  train/test gap of lesson 06, live.
- A freshly initialized 10-class model should score a loss of $-\log(1/10) \approx 2.30$
  (a classic sanity check). The printed epoch-1 average is already down to 0.5283: most of
  the learning happens in the first few hundred updates.
- The 101,770-parameter count from the Concept section and the `architecture:` line above
  is just the size of every weight matrix and bias vector added up:
  $784 \times 128 + 128 + 128 \times 10 + 10 = 100{,}352 + 128 + 1{,}280 + 10 = 101{,}770$.

## With a library

There is deliberately no `with_library.py` in this lesson: the library version of this
exact network **is** [Lesson 12](12-deep-learning-intro.md), where PyTorch's autograd
re-derives our seven hand-written gradient lines automatically, `optimizer.step()`
replaces the update loop, and the same architecture is declared in three lines of
`torch.nn`. Finish this lesson first — every line of lesson 12 corresponds to something
you just wrote by hand. Writing the backward pass by hand, as you just did in
`from_scratch.py`, is what makes the autograd version legible rather than magical: every
`.backward()` call in lesson 12 is doing the same five gradient equations you just traced
through this script's forward and backward passes, just without the code that spells
them out.

## Check your understanding

1. Delete the ReLU, so that $A_1 = Z_1$. What family of functions can the network
   represent then, and which earlier lesson's model does it essentially become?
2. Derive $\partial L / \partial \mathbf{z} = \hat{\mathbf{y}} - \mathbf{y}$ for one
   example, starting from $L = -\log \hat{y}_{\text{true}}$ and the definition of softmax.
   (Hint: treat the true-class logit and the other logits separately.)
3. Why does He initialization use variance $2/n_{\text{in}}$ rather than $1/n_{\text{in}}$?
   And why would initializing *all* weights to zero fail, when initializing only the
   biases to zero is fine?
4. The code computes `z2 -= z2.max(axis=1, keepdims=True)` before exponentiating. Why does
   this not change the softmax output, and what would happen without it for a logit of
   1000?
5. With batch size 1 you get 10,000 updates per epoch; with batch size 10,000 you get one.
   Name one advantage of each extreme, and one reason 64 is a reasonable middle ground.
6. `predict` computes `(a1 @ w2 + b2).argmax(axis=1)` and never calls `np.exp` or
   normalizes anything. Why is that a valid way to get the predicted class, and name one
   situation where you would need the actual softmax probabilities instead of just the
   arg max.
7. `rng.permutation(n)` is called once per epoch from the *same* `rng` object, rather than
   creating a fresh `np.random.default_rng(SEED)` inside the epoch loop. What would change
   about training if every epoch re-seeded the generator to the same `SEED` before
   shuffling?

From here, you're now ready to proceed to try and reproduce
[Lesson 12 — Intro to Deep Learning](12-deep-learning-intro.md).

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
