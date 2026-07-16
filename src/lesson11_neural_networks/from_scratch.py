"""Lesson 11 — Neural networks, from scratch.

A two-layer MLP (784 -> 128 ReLU -> 10 softmax) trained on a subset of
MNIST with mini-batch SGD. Everything is NumPy: the forward pass, the
cross-entropy loss, and every backpropagation gradient written out by
hand — one line of code per formula in the lesson doc. Lesson 12 trains
the same network with PyTorch and lets autograd do the backward pass.

Run from the repo root:

    python src/lesson11_neural_networks/from_scratch.py
"""

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

DATA = Path(__file__).resolve().parents[2] / "data"

SEED = 42
N_TRAIN = 10_000  # first 10k of the 60k training images (deterministic subset)
N_TEST = 2_000    # first 2k of the 10k test images
HIDDEN = 128
BATCH = 64
LR = 0.3
EPOCHS = 8


def load_idx_images(path: Path) -> np.ndarray:
    """Parse an IDX image file: 16-byte header (magic, count, rows, cols as
    big-endian int32), then one uint8 per pixel. Returns shape (n, rows*cols)."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n, rows, cols = np.frombuffer(raw, dtype=">i4", count=4)
    assert magic == 0x00000803, f"bad magic in {path.name}: {magic:#x}"
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows * cols)


def load_idx_labels(path: Path) -> np.ndarray:
    """Parse an IDX label file: 8-byte header (magic, count), then uint8 labels."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n = np.frombuffer(raw, dtype=">i4", count=2)
    assert magic == 0x00000801, f"bad magic in {path.name}: {magic:#x}"
    return np.frombuffer(raw, dtype=np.uint8, offset=8)[:n]


def predict(x, w1, b1, w2, b2) -> np.ndarray:
    """Forward pass, returning the predicted class per row. Softmax is
    monotonic, so argmax over the logits equals argmax over the probabilities."""
    a1 = np.maximum(x @ w1 + b1, 0.0)
    return (a1 @ w2 + b2).argmax(axis=1)


def main() -> None:
    rng = np.random.default_rng(SEED)

    x_train = load_idx_images(DATA / "mnist-train-images-idx3-ubyte.gz")[:N_TRAIN]
    y_train = load_idx_labels(DATA / "mnist-train-labels-idx1-ubyte.gz")[:N_TRAIN]
    x_test = load_idx_images(DATA / "mnist-t10k-images-idx3-ubyte.gz")[:N_TEST]
    y_test = load_idx_labels(DATA / "mnist-t10k-labels-idx1-ubyte.gz")[:N_TEST]

    # Scale pixels from [0, 255] to [0, 1]. float32 keeps the arrays small.
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    n, d = x_train.shape  # 10000, 784
    k = 10                # digit classes 0..9

    # He initialization: W ~ N(0, 2/fan_in). The factor 2 compensates for
    # ReLU zeroing half its inputs, keeping activation variance stable.
    # Biases start at zero — the random weights already break symmetry.
    w1 = rng.standard_normal((d, HIDDEN), dtype=np.float32) * math.sqrt(2.0 / d)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    w2 = rng.standard_normal((HIDDEN, k), dtype=np.float32) * math.sqrt(2.0 / HIDDEN)
    b2 = np.zeros(k, dtype=np.float32)

    n_params = w1.size + b1.size + w2.size + b2.size
    print("== Neural network: from scratch ==")
    print(f"architecture: 784 -> {HIDDEN} (ReLU) -> 10 (softmax), {n_params} parameters")
    print(f"train: first {n} MNIST train images, test: first {len(y_test)} t10k images")
    print(f"mini-batch SGD: batch={BATCH}, lr={LR}, epochs={EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        # Reuse the same rng: the shuffle sequence is deterministic across runs.
        order = rng.permutation(n)
        loss_sum = 0.0
        for start in range(0, n, BATCH):
            idx = order[start : start + BATCH]
            x, y = x_train[idx], y_train[idx]
            m = len(idx)

            # ---- forward pass -------------------------------------------
            z1 = x @ w1 + b1                     # Z1 = X W1 + b1
            a1 = np.maximum(z1, 0.0)             # A1 = ReLU(Z1)
            z2 = a1 @ w2 + b2                    # Z2 = A1 W2 + b2 (logits)
            z2 -= z2.max(axis=1, keepdims=True)  # softmax is shift-invariant; avoids overflow
            e = np.exp(z2)
            p = e / e.sum(axis=1, keepdims=True)  # Yhat = softmax(Z2)

            # Cross-entropy: L = -(1/m) sum_i log Yhat[i, y_i]
            loss_sum += -np.log(p[np.arange(m), y]).sum()

            # ---- backward pass: the chain rule, layer by layer ----------
            y_onehot = np.zeros((m, k), dtype=np.float32)
            y_onehot[np.arange(m), y] = 1.0
            dz2 = (p - y_onehot) / m   # dL/dZ2 = (Yhat - Y)/m  (softmax+CE shortcut)
            dw2 = a1.T @ dz2           # dL/dW2 = A1^T dL/dZ2
            db2 = dz2.sum(axis=0)      # dL/db2 = column sums of dL/dZ2
            da1 = dz2 @ w2.T           # dL/dA1 = dL/dZ2 W2^T
            dz1 = da1 * (z1 > 0)       # dL/dZ1 = dL/dA1 * ReLU'(Z1), ReLU' = 1[Z1 > 0]
            dw1 = x.T @ dz1            # dL/dW1 = X^T dL/dZ1
            db1 = dz1.sum(axis=0)      # dL/db1 = column sums of dL/dZ1

            # ---- SGD update: theta <- theta - lr * dL/dtheta ------------
            w1 -= LR * dw1
            b1 -= LR * db1
            w2 -= LR * dw2
            b2 -= LR * db2

        acc = (predict(x_test, w1, b1, w2, b2) == y_test).mean()
        print(f"epoch {epoch}  train_loss={loss_sum / n:.4f}  test_acc={acc:.4f}")

    print()
    print(f"final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
