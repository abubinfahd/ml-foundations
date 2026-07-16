"""Lesson 12 — Deep learning with PyTorch: the lesson 11 MLP.

The same two-layer network as lesson 11 (784 -> 128 ReLU -> 10), now
built with torch.nn and trained on Fashion-MNIST. Nothing about the math
changed — what changed is who does the work: autograd records the
forward pass and derives every gradient we wrote by hand in lesson 11,
and the optimizer applies the SGD update. Compare with cnn.py, which
swaps the flat layers for convolutions.

Run from the repo root:

    python src/lesson12_deep_learning/with_library.py
"""

import gzip
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path(__file__).resolve().parents[2] / "data"

SEED = 42
N_TRAIN = 10_000  # first 10k of the 60k training images (deterministic subset)
N_TEST = 2_000    # first 2k of the 10k test images
BATCH = 64
LR = 0.2
EPOCHS = 3


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


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """Fraction of correct predictions. no_grad: evaluation needs no graph."""
    model.eval()
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
    return (pred == y).float().mean().item()


def main() -> None:
    torch.manual_seed(SEED)   # fixes the random weight initialization
    torch.set_num_threads(1)  # one thread => one summation order => reproducible floats

    x_train = load_idx_images(DATA / "fashion-train-images-idx3-ubyte.gz")[:N_TRAIN]
    y_train = load_idx_labels(DATA / "fashion-train-labels-idx1-ubyte.gz")[:N_TRAIN]
    x_test = load_idx_images(DATA / "fashion-t10k-images-idx3-ubyte.gz")[:N_TEST]
    y_test = load_idx_labels(DATA / "fashion-t10k-labels-idx1-ubyte.gz")[:N_TEST]

    # Scale to [0, 1] float32 (as in lesson 11), then wrap as tensors.
    x_train_t = torch.from_numpy(x_train.astype(np.float32) / 255.0)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    x_test_t = torch.from_numpy(x_test.astype(np.float32) / 255.0)
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    # DataLoader replaces lesson 11's rng.permutation(n): a seeded generator
    # makes the shuffle order identical on every run.
    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    # nn.Linear owns the W and b we allocated by hand in lesson 11 (w1, b1,
    # w2, b2). The last layer outputs raw logits — no softmax module, because
    # the loss below applies it.
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # CrossEntropyLoss = softmax + cross-entropy fused, taken on the logits.
    # Same pairing as lesson 11, same clean gradient (yhat - y) at the logits.
    loss_fn = nn.CrossEntropyLoss()
    # The optimizer replaces lesson 11's update lines: w -= LR * dw.
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print("== Deep learning: PyTorch MLP ==")
    print(f"architecture: 784 -> 128 (ReLU) -> 10, {n_params} parameters")
    print(f"train: first {N_TRAIN} Fashion-MNIST train images, test: first {N_TEST} t10k images")
    print(f"SGD: batch={BATCH}, lr={LR}, epochs={EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            logits = model(x)          # forward pass (autograd records the graph)
            loss = loss_fn(logits, y)  # scalar batch loss
            optimizer.zero_grad()      # grads accumulate in torch — reset first
            loss.backward()            # replaces lesson 11's seven hand-written gradient lines
            optimizer.step()           # SGD update: p -= lr * p.grad for every parameter
            loss_sum += loss.item() * len(y)
        acc = evaluate(model, x_test_t, y_test_t)
        print(f"epoch {epoch}  train_loss={loss_sum / N_TRAIN:.4f}  test_acc={acc:.4f}")

    print()
    print(f"final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
