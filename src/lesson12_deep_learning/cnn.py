"""Lesson 12 — Deep learning with PyTorch: a small CNN.

Replaces the MLP's flat 784-pixel view of Fashion-MNIST with two
convolution + pooling blocks that see the image as a 28x28 grid.
Convolutions detect local patterns and reuse the same weights at every
position, so this network has ~11x fewer parameters than the MLP in
with_library.py — and a higher test accuracy. Same data, same loop,
same number of epochs: only the architecture changes.

Run from the repo root:

    python src/lesson12_deep_learning/cnn.py
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
LR = 0.25
EPOCHS = 3


def load_idx_images(path: Path) -> np.ndarray:
    """Parse an IDX image file: 16-byte header (magic, count, rows, cols as
    big-endian int32), then one uint8 per pixel. Returns shape (n, rows, cols)."""
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n, rows, cols = np.frombuffer(raw, dtype=">i4", count=4)
    assert magic == 0x00000803, f"bad magic in {path.name}: {magic:#x}"
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)


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

    # Conv2d wants (batch, channels, height, width) — one grayscale channel,
    # so shape (n, 1, 28, 28). No flattening: the CNN keeps the 2-D grid.
    x_train_t = torch.from_numpy(x_train.astype(np.float32) / 255.0).unsqueeze(1)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    x_test_t = torch.from_numpy(x_test.astype(np.float32) / 255.0).unsqueeze(1)
    y_test_t = torch.from_numpy(y_test.astype(np.int64))

    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, padding=1),   # (1, 28, 28) -> (8, 28, 28)
        nn.ReLU(),
        nn.MaxPool2d(2),                             # -> (8, 14, 14)
        nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> (16, 14, 14)
        nn.ReLU(),
        nn.MaxPool2d(2),                             # -> (16, 7, 7)
        nn.Flatten(),                                # -> 784
        nn.Linear(16 * 7 * 7, 10),                   # -> 10 logits
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Parameter count of the with_library.py MLP, for comparison:
    # (784*128 + 128) + (128*10 + 10) = 101770.
    mlp_params = 784 * 128 + 128 + 128 * 10 + 10
    cnn_params = sum(p.numel() for p in model.parameters())
    print("== Deep learning: PyTorch CNN ==")
    print("architecture: conv 1->8 (3x3) + pool -> conv 8->16 (3x3) + pool -> linear 784 -> 10")
    print(f"parameters: CNN {cnn_params} vs MLP {mlp_params}")
    print(f"train: first {N_TRAIN} Fashion-MNIST train images, test: first {N_TEST} t10k images")
    print(f"SGD: batch={BATCH}, lr={LR}, epochs={EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in loader:  # identical loop to with_library.py
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
        acc = evaluate(model, x_test_t, y_test_t)
        print(f"epoch {epoch}  train_loss={loss_sum / N_TRAIN:.4f}  test_acc={acc:.4f}")

    print()
    print(f"final test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
